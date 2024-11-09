import os
import cv2
import json
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

#camera specs
'''
{
  "camera_angle_x": 0.9469152137575624,"camera_angle_y": 0.4718488425749604,
  "fl_x": 1249.2110972577298, "fl_y": 1497.4958567908102,
  "k1": 0.04383280598405508, "k2": -0.13180026880820486, "k3": 0, "k4": 0,
  "p1": 0.002400169988152647, "p2": -0.0005919284654390464, "is_fisheye": false,
  "cx": 629.1203046136077, "cy": 333.43930530402406, "w": 1280.0, "h": 720.0, "aabb_scale": 16,
  "frames": [
    {
      "file_path": "./images/0014.jpg",
      "sharpness": 209.6658102442318,
      "transform_matrix": [
        [-0.21770364897213842,0.035770173854138186,-0.9753592240274488,-3.2431310689807606],
        [-0.975922368269063,0.005783834263049029,0.21804146021646859,1.515309282752658],
        [0.01344069703806407,0.9993433053422831,0.03364975095701072,-0.009758798570238534],
        [0.0,0.0,0.0,1.0]
      ]
    }
  ]
}
'''

#input normalized to be in [-1,1]
#x=torch.tensor([0.5, -0.2, 0.8])
#x=torch.cat([posencodings(xi.unsqueeze(0),10) for xi in x], dim=-1)
#according to the paper for the location it has to be in the range of 10

def posencodings(L,p):
    encodings=[]
    for i in range(L):
        encodings.append(torch.sin(np.power(2,i)*np.pi*p))
        encodings.append(torch.cos(np.power(2,i)*np.pi*p))
    return torch.cat(encodings,dim=-1)

#goal? to project out rays from a camera onto a 3d scene thru each pixel of a 2d image
#this is done to produce the direction of ray that passes thru each pixel on the image
#the camera's position is taken to be the origin and each pixel in the image corresponds to a different ray

def projectrays(height,width,camera_angle_x,transform_matrix):
    fclen=0.5*width/np.tan(0.5*camera_angle_x)
    count=0
    cols=np.zeros([height,width],dtype=float)
    rows=np.zeros([height,width],dtype=float)
    zdir=np.ones([height,width],dtype=float)
    for i in range(len(cols)):
        for j in range(len(cols[i])):
            cols[i,j]=count
        count+=1
    count=0
    for i in range(len(rows)):
        for j in range(len(rows[i])):
            rows[i,j]=count
            count+=1
        count=0
    for i,j in cols:
        cols[i,j]=(cols[i,j]-height/2)/fclen
    for i,j in rows:
        rows[i,j]=(rows[i,j]-height/2)/fclen    
    rays=np.zeros([height,width,3],dtype=float)
    for i in range(height):
        for j in range(width):
            rays[i,j,0]=cols[i,j]
            rays[i,j,1]=rows[i,j]
            rays[i,j,2]=zdir[i,j]
    #this matrix relates to the extrinsic parameters of the camera which describe its orientation and position in 3D space
    transform_matrix=np.array(transform_matrix)
    #tranform the directional rays from being relative to the camera to the 3D world
    raysactual=np.sum(rays[...,np.newaxis,:]*transform_matrix[:3,:3],axis=-1)
    #to make sure the origin is the same for all rays
    raysfromorigin=np.broadcast_to(transform_matrix[:3,3],raysactual)
    return raysfromorigin,raysactual

def dataloader(frames):
    images=[]
    rotations=[]
    transform_matrices=[]
    for frame in frames:
        file_path=(os.path.join(os.path.join("nerf_synthetic/drums",frame["file_path"][1:]),r'.png'))
        image=cv2.imread(file_path)
        images.append(image)
        rotations.append(frame["rotation"])
        transform_matrices.append(frame["transform_matrix"])
    return images,rotations,transform_matrices


def volumerender(rgb,sigma,sampledpts,ray):
    diffs=sampledpts[...,1:]-sampledpts[...,:-1]
    diffs=np.concatenate([diffs,1e10*np.ones_like(diffs[...,:1])])
    alpha=1.0-np.exp(diffs*sigma*-1)
    transmittance=np.cumprod(np.concatenate([np.ones((alpha.shape[0],1)),1.0-alpha+1e-10],dim=-1),dim=-1)[:,:-1]
    return np.sum(transmittance*rgb*alpha.unsqueeze(-1),dim=0)
    

def getimage(height,width,fclen,start,end,samples,rgb,sigma):
    rays=projectrays(height,width,fclen)
    matrix=np.zeros((height,width,3))
    for i in range(height):
        for j in range(width):
            sampledpts=np.linspace(start,end,samples)
            matrix[i,j]=volumerender(rgb,sigma,sampledpts,rays[i,j])
    return matrix

def hasher(ind,size):
    return ind%size

def interpolation(features,fract):
    x=fract[:,0].unsqueeze(1)*features[:,1]+(1-fract[:,0].unsqueeze(1))*features[:,0]
    y=fract[:,1].unsqueeze(1)*features[:,3]+(1-fract[:,1].unsqueeze(1))*features[:,2]
    z=fract[:,2].unsqueeze(1)*features[:,5]+(1-fract[:,2].unsqueeze(1))*features[:,4]
    return (x+y+z)/3

class multiresolutionhashencoding(nn.Module):
    def __init__(self,levels,features,sizeoftable):
        super().__init__()
        self.levels=levels
        self.features=features
        self.sizeoftable=sizeoftable
        #for every resolution varying from coarser to the finer ones, hash tables are created
        self.tables=nn.ModuleList([nn.Parameter(torch.randn(sizeoftable,features)) for i in range(levels)])
    def forward(self,x):
        out=[]
        for level in range(self.levels):
            gridres=2**(level+1)
            voxelcoords=np.floor(x*gridres).long()
            fract=x*gridres-voxelcoords
            outhash=hasher(voxelcoords.sum(dim=1),self.sizeoftable)
            voxelfeat=self.tables[level][outhash]
            outinterpolate=interpolation(voxelfeat,fract)
            out.append(outinterpolate)
        return torch.cat(out,dim=1)        

class flow(nn.Module):
    def __init__(self):
        super().__init__()
        self.fcin=nn.Linear(5,256)
        self.fcmiddle=nn.Linear(256,256)
        self.fcout=nn.Linear(256,257)
    def forward(self,x):
        x=F.relu(self.fcin(x))
        for i in range(0,7): x=F.relu(self.fcmiddle(x))
        x=F.relu(self.fcout(x))
        return x

class end(nn.Module):
    def __init__(self):
        super().__init__()
        self.fcin=nn.Linear(256,128)
        self.fcout=nn.Linear(128,3)
    def forward(self,x):
        x=F.relu(self.fcin(x))
        x=F.relu(self.fcout(x))
        return x
dataset=[]
flower=flow()
ender=end()
optimizer=optim.sgd(flower.parameters(),lr=0.0001)
criterion=nn.CrossEntropyLoss()
for data in dataset:
    for image,theta,phi in data:
        for R,G,B in image:
            outflow=flower(R,G,B,theta,phi)
            sigma=outflow[-1]
            outend=ender(outflow[:-1])
            var=getimage()
    #routine volume rendering
    optimizer.zero_grad()
    loss=criterion(var,image)
    loss.backward()
    optimizer.step()


def loadjson(path):
    with open(path,'r') as file:
        data=json.load(file)
    camera_angle_x=data['camera_angle_x']
    frames=data['frames']
    return camera_angle_x,frames

def retrievecamerainfo(path):
    with open(path,'r') as file:
        data=json.load(file)
    camera_angle_y=data["camera_angle_y"]
    fl_x=data["fl_x"]
    fl_y=data["fl_y"]
    k1=data["k1"]
    k2=data["k2"]
    k3=data["k3"]
    k4=data["k4"]
    p1=data["p1"]
    p2=data["p2"]
    is_fisheye=data["is_fisheye"]
    cx=data["cx"]
    cy=data["cy"]
    w=data["w"]
    h=data["h"]
    aabb_scale=data["aabb_scale"]
    return camera_angle_y,fl_x,fl_y,k1,k2,k3,k4,p1,p2,is_fisheye,cx,cy,w,h,aabb_scale

dataset=np.ndarray([])


#push forward the network and then generate the RGBA values to perfrom volume rendering over and eventually gradient descent

def stratifiedsampling(oray,dray,samples,near,far):
    tvals=torch.linspace(near,far,samples)
    tvals=tvals+torch.rand(tvals.shape)*(far-near)/samples
    pts=oray[...,None,:]+dray[...,None,:]*tvals[...,:,None]
    return pts,tvals

def importancesampling(tcoarse,weightscoarse,samples):
    pdf=weightscoarse/torch.sum(weightscoarse,dim=-1,keepdim=True)
    cdf=torch.cumsum(pdf,dim=-1)
    cdf=torch.cat([torch.zeros_like(cdf[..., :1]),cdf],dim=-1)
    u=torch.rand(list(cdf.shape[:-1])+[samples])
    inds=torch.searchsorted(cdf, u, right=True)
    tfine=torch.gather(tcoarse,-1,inds)
    return tfine

def volumerendering(rgb, sigma, t_vals, deltas):
    alpha=1.0-torch.exp(-sigma*deltas)
    transmittance=torch.cumprod(1.0-alpha+1e-10,dim=-1)
    weights=alpha*transmittance
    renderedrgb=torch.sum(weights.unsqueeze(-1)*rgb,dim=-2)
    return renderedrgb

def train_nerf(epochs,batchsz,coarsenet,finenet,optimizer,images,transforms,height,width,camera_angle_x):
    for epoch in range(epochs):
        totloss=0.
        for batch in range(0,len(images),batchsz):
            optimizer.zero_grad()
            currimgs=images[batch:batch+batchsz]
            currtransforms=transforms[batch:batch+batchsz]
            orayall=[]
            drayall=[]
            for transform in currtransforms:
                oray,dray=projectrays(height,width,camera_angle_x,transform)
                orayall.append(oray)
                drayall.append(dray)
            oraycurr=torch.stack(orayall)
            draycurr=torch.stack(drayall)
            coarsepts,tcoarse=stratifiedsampling(oraycurr,draycurr,samples=64,near=2.0,far=6.0)
            coarseout=coarsenet(coarsepts)
            rgbcoarse,sigmacoarse=coarseout[..., :3], coarseout[..., 3]
            deltas=torch.ones_like(tcoarse)*(1.0 / 64)
            weightscoarse=volumerendering(rgbcoarse,sigmacoarse,tcoarse, deltas)
            finepts=importancesampling(tcoarse, weightscoarse,samples=128)
            fineout=finenet(finepts)
            rgbfine,sigmafine=fineout[..., :3],fineout[..., 3]
            renderedpixels=volumerendering(rgbfine,sigmafine, tcoarse,deltas)
            loss=torch.mean((renderedpixels-currimgs)**2)
            loss.backward()
            optimizer.step()
            totloss+=loss.item()
        print(f"Epoch {epoch}/{epochs}, loss={totloss}")
    model=()
    torch.save(model.state_dict(),'params.pth')


class custom_mlp(nn.Module):
    def __init__(self,hashencoding,innerdim,outdim):
        super(custom_mlp,self).__init__()
        self.hashencoding=hashencoding
        indim=hashencoding.levels*hashencoding.features
        self.fcin=nn.Linear(indim,innerdim)
        self.fcmid=nn.Linear(innerdim,innerdim)
        self.fcout=nn.Linear(innerdim,outdim)
    def forward(self,x):
        x=F.relu(self.fcin(self.hashencoding(x)))
        x=F.relu(self.fcmid(x))
        x=self.fcout(x)
        return x


if __name__=="__main__":
    #load the transforms.json file and extract the dataset
    camera_angle_x,frames=loadjson(r"nerf_synthetic\drums\transforms_train.json")
    camera_angle_y,fl_x,fl_y,k1,k2,k3,k4,p1,p2,is_fisheye,cx,cy,width,height,aabb_scale=retrievecamerainfo(r"camera_info.json")
    images,rotations,transforms=dataloader(frames=frames)
    #push the relevant information through the network
    #do gradient descent over the batch
    #continue training over the batch
    hashencoding=multiresolutionhashencoding(levels=16,sizeoftable=2**19,features=2)
    coarsenet=custom_mlp(hashencoding)
    finenet=custom_mlp(hashencoding)
    optimizer=optim.Adam(list(coarsenet.parameters())+list(finenet.parameters()),lr=1e-3)
    train_nerf(epochs=1000,batchsz=32,coarse=coarsenet,fine=finenet,optimizer=optimizer,images=images,transforms=transforms,height=height,width=width,camera_angle_x=camera_angle_x)
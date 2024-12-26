import os
import cv2
import json
import torch
import numpy as np
from tqdm import tqdm
import torch.nn as nn
from torch.utils.data import DataLoader

class MLP(nn.Module):
    def __init__(self,position_dim=10,direction_dim=4,inner_dim=128):
        
        super(MLP,self).__init__()

        # Individual Layers and utilities that construct the NeRF model

        self.inner_dim=inner_dim
        self.position_dim=position_dim
        self.direction_dim=direction_dim

        self.sigmoid=nn.Sigmoid()
        self.relu=nn.ReLU()

        self.layer11=nn.Linear(position_dim*6+3,inner_dim)
        self.layer12=nn.Linear(position_dim*6+inner_dim+3,inner_dim)
        self.layer13=nn.Linear(direction_dim*6+inner_dim+3,inner_dim//2)
        self.layer21=nn.Linear(inner_dim,inner_dim+1)
        self.layer2=nn.Linear(inner_dim,inner_dim)
        self.out=nn.Linear(inner_dim//2,3) 
        '''
        
        Various combinations of the above layers are used to build blocks that encapsulate the entirety of the NeRF network.

        For example: After the first 4 layers, an additional input is given to the network. 

        This pattern of chipping away certain values and contracting the size of the network towards the end is visible in the forward method.

        '''

    @staticmethod
    def positional_encoding(x,L):
        
        # γ(p) = (sin((2^0)*πp), cos((2^0)*πp), ··· , sin((2^L−1)*πp), cos((2^L−1)*πp))
        # γ(p) is applied to the 3 coordinate values

        encodings=[x]
        
        for i in range(L):

            encodings.append(torch.sin(2**i*x))
            encodings.append(torch.cos(2**i*x))

        return torch.cat(encodings,dim=1)
    
    def forward(self,pos,dir):
        
        # The 5 dimensional input (3 for the position and 2 for the direction) must be mapped to the following two: 
        # 1. 63 dimension feature vector (for the position)
        # 2. 24 dimension feature vector (for the direction)

        # The position vector's (3D vector) feature vector is computed
        embeddings_position=self.positional_encoding(pos,self.position_dim)

        # The direction vector's (2D vector) feature vector is computed
        embeddings_direction=self.positional_encoding(dir,self.direction_dim)

        embeddings_position=embeddings_position.float()
        embeddings_direction=embeddings_direction.float()

        block1=self.layer11(embeddings_position)
        block1=self.relu(block1)

        block1=self.layer2(block1)
        block1=self.relu(block1)
        
        block1=self.layer2(block1)
        block1=self.relu(block1)
        
        block1=self.layer2(block1)
        block1=self.relu(block1)

        # Density estimation 

        block2=self.layer12(torch.cat((block1,embeddings_position),dim=1))
        block2=self.relu(block2)
        
        block2=self.layer2(block2)
        block2=self.relu(block2)
        
        block2=self.layer2(block2)
        block2=self.relu(block2)
        
        # An additional value to corresponding to sigma
        block2=self.layer21(block2)
        
        # Chip away the end for the volume opacity(sigma)
        block1,sigma=block2[:,:-1],self.relu(block2[:,-1])

        # The rest of the network tries to approximate the color of that particular coordinate in space viewed from that particular direction.
        block1=self.layer13(torch.cat((block1,embeddings_direction),dim=1))
        block1=self.relu(block1)

        # Output the RGB values
        rgb=self.out(block1)
        rgb=self.sigmoid(rgb)

        return rgb,sigma
    
def accumulated_transmittance(alphas):

    '''
    
    Transmittance represents the fraction of light that is not absorbed along the ray upto the point under scrutiny.

    The cumulative product of the elements of the tensor alpha is the accumulated transmittance until that point. A leading transmittance of 1 is prepended to the tensor. 
    
    ''' 

    accumulatedtransmittance=torch.cumprod(alphas,1)

    toprependwithone=torch.ones((accumulatedtransmittance.shape[0],1),device=alphas.device)

    return torch.cat((toprependwithone,accumulatedtransmittance[:,:-1]),dim=-1)


def render_rays(model,origin_rays,ray_directions,near=0,far=0.5,bins=192):

    '''

    The idea is to project out rays from a camera from different viewing angles that points toward the object/scene. 

    A set number of points are sampled along each of these rays and the network tries to predict the color of the scene/object at all of these points in space.

    For a given dataset, from the many viewing angles that the images are shot, every pixel on every image in the dataset corresponds to a ray whose individually sampled points' RGBA (rgb and sigma(volume opacity)) are learnt over time.

    '''

    device=origin_rays.device

    # Sample points along each ray
    t=torch.linspace(near,far,bins,device=device).expand    (origin_rays.shape[0],bins)
    '''

    TO STRATIFY THE SAMPLE:

    The sampled points could be directly used as such, but we stratify this sample (modify and adjust the position of the points in space). This is done to add a little bit of randomness to the system. Moreover a random pertubation is applied which moves the point along the ray a bit to the right or left.
    
    Example: t=[[0.0,0.25,0.5,0.75,1.0]]

    the center of any 2 points sampled in 't' is computed and that forms the following tensor 
    midpts=[[0.125,0.375,0.625,0.875]]

    lower is a tensor with one additional element i.e. the first sampled point: lower=[[0.0,0.125,0.375,0.625,0.875]]
    
    upper is a tensor with one additional element i.e. the last sampled point: upper=[[0.125,0.375,0.625,0.875,1.0]]
    
    '''
    midpts=(t[:,:-1]+t[:,1:])/2
    lower=torch.cat((t[:,:1],midpts),-1)
    upper=torch.cat((midpts,t[:,-1:]),-1)

    # Random Pertubation
    abit=torch.rand(t.shape,device=device)
    t=lower+(upper-lower)*abit
    
    # Delta contains the intervals that are used in volume rendering. 10**10 is appeneded to the end to point to the last point in the sample.
    delta=torch.cat((t[:,1:]-t[:,:-1],torch.tensor([1e10],device=device).expand(origin_rays.shape[0],1)),-1)

    '''
    
    Compute the 3D points along each ray:

    x(t)=o+t*d

    x(t): 3D point in space along the ray at a depth of d.
    o: The ray from the origin.
    d: The direction of the ray.
    t: The sampled points.

    The tensors are relevantly adjusted in their dimensional sizes.

    x is finally of the shape: [batchsize,bins,3] (the 3 represents the 3D coordinates of the point)

    '''
    
    x=origin_rays.unsqueeze(1)+t.unsqueeze(2)*ray_directions.unsqueeze(1)

    # Adjust the dimension of ray_directions to that of x
    ray_directions=ray_directions.expand(bins,ray_directions.shape[0],3).transpose(0,1)

    # Query the model for RGBA output.
    rgb,sigma=model(x.reshape(-1,3),ray_directions.reshape(-1,3))

    rgb=rgb.reshape(x.shape)    
    sigma=sigma.reshape(x.shape[:-1])

    '''

    Once the MLP has generated the RGBA values for the sampled points we must now take integrate the color and opacity along every ray in the batch to determine the rgb value on the 2D image.

    C(r)=summation(T(i)*(1-exp(-sigma(i)*delta(i))))
    T(i)=exp(-summation(sigma(j)*delta(j)))

    delta(i)=t(i+1)-t(i)
    alpha(i)=1-exp(-sigma(i)*delta(i)) shape: [batchsize,bins]

    Accumulated transmittance represents the amount of light that reaches that particular point without being blocked by something else before that point. 
    
    To compute the accumulated transmittance for every point, we use T(i)=exp(-summation(sigma(j)*delta(j)))

    Each point in space now has a weight that represents its contribution to the actual 2D image's pixels' RGB value.

    This weight is a combination of the accumulated transmittance and the alpha values.
    
    Finally, the pixel color is computed as a sum of the weights product with the rgb value at that point in space.

    For rays that do not interact with the 3D object in question they must render a constant background color as their pixel color.

    '1-weightsum' represents the unblocked light fraction or the contribution to the background

    '''    
    
    alpha=1-torch.exp(-sigma*delta) 
    weights=accumulated_transmittance(1-alpha).unsqueeze(2)*alpha.unsqueeze(2)

    pixelrgb=(weights*rgb).sum(dim=1)
    weightsum=weights.sum(-1).sum(-1)

    return pixelrgb+1-weightsum.unsqueeze(-1)

def train(model,optimizer,scheduler,dataloader,device='cpu',bins=192,height=800,width=800,near=0,far=1,epochs=10000):

    trainingloss=[]
    for epoch in tqdm(range(epochs)):
        for batch in tqdm(dataloader):
            
            #The first 3 columns correspond to the origin for every ray in the batch, the following 3 columns correspond to the directions of every ray in the batch.
            
            origin_rays=batch[:,:3].to(device)
            ray_directions=batch[:,3:6].to(device)

            actualpixelvals=batch[:,7:].to(device)

            predictedpixelvals=render_rays(model,origin_rays,ray_directions,near=near,far=far,bins=bins)

            # MSE Loss over the rendered color and actual color.

            loss=((actualpixelvals-predictedpixelvals)**2).sum()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            trainingloss.append(loss.item())
        
        scheduler.step()
    return trainingloss

if __name__=="__main__":
    
    torch.set_default_dtype(torch.float32)
    device='cuda'
    dataset=torch.from_numpy(np.load('training_data.pkl',allow_pickle=True))
    model=MLP(inner_dim=256).to(device=device)
    
    '''
    The hyperparameters used and implementation details have been taken from the research paper. 
    '''

    optimizer=torch.optim.Adam(model.parameters(),lr=5e-4)
    scheduler=torch.optim.lr_scheduler.MultiStepLR(optimizer=optimizer,milestones=[2,4,8],gamma=0.5)
    dataloader=DataLoader(dataset=dataset,batch_size=1024,shuffle=True)
    train(model,optimizer,scheduler,dataloader,epochs=20,device=device,near=2,far=6,bins=192,height=400,width=800)

import os
import cv2
import json
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

def hasher(ind,size): return ind%size

def interpolation(features,fractionalPart):

    x=fractionalPart[:,0].unsqueeze(1)*features[:,1]+(1-fractionalPart[:,0].unsqueeze(1))*features[:,0]
    
    y=fractionalPart[:,1].unsqueeze(1)*features[:,3]+(1-fractionalPart[:,1].unsqueeze(1))*features[:,2]
    
    z=fractionalPart[:,2].unsqueeze(1)*features[:,5]+(1-fractionalPart[:,2].unsqueeze(1))*features[:,4]

    return (x+y+z)/3

class mlp(nn.Module):

    def __init__(self,hashencoding,hiddenLayer,finalLayer):
    
        super(mlp,self).__init__()

        # Apply a multi-resolution hash encoding to the input.
        self.hashencoding=hashencoding
        inputDim=hashencoding.levels*hashencoding.features
        
        # Unique layers in the MLP.
        self.fcin=nn.Linear(inputDim,hiddenLayer)
        self.fcmid=nn.Linear(hiddenLayer,hiddenLayer)
        self.fcout=nn.Linear(hiddenLayer,finalLayer)
    
    def forward(self,x):

        x=F.relu(self.fcin(self.hashencoding(x)))
        x=F.relu(self.fcmid(x))
        x=self.fcout(x)
        
        return x

class multiResolutionHashEncoding(nn.Module):

    def __init__(self,levels,features,sizeOfTable):
        
        super().__init__()
        
        self.levels=levels
        self.features=features
        self.sizeOfTable=sizeOfTable
        
        #for every resolution varying from coarser to the finer ones, hash tables are created
        self.tables=nn.ModuleList([nn.Parameter(torch.randn(sizeOfTable,features)) for i in range(levels)])
        self.tables=nn.ParameterList(self.tables)
    
    def forward(self,x):
        
        # Initialize the result and Iterate over all the levels.
        result=[]
        
        for level in range(self.levels):

            gridResolution=2**(level+1)
            
            # Get the voxel that encloses that 3D space point.
            voxelCoordinates=np.floor(x*gridResolution).long()
            fractionalPart=x*gridResolution-voxelCoordinates

            # Get a feature vector after hashing the voxel coordinates.
            hashedResult=hasher(voxelCoordinates.sum(dim=1),self.sizeOfTable)
            voxelFeatures=self.tables[level][hashedResult]
            
            # Interpolate to smoothen the result and append.
            interpolatedResult=interpolation(voxelFeatures,fractionalPart)
            result.append(interpolatedResult)
        
        return torch.cat(result,dim=1)      

def loadjson(path):

    # Load the metadata from the transforms.json file.
    with open(path,'r') as file:
        data=json.load(file)

    camera_angle_x=data['camera_angle_x']
    frames=data['frames']

    return camera_angle_x,frames

def retrieveCameraInfo(path):

    # Relevant Camera Information loaded into memory.
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
    aabb_scale=data["aabb_scale"]
    cx=data["cx"]
    cy=data["cy"]
    w=data["w"]
    h=data["h"]
    
    return camera_angle_y,fl_x,fl_y,k1,k2,k3,k4,p1,p2,is_fisheye,cx,cy,w,h,aabb_scale

def dataloader(frames):

    # Initialize Data Matrices.
    images=[]
    rotations=[]
    transform_matrices=[]

    # Iterate over every frame in frames, to get data. 
    for frame in frames:

        file_path=(os.path.join("nerf_synthetic/drums/",frame["file_path"][1:].lstrip("/"))+'.png')
        image=cv2.imread(file_path)

        images.append(image)
        rotations.append(frame["rotation"])
        transform_matrices.append(frame["transform_matrix"])

    return images,rotations,transform_matrices

# Stratified samples and the use of PDF's for Sampling Important points.

def stratifiedSampling(originOfRay,directionOfRay,numberOfSamples,nearestPoint,farthestPoint):

    # Add jitter to reduce anti-aliasing and to acheive smoother rendering.
    # Gives the distance from the origin for that sample.
    sampledDistances=torch.linspace(nearestPoint,farthestPoint,numberOfSamples)
    sampledDistances+=torch.rand(sampledDistances.shape)*(farthestPoint-nearestPoint)/numberOfSamples

    # Increament in Dimensions of tensors to obtain one with the 3D locations of the sampled points.
    sampledPoints=originOfRay[..., None, :]+directionOfRay[..., None, :]*sampledDistances[..., :, None]

    return sampledPoints,sampledDistances

def importanceSampling(coarse,coarseWeights,numberOfSamples):

    # A PDF to convey the location of the more important samples.
    probabilityDistributionFunction=coarseWeights/torch.sum(coarseWeights,dim=-1,keepdim=True)

    # A CDF to determine the intervals where the samples will lie starting from 0.
    cumulativeDistributionFunction=torch.cumsum(probabilityDistributionFunction,dim=-1)
    cumulativeDistributionFunction=torch.cat([torch.zeros_like(cumulativeDistributionFunction[..., :1]),cumulativeDistributionFunction],dim=-1)

    # Search over the CDF using the values of a randomly generated tensor of numbers to store the matching index in indices. 
    indices=torch.searchsorted(cumulativeDistributionFunction,torch.rand(list(cumulativeDistributionFunction.shape[:-1])+[numberOfSamples]),right=True)

    # Collect the relevant values of the coarseNet's Sampled points using indices.
    fineSamples=torch.gather(coarse,-1,indices)

    return fineSamples

    

def volumeRendering(rgbValues,volumeOpacity,distanceIntervals):

    # Alpha encapsulates the contribution of color from each point in the 3D space based on the density at that point.
    alpha=1.0-torch.exp(-1*volumeOpacity*distanceIntervals)

    # How much light can actually reach that point based off of alpha.
    transmittance=torch.cumprod(1.0-alpha+1e-10,dim=-1)

    # The effect of each point on the final color is given by this tensor.
    weights=alpha*transmittance

    # Blend all the colors along the ray 
    pixels=torch.sum(weights.unsqueeze(-1)*rgbValues,dim=-2)

    return pixels

def projectThoseRaysFromEveryPixelASAP(height,width,camera_angle_x,transform):

    # Calculate the focal length initialize coordinate matrices.
    fclen=0.5*width/torch.tan(0.5*camera_angle_x)
    
    zdir=torch.ones([height,width],dtype=float)
    rows=torch.zeros([height,width],dtype=float)
    cols=torch.zeros([height,width],dtype=float)
    
    
    # Create grids.
    count=0
    
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
    
    # Initialize a matrix to hold the direction of the rays and copy the values relevantly.
    rays=torch.zeros([height,width,3],dtype=float)
    
    for i in range(height):
        for j in range(width):
            rays[i,j,0]=cols[i,j]
            rays[i,j,1]=rows[i,j]
            rays[i,j,2]=zdir[i,j]
    
    # This matrix tells us about the extrinsic parameters of the camera which describe its orientation and position in 3D space.
    transform=torch.array(transform)

    # Tranform the directional rays from being relative to the camera to the general 3D space.
    raysIn3DSpace=torch.sum(rays[...,torch.newaxis,:]*transform[:3,:3],axis=-1)

    # Make sure the origin is the same for all rays.
    raysFromOrigin=torch.broadcast_to(transform[:3,3],raysIn3DSpace)

    return raysFromOrigin,raysIn3DSpace


def trainNerf(epochs,batchSize,coarseNet,fineNet,optimizer,images,transforms,height,width,camera_angle_x):
    
    for epoch in range(epochs):
        
        # Set loss to 0.
        loss=0.0
        
        for batch in range(0,len(images),batchSize):

            # Reset the Gradients to zero.
            optimizer.zero_grad()
            
            # Batch Data. 
            currentImages=images[batch:batch+batchSize]
            curentTransformations=transforms[batch:batch+batchSize]

            # For all pixels in the batch, get the directions and origin of each ray generated from that pixel
            alldirections=[]
            allorigins=[]

            for transform in curentTransformations:
                rayOrigin,rayDirection=projectThoseRaysFromEveryPixelASAP(height=height,width=width,camera_angle_x=camera_angle_x,transform=transform)
                alldirections.append(rayDirection)
                allorigins.append(rayOrigin)
            
            # Create a tensor off of the appended values.
            alldirections=torch.stack(alldirections)
            allorigins=torch.stack(allorigins)

            # After the generation of the sampled points, Query the Coarse Network.
            sampledPoints,sampledDistances=stratifiedSampling(originOfRay=allorigins,directionOfRay=alldirections,numberOfSamples=64,nearestPoint=2.0,farthestPoint=6.0)
            coarseOut=coarseNet(sampledPoints)
            
            # Decode the RGBA values from the Coarse Network.
            rgbCoarse,sigmaCoarse=coarseOut[..., :3],coarseOut[..., 3]

            # Create a interval based tensor spaced by the sampled points.
            deltas=torch.ones_like(sampledDistances)*(1.0 / 64)

            # VolumeRendering to generate the Pixel value for this ray (Coarse Network samples).
            weightsCoarse=volumeRendering(rgbCoarse,sigmaCoarse,deltas)

            # Sample the Fine points based on the PDF's & CDF's to Query the Fine Network. 
            finePoints=importanceSampling(coarse=sampledDistances,coarseWeights=weightsCoarse,numberOfSamples=128)
            fineOut=fineNet(finePoints)

            # Decode the RGBA values from the Fine Network.
            rgbfine,sigmafine=fineOut[..., :3],fineOut[..., 3]

            # VolumeRendering to generate the Pixel value for this ray (Fine Network samples).
            renderedPixels=volumeRendering(rgbfine,sigmafine,deltas)
            
            # Calculate Loss.
            currentLoss=torch.mean((renderedPixels-currentImages)**2)

            # Backwards computational graph to optimize over.
            currentLoss.backward()
            optimizer.step()

            loss+=currentLoss.item()

        print(f"Epoch : {epoch+1}/{epochs} ||| Loss : {loss:.3f}")

    torch.save({'coarseNetStateDict':coarseNet.state_dict(),'fineNetStateDict':fineNet.state_dict(),'optimizerStateDict':optimizer.state_dict()},'model.pth')


if __name__=="__main__":

    # Path to relevant files.
    pathToTransforms=r"nerf_synthetic\drums\transforms_train.json"
    pathToCameraInfo=r"camera_info.json"

    # Load the json files.
    camera_angle_x,frames=loadjson(path=pathToTransforms)
    camera_angle_y,fl_x,fl_y,k1,k2,k3,k4,p1,p2,is_fisheye,cx,cy,width,height,aabb_scale=retrieveCameraInfo(path=pathToCameraInfo)
    images,rotations,transforms=dataloader(frames=frames)
    
    # Instantiate the Coarse and Fine networks.
    hashencoding=multiResolutionHashEncoding(levels=16,sizeOfTable=2**19,features=2)
    coarseNet=mlp(hashencoding=hashencoding)
    fineNet=mlp(hashencoding=hashencoding)
    
    # Train!!!.
    optimizer=optim.Adam(list(coarseNet.parameters())+list(fineNet.parameters()),lr=1e-3)
    trainNerf(epochs=1000,batchSize=32,coarseNet=coarseNet,fineNet=fineNet,optimizer=optimizer,images=images,transforms=transforms,height=height,width=width,camera_angle_x=camera_angle_x)
import os
import cv2
import json

def loadjson(path):

    # Load the metadata from the transforms.json file.
    with open(path,'r') as file:
        data=json.load(file)

    camera_angle_x=data['camera_angle_x']
    frames=data['frames']

    return camera_angle_x,frames

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


def retrievecamerainfo(path):

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


if __name__=="__main__":

    # Path to relevant files.
    pathToTransforms=r"nerf_synthetic\drums\transforms_train.json"
    pathToCameraInfo=r"camera_info.json"

    # Load the json files.
    camera_angle_x,frames=loadjson(path=pathToTransforms)
    camera_angle_y,fl_x,fl_y,k1,k2,k3,k4,p1,p2,is_fisheye,cx,cy,width,height,aabb_scale=retrievecamerainfo(path=pathToCameraInfo)
    images,rotations,transforms=dataloader(frames=frames)
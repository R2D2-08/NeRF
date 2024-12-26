import os
import json
import pickle
import numpy as np
from PIL import Image
from pathlib import Path
from tqdm import tqdm
def compute_intrinsic_params(camera_angle_x,width):
    focal_length=0.5*width/np.tan(0.5*camera_angle_x)
    return np.array([[focal_length,0,width/2],[0,focal_length,width/2],[0,0,1]])

def computerays(img_path, transform_matrix, intrinsics):

    # Normalizing the RGB values
    img=np.array(Image.open(img_path))/255.0 
    height,width,_=img.shape

    rays=[]
    for i in tqdm(range(height)):
        for j in range(width):
            pixel_coord=np.array([j + 0.5, i + 0.5, 1.0])
            camera_coord=np.linalg.inv(intrinsics) @ pixel_coord
            world_coord=transform_matrix[:3, :3] @ camera_coord + transform_matrix[:3, 3]
            ray_origin=transform_matrix[:3, 3]
            ray_direction=world_coord-ray_origin
            ray_direction=ray_direction/np.linalg.norm(ray_direction)
            pixel_color=img[i, j]
            rays.append([*ray_origin,*ray_direction,*pixel_color])
    return np.array(rays)

# Load transforms.json
with open(r'transforms_test.json','r') as file:
    transforms=json.load(file)

camera_angle_x=transforms['camera_angle_x']
frames=transforms['frames']

dataset=[]
for frame in tqdm(frames):
    img_path=frame['file_path']
    flie_path=Path(img_path)
    fixed_path=flie_path.with_suffix(".png")
    transform_matrix=np.array(frame['transform_matrix'])
    image=np.array(Image.open(fixed_path))
    height,width,_=image.shape
    intrinsics=compute_intrinsic_params(camera_angle_x,width)
    rays=computerays(fixed_path,transform_matrix,intrinsics)
    dataset.append(rays)

dataset=np.concatenate(dataset,axis=0)

with open('testing_data.pkl','wb') as file:
    pickle.dump(dataset,file)

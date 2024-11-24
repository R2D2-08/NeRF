# Neural Radiance Fields with a Multi-Resolution Hash Encoding

This repository was supposed to be a successful implementation of a Neural Radiance Field pipeline with a Multiresolution Hash Encoding but it is NOT that, the logic involved in the research papers has been implemented to the best of my ability, needless to say none of the code acctually works. The model takes as input, 5 Dimensional values and produces 3D-rendered images by volume rendering. This repository could support training NeRF models on nerf synthetic datasets.

![NeRF Training. Source: Mildenhall et al.](assets/nerf-training.png)

## Overview

NeRF is a method for generating novel views of complex 3D scenes or 3D objects by training a Feed Forward Network (MLP) to optimize over a volumetric scene function. The implementation involves:
- An MLP to output RGB and density values by processing the 5 dimensional input.
- Multiresolution hash encoding.

### Features

- **Structured Input.** The 5D input is not just available to be fed into the network, the data that is ultimately fed into the network is not what is available i.e. the raw pixel's RGB value and a transformation matrix. The transformation matrix conveys the orienation of the camera that shot that particular image, using this transformation matrix we can generate the theta and phi values required to feed into our MLP.  

- **Multi Layer Perceptron** that takes in the RGB, theta and phi values for every pixel in every image in the dataset as input and outputs RGB and volume opacity values for the relevant coordinates in the 3D space.

- **Volume rendering** to synthesize images by integrating through a learned 3D space. This is done by projecting rays from every pixel in every image. For each of these rays we sample points along those rays where we have RGA and volume opacity values generated using the MLP. Integrating the color and density values along these rays gives us the predicted RGB value for that particular pixel. Once we have the predicted values for the entire batch, the mean squared loss is taken to optimize the MLP.

- **Multiresolution Hash Encoding** based on Instant NGP. This has to shown accelerate learning. Instead of storing and learning for every voxel in the 3D space, grids over multiple levels of resolution are used to map voxels to feature vectors using hash tables. Fine and Coarse (Multiresolution) grids are employed to map intricate or global features that are relevant to the current grid using the hash tables. This approach significantly reduces the parameter footprint of the MLP whilst splitting the learning between the MLP and the hash tables.


![Multiresolution Hash Encoding. Source: Müller et al.](assets/multiresolution-hash-encoding.png)

## Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/R2D2-08/NeRF.git
   cd NeRF

## References 

1. Mildenhall, Ben, et al. [NeRF: Representing Scenes as Neural Radiance Fields for View Synthesis](https://arxiv.org/abs/2003.08934). arXiv, 3 Aug. 2020

2. Müller, Thomas, et al. [Instant Neural Graphics Primitives with a Multiresolution Hash Encoding](https://arxiv.org/abs/2201.05989v2). ACM Transactions on Graphics, vol. 41, no. 4, July 2022, pp. 1–15

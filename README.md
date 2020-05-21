# topography_dl
Implementations of AE, VAE, DFC VAE, VAE-GAN for topography data, i.e. the GEBCO dataset. 


## Requirements:
Python v3.7 

Keras vx.x (with tensorflow vx.x )

netCDF4

Numpy


## Dataset
The GEBCO dataset. GEBCO Compilation Group (2019), GEBCO_2019 Grid (doi: 10.5285/836f016a-33be-6ddc-e053- 6c86abc0788e), http://www.gebco.net.


## Description
The jupyter notebook "load_model" can be used to load the pre-trained weights of any of the models and evaluate them on the GEBCO dataset (or part of it). 

Accordignly, the jupyter notebook "train_model" can be used to train any of the models and evaluate them.

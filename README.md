# topography_dl
Implementations of AE, VAE, DFC VAE, VAE-GAN for topography data, i.e. the GEBCO dataset. 


## Requirements:
Python v3.7 

Keras v2.2.4 (with tensorflow-gpu v1.14.0)

netCDF4 v1.4.2

Scikit-learn

Numpy


## Dataset
The GEBCO dataset. GEBCO Compilation Group (2019), GEBCO_2019 Grid (doi: 10.5285/836f016a-33be-6ddc-e053- 6c86abc0788e), http://www.gebco.net.


## Description
"train.py" can be used to train any of the models.

Alternatively, the jupyter notebooks "load_model" & "train_model" can be used to load pre-trained weights or train any of the models and evaluate them on the GEBCO dataset (or part of it). 

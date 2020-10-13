# Master-Thesis

Synthesizing Pseudo CTs with Generative Adversarial Networks and X-Ray based a priori Information

# Abstract

Computed Tomography scans have been established as one of the most important
imaging diagnosis tools in medicine. Even though CT scans offer many advantages
as an imaging tool, they also come at the cost of exposing the patient to unwanted
radiation.
This work explores an alternative method of reconstructing CT images with the
use of biplanar X-rays and generative adversarial networks with the goal of reducing
the radiation exposure experienced by the patient. Additionally, with the augmentation
of existing X-ray image datasets in mind, this work also develops a method
to allow single image inference on a model that was trained using multiple inputs.
Several different architectures were evaluated and it was shown that a VAE/GAN
based approach was more successful in generating detailed CT reconstructions. Different
methods for single image inference were also proposed and evaluated with the
KNN prediction method showing the best results.

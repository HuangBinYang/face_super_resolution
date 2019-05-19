# 	CS-E4890 - Deep Learning - Aalto University

Super-Resolution Generative Adversarial Network with WGAN-GP loss trained on (preprocessed) Flickr-Faces-HQ Dataset.

Project for CS-E4890 - Deep Learning course in Aalto University.

### Contents

The main two scripts are:
* `main.py` - training script
* `compare.py` - generating super-resolution reconstructions

There is pretrained model uploaded and example images, so you can generate super-resolution images right away!


### Running
Install Conda environment from `env.yml`, read command line arguments in the beginning of `main.py`/`compare.py` scripts, run and have fun!

Running `python compare.py` with default arguments will generate images for you instantaneously :)

### References
* Martin Arjovsky, Soumith Chintala, and Léon Bottou. Wasserstein gan. 2017.
* Ishaan Gulrajani, Faruk Ahmed, Martin Arjovsky, Vincent Dumoulin, and Aaron Courville. Improved training of wasserstein gans. 2017.
* Xianxu Hou, Linlin Shen, Ke Sun, and Guoping Qiu. Deep feature consistent variational autoencoder. 2016.
* Tero Karras, Samuli Laine, and Timo Aila. A style-based generator architecture for generative adversarial networks. 2018.
* Diederik P Kingma and Max Welling. Auto-encoding variational bayes. 2013.
* Christian Ledig, Lucas Theis, Ferenc Huszar, Jose Caballero, Andrew Cunningham, Alejandro Acosta, Andrew Aitken, Alykhan Tejani, Johannes Totz, Zehan Wang, and Wenzhe Shi. Photo-realistic single image super-resolution using a generative adversarial network. 2016.
### Author

me


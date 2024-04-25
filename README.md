# Synthetic-Data-Generation
In this repo , we will use SinGAN to harmonize the synthtic data to make it look real . In addressing the challenge of aircraft detection and recognition in satellite imagery, where access to comprehensive real-world datasets is limited, a novel approach was pursued involving the seamless integration of synthetic aircraft patches into real satellite images. This integration process is facilitated by a Generative Adversarial Network (GAN), which effectively harmonized the synthetic elements with the visual characteristics and contextual nuances present in the real data through style transfer techniques. Subsequently, a Deep Learning (DL) model is deployed to conduct aircraft
prediction tasks, wherein various proportions of synthetic and real data were mixed. 

# Code
Install dependencies

~
python -m pip install -r requirements.txt
~

This code was tested with python 3.6, torch 1.4

Please note: the code currently only supports torch 1.4 or earlier because of the optimization scheme.

For later torch versions, you may try this repository: https://github.com/kligvasser/SinGAN (results won't necessarily be identical to the official implementation).
Train

To train SinGAN model on your own image, put the desired training image under Input/Images, and run

~
python main_train.py --input_name <input_file_name>
~

This will also use the resulting trained model to generate random samples starting from the coarsest scale (n=0).

To run this code on a cpu machine, specify --not_cuda when calling main_train.py

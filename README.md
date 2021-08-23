# Estimation of monocular depth using CNN

## Introduction

Our model is based on similar work [Towards Good Practice for CNN-Based Monocular Depth Estimation](https://openaccess.thecvf.com/content_WACV_2020/papers/Fang_Towards_Good_Practice_for_CNN-Based_Monocular_Depth_Estimation_WACV_2020_paper.pdf). Our aim was to recreate given research on NYUv2 labeled dataset containing 1449 pictures.

## Presentation
[Link to pdf presentation](https://github.com/m-grbic/psiml7/blob/main/Monocular%20Depth%20Estimation.pdf)

## Model
![DispNet arhitecture](https://i.ibb.co/DRrzWXK/Annotation-2021-08-07-191841.png)
Model was implemented using PyTorch (VGG-16 as encoder, DispNet as decoder).

## Important files
Changing hyperparameters: <br />
hyperparameters.py

Training: <br />
train.py

Test: <br />
test.py

Test on sample image: <br />
test_sample.py



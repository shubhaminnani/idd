# India Driving Dataset - Semantic Segmantation Challenge

Semantic scene segmentation has become a key application in computer vision and is an essential part of intelligent transportation systems for complete scene understanding of the surrounding environment. While several methods based on deep fully Convolutional Neural Network (CNN) have been emerging, there are two main challenges: (i) They mainly focus on improvement of the accuracy than efficiency. (ii) They assume structured driving environment like in USA and Europe. While most of the current works focus on the well structured driving environment, we focus our research on India Driving Dataset (IDD) which contains data from unstructured traffic scenario.



## Architecture Approach
We design a architecture with modifications in the DeepLabV3+ framework by using lower atrous rates in Atrous Spatial Pyramid Pooling (ASPP) module for dense traffic prediction. We propose to use dilated Xception network as the backbone for feature extraction.


## SpecPatConv3D Network

### 📌Description

This is an implementation code for <a href="https://arxiv.org/abs/1906.11981">Convolution Based Spectral Partitioning Architecture for Hyperspectral Image Classification</a>
 published at International Geoscience and Remote Sensing symposium 2019.
 
If you have any question related to the paper or the code, feel free to contact Ringo Chu (ringo.chu.16@ucl.ac.uk)
 
<img src="http://www.doc.ic.ac.uk/~swc3918/img/arch6.png" width="95%"><br>

This is a neural network architecture using 3D Convolutional neural network with partitioning on \lamba dimension to classify unlabelled pixel on Hyperspectral images,
 and achieving the state-of-art performance on classification with labelled dataset on Indian Pines, Salinas scenes.
 
### 🗂Prerequisites
*  Python3.7
*  TensorFlow 1.10
*  numpy
*  scipy
*  argparse
*  tqdm
*  (Optional) CUDA v9.0 and cuDNN v7.0
*  (Optional) TensorFlow GPU

We recommend you to create a Python Virtual environment, issue the 
command `pip install -r requirement.txt` in your command prompt/terminal to install all dependencies required.

If you use Anaconda, you could issue this command to create a virtual environment: `conda env create --file SpecPatConv3D.yml`

**_The code will be soon updated to TensorFlow 2.0 and will also be providing a PyTorch implementation, hold on tight and get updated!_**

All programs are texted under Ubuntu, MacOS, Windows10. For windows user, you'll need to download dataset manually with instructions below.

### 🛠General guidance using this repository

- **Acquire the Dataset (Do this Step if you're using Windows)**
    - Hyperspectral Datasets used in this research work Several hyperspectral datasets are available on the <a href="">UPV/EHU wiki</a>. Download IndianPines_corrected.mat or Salinas_corrected.mat or KSC.mat or Botswana.mat and unzip in the folder `data`
 
- **Preprocess and prepare dataset**
    - Run the following command: `python preprocess.py --data Indian_pines --train_ratio 0.15 --validation_ratio 0.05`
    - --data : Choose from Indian Pines, Salinas, KSC or Botswana
    
- **Train the model**
    - Run the following command: `python train.py --data Indian_pines --epoch 650`
    - --data : Choose from Indian Pines, Salinas, KSC or Botswana

- **Evaluate the model**
    - Run this command: `python evaluate.py --data Indian_pines`
    
### 📝Citation
If you find paper helpful, please consider citing us.❤
```

@inproceedings{igarss19chu,
  author    = {Ringo S.W. Chu and Ho-Chung Ng and Xiwen Wang and Wayne Luk},
  title     = {Convolution Based Spectral Partitioning Architecture for Hyperspectral Image Classification},
  booktitle = {{IEEE Geoscience and Remote Sensing symposium}},
  year      = {2019}
}
```

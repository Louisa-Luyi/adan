# ADAN
Source code for the paper "Adversarial Data Augmentation Network for Speech Emotion Recognition" and "Improving Speech Emotion Recognition with Adversarial Data Augmentation Network".

## Dependencies
    Python 3.x
    Tensorflow 1.10.0
    Keras 2.1.6
    numpy
    tqdm
    scipy
    seaborn
    matplotlib
    scikit_learn

## Data Preparation
The emotion feature vectors were extracted using the openSMILE toolkit. Codes in folders `matlab` and `scripts` can help extract emotion features from waveforms and store the feature vectors in .mat files.

## Data Augmentation
- `train_adan.py` trains the augmentation network based on adversarial learning (ADAN) or based on Wasserstein divergence (WADAN), and creates augmented sets;
- `adan.py` defines the network structure and training process of ADAN and WADAN;
- `utils.py` defines the additional functions used in adan.py;
- `tsne_plot.py` defines the functions for plotting the t-SNE plots reflecting the distributions of the real samples and synthetic samples. 

## Classification Task
- `svm_cls.py` trains an SVM classifier and outputs the UAR and WA;
- `dnn_cls.py` trains a DNN classifier and outputs the UAR and WA.

## How to run
To train an ADAN or WADAN with default settings:    
    
    python train_adan.py 
    
Apart from changing the python code file to redefine the parameters, you can also define the parameters directly through the command line. For example, when you want to change the dataset used to train the model:   

    python train_adan.py --dataset=iemocap  
    
and changing the number of training epochs:  

    python train_adan.py --dataset=iemocap --epoch=400

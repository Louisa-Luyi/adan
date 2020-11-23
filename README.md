# ADAN
Source code for the paper "Adversarial Data Augmentation Network for Speech Emotion Recognition" and "Improving Speech Emotion Recognition with Adversarial Data Augmentation Network".

## Dependencies
    Python 3.x
    Tensorflow 1.10.0

## Data Preparation
The emotion feature vectors were extracted using the openSMILE toolkit. Codes in folders `matlab` and `scripts` can help to extract emotion features from waveforms and store the feature vectors in .mat files.

## Data Augmentation
- `train_adan.py` trains the ADAN and creates augmented sets;
- `train_wadan.py` trains the WADAN and creates augmented sets;
- `adan.py` defines the ADAN (based on adversarial learning);
- `wadan.py` defines the WADAN (based on Wasserstein divergence);
- `utils.py` defines the additional functions used in adan.py and wadan.py;
- `tsne_plot.py` defines the functions for plotting the t-SNE plots reflecting the distributions of the real samples and synthetic samples. 

## Classification Task
- `svm_cls.py` trains an SVM classifier and outputs the UAR and WA;
- `dnn_cls.py` trains a DNN classifier and outputs the UAR and WA.

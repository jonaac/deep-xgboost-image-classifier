# Deep XGBoost Image Classifier

CNN+XGBoost architectures are very accurate in solving non-image classification problems. In this project, I adapt this model to show a shockingly powerful method for image classification. The idea is to leverage CNN's feature extraction capabilities and XGBoost's classification accuracy. I use the CIFAR-10 data set and I test the hybrid model on three different CNN architectures. A baseline CNN architecture, the VGG16 architecture, and the ResNet architecture.

![image](https://raw.githubusercontent.com/jonaac/deep-xgboost-image-classifier/main/imgs/cnn_xgboost_small.jpg)

## Files
```
code -- |- baseline
	|    |
	|    |-- cnn.py
	|    |-- cnn_xgboost.py
	|    |-- accuracy_baseline.py
	|
	|- resnet
	|    |
	|    |-- cnn_resnet.py
	|    |-- cnn_resnet_xgboost.py
	|    |-- accuracy_resnet.py
	|
	|- vgg16
	     |
	     |-- cnn_vgg.py
	     |-- cnn_vgg_xgboots.py
	     |-- accuracy_vgg16.py
```
## Getting Started

A list of all the prerequisites you'll need to run the experiments and the files the code will generate with the parameters to load the CNN and CNN+XGBoost models for each iteration.

### Prerequisites

```
Python
Keras
tensorflow
xgboost
sklearn
numpy
scipy
pickle
```

### Files Generated

For each CNN+XBoost the code will create files to load and train the different models used in this project.

```
For each iteration:
model.json			/* CNN model */
model.h5			/* CNN model trained weight */
cnn_xgboost_final.pickle.dat	/* CNN+XGBoost model* /
```

## Running Experiments

For each iteration, I train the original CNN model, I used the train model to generate the CNN+XGBoost model and I compare the accoracy of each model.
### Baseline
```
python3 cnn.py
python3 cnn_xgboost.py
python3 accuracy_baseline.py
```
### VGG16
```
python3 cnn_vgg16.py
python3 cnn_vgg16_xgboost.py
python3 acuoracy_vgg16.py
```
### ResNet50
```
python3 cnn_resnet.py
python3 cnn_resnet_xgboost.py
python3 accuracy_resnet.py
```

## Results

# Deep XGBoost Image Classifier

CNN+XGBoost architectures are very accurate in solving non-image classification problems. In this project, I adapt this model to show a shockingly powerful method for image classification. The idea is to leverage CNN's feature extraction capabilities and XGBoost's classification accuracy. I use the CIFAR-10 data set and I test the hybrid model on three different CNN architectures. A baseline CNN architecture, the VGG16 architecture, and the ResNet architecture.

![cnn_xgboost](https://raw.githubusercontent.com/jonaac/deep-xgboost-image-classifier/main/imgs/cnn_xgboost.jpg)

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

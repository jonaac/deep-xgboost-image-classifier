# Deep XGBoost Image Classifier

For this project I develop a CNN-XGBoost model for image classification. The idea is to leverage CNN's feature extraction capabilities and XGBoost's accuracy when it comes to classification. I will be working with the CIFAR-10 data set and I will be testing the hybrid model on three different CNN architectures. A baseline CNN architecture, the VGG16 architecture, and the ResNet architecture.

![cnn_xgboost](https://raw.githubusercontent.com/jonaac/deep-xgboost-image-classifier/main/imgs/cnn_xgboost.jpg)

'''
code	|- baseline
		|	|
		|	|-- cnn.py
		|	|-- cnn_xgboost.py
		|	|-- accuracy_baseline.py
		|
		|- resnet
		|	|
		|	|-- cnn_resnet.py
		|	|-- cnn_resnet_xgboost.py
		|	|-- accuracy_resnet.py
		|
		|- vgg16
			|
			|-- cnn_vgg.py
			|-- cnn_vgg_xgboots.py
			|-- accuracy_vgg16.py
'''

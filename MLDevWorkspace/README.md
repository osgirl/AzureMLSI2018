# Azure ML Facial Recognition Model

Utilizing the following models.

[Keras-VggFace](https://travis-ci.org/rcmalli/keras-vggface.svg?branch=master)](https://travis-ci.org/rcmalli/keras-vggface)

[Keras-Vgg19](https://github.com/keras-team/keras-applications)

[Azure Face API](https://azure.microsoft.com/en-us/services/cognitive-services/face/)


## VGGFace

- Model trained on Facenet dataset 

### Issues 

- There is a current issue with Keras 2.2.2 and its implementation of VGGFace.
- Work around [ImportError 'obtain_input_shape'](https://stackoverflow.com/questions/49113140/importerror-cannot-import-name-obtain-input-shape-from-keras)

## VGG19

- Model trained on Imagenet dataset.

## Azure Face API

- Microsoft in house mode. 


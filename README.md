# Image Captioning

This project uses neural networks (CNN and RNN) to automatically generate captions from images. I used the Microsoft Common Objects in COntext [(MS COCO) dataset](http://cocodataset.org/#home) to train the network.

## Project Files
The project is structured as a series of Jupyter notebooks that are designed to be completed in sequential order:

* 0_Dataset.ipynb - use COCO API to obtain the data
* 1_Preliminaries.ipynb - explore the data loader, experiment with the CNN Encoder & implement the RNN Decoder (in models.py)
* 2_Training.ipynb - train the model
* 3_Inference.ipynb - use trained model for image captioning on images

## MS COCO Instructions  
1. Clone this repo: https://github.com/cocodataset/cocoapi  
```
git clone https://github.com/cocodataset/cocoapi.git  
```

2. Setup the coco API (also described in the readme [here](https://github.com/cocodataset/cocoapi)) 
```
cd cocoapi/PythonAPI  
make  
cd ..
```

3. Download some specific data from here: http://cocodataset.org/#download (described below)

* Under **Annotations**, download:
  * **2014 Train/Val annotations [241MB]** (extract captions_train2014.json and captions_val2014.json, and place at locations cocoapi/annotations/captions_train2014.json and cocoapi/annotations/captions_val2014.json, respectively)  
  * **2014 Testing Image info [1MB]** (extract image_info_test2014.json and place at location cocoapi/annotations/image_info_test2014.json)

* Under **Images**, download:
  * **2014 Train images [83K/13GB]** (extract the train2014 folder and place at location cocoapi/images/train2014/)
  * **2014 Val images [41K/6GB]** (extract the val2014 folder and place at location cocoapi/images/val2014/)
  * **2014 Test images [41K/6GB]** (extract the test2014 folder and place at location cocoapi/images/test2014/)


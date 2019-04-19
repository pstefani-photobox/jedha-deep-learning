# Deep Learning for Image Recognition  

## Installation

* Install the requirements using `pip install -r requirements.txt`
* Install torch depending on your environment (Mac, Ubuntu, Cuda ...)

###### Flickr crawler
If you want to train a model on your own images, you can do so with `flickr_crawler.py`. Instructions to retrieve the configuration file will be given during the talk.

###### Train images
You can retrieve the pre-downloaded and formatted fruit images used in this talk to train deep networks.
* `wget <URL>`
* `tar -xvzf images.tar.gz images/` 

## Files

`models.py` contains the DeepNet class used in the tutorial, and its training script

`models/` contains an already working DeepNet model.

`test_images/` is a folder for pictures you want to play with. You can add your own if you want

`notebook.ipynb` is the starting point. Open it to follow the tutorial 

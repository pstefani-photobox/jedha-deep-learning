# Deep Learning for Image Recognition  

## Installation
Install torch depending on your environment (Mac, Ubuntu, Cuda ...) https://pytorch.org/get-started/locally/

###### Flickr crawler

Install the fickr api:
`pip install flickrapi`

You will need Flickr API keys as well to run the crawler.

If you want to train a model on your own images, you can do so with `flickr_crawler.py`. You can find an example of a configuration file in the repositiory, `flickr_crawler_config.json` 

###### Train images
You can retrieve the pre-downloaded and formatted fruit images used in this talk to train deep networks.
* `wget <URL>`
* `tar -xvzf images.tar.gz images/` 

## Files

`models.py` contains the DeepNet class used in the tutorial, and its training script

`models/` contains an already working DeepNet model.

`test_images/` is a folder for pictures you want to play with. You can add your own if you want

`notebook.ipynb` is the starting point. Open it to follow the tutorial 

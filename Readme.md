# Notung
This is a toy repository which implemented several deep learning models in Tensorflow. The name, Notung, is came from the sword in Richard Wagner's opera *Ring* used by Siegfried. Before the widely use of DL, every field like image recognization, audio recognization and natural language processing has own methods which deeply depend on human-construct features. But now, DL almost dominates these fields, just like the Notung chopped off Wotan's lance.
## Image cation
Image cation is a interesting field in DL which combines Convolutional Neural Network and Recurrent Neural Network. It first extracts image features by CNN and feeds them into a RNN to generate a description of the image.
This part is based of Karpathy's [neuraltalk2](https://github.com/karpathy/neuraltalk2) which implemented by Torch. I reimplement it in Tensorflow and convert the pretrained model. The model uses VGG-16 and LSTM architecture. Maybe Google's [im2txt](https://github.com/tensorflow/models/tree/master/im2txt) is a better choice, but there is no pretrained model available and I don't have a powerful graphic card to train own model. For more details, please see these two repos.
### Some good examples
caption: a cat and a dog are standing in a room
![](./imgs/dog.jpg)
caption: a white and black cat is sitting on the ground
![](./imgs/cat.jpg)
### Some wrong examples
caption: a brown bear sitting on top of a rock
![](./imgs/dog_wrong.jpg)
(emm, a brown bear, not that bad :P)
caption: a black and white cat sitting on top of a table
![](./imgs/cat_wrong.jpg)
As you see, the objects in image are recongnized, but the text is in a wrong sequence.
## Image Search
### Image similarity search
In last decades, some features like visual bag of words was developed for image search. But now, almost every search engine use DL to extract image features automatically. To avoid training a new model, I use the VGG-16 above as feature extractor. But it was finetuned for image caption, so the result is not that good as I thought. I will replace the model with original VGG or Inception trained on ImageNet in the future.
I use the last pooling layer as the feature and KDTree to search top-k nearest neighbors. The similarity is measured in euclidean distance. I build the database by a subset of MSCOCO with 10000 images. There is an example below.
query
![](./imgs/query.jpg)
top-3 inqueries
![](./imgs/inquery1.jpg)![](./imgs/inquery2.jpg)![](./imgs/inquery3.jpg)
### semantic similarity search
Using captions as keywords, it is easily to search similar images containing same objects.
As we know, keyword search is a complex optimization problem. I simply use weighted sum as objective function that all nouns and verbs with 1.0, adjectives and adverbs with 0.5 and all others with 0.1. The larger the weighted sum is, the more similar. There is an example below.
query
*a red car*
top-3 inqueries
![](./imgs/sem_inquery1.jpg)![](./imgs/sem_inquery2.jpg)![](./imgs/sem_inquery3.jpg)
## Neural Style
### Neural Style
The first nice neural style work is this [paper](https://arxiv.org/abs/1508.06576) which combine content features and style well and sovle it as optimization problem. For more details, you can see jcjohnson's [repo](https://github.com/jcjohnson/neural-style) implemented by Torch.
#### An example
I use the famous painting *Girl with a Pearl Earring* as content and *The Starry Night* as style.
content
![](./imgs/girl.jpg)
style
![](./imgs/star.jpeg)
result
![](./imgs/result.jpg)
### Fast Neural Style
jcjohnson improved the neural style by training a transform Net which transfer a style into images in real time.([paper](https://arxiv.org/abs/1603.08155) and [repo in Torch](https://github.com/jcjohnson/fast-neural-style))
I reimplement it in Tensorflow, but I hadn't trained a usable model because it almost take 20 hours on my graphic card. I have no enough time for experiments to find good hyperparameters by now, and I will do it in the future.
You can see this [repo](https://github.com/lengstrom/fast-style-transfer) also implemented in Tensorflow.
### Some Others
This is a [fancinating implementation of Neural Style](https://github.com/cysmith/neural-style-tf) in Tensorflow
This is a [implematation](https://github.com/rtqichen/style-swap) in Tensorflow of the [paper](https://arxiv.org/abs/1612.04337) which can transfer arbitrary style.
## GUI
A simple GUI was wrote which contains all the methods above.
![](./imgs/gui1.png)
![](./imgs/gui2.png)
## Requirement
* tensorflow
* PIL
* numpy
* sklearn
* nltk
* PyQt4
This is the [pretrained model](https://pan.baidu.com/s/1o8vx0DW), place it in the root.(I will also find an online disk out of chinese mainland.)
## Acknowledgement
Some ideas are borrowed from:
* [karpathy's neuraltalk2](https://github.com/karpathy/neuraltalk2)
* [jcjohnson's neural style](https://github.com/jcjohnson/neural-style)
* [jcjohnson's fast neural style](https://github.com/jcjohnson/fast-neural-style)
* [lengstrom's fast neural style](https://github.com/lengstrom/fast-style-transfer)

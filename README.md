# KNNDigitsRecognition
Machine Learning do the digits recogntion.

This is a basic project for those noobs to this area (like me) :)

I got a lot of inspiration and idea from @Pavitrakumar's work. I make some comments and try to do some improvements based his/her version.

Since I am a nood in data-mining, so I made really a huge mount of comments in this file and linked to a lot of articles. I come from China so some articles are in Chinese. Sorry for the inconvinient while I would try my best to make my comments understandable.

The main class is **class KNN_MODEL()**.

This class has two attributes, KNN parameter **k** and the KNN model **model**. Besides the init() method, this class has two methods, train and predict.

Besides this class, we have one function to do the digit recognitions.

The training stage would use one training model **train_image.jpg** to do the training, and it would generate a new picture **train_image_overlay.png** as the result.

It is **KNN_MachineLearning(INPUT_IMAGE, model)** given a input image **input_image.jpg**, it could generate two pictures.

One picture is **input_image_overlay.png** with the original image but with rectangle with digits on it.

Another picture is **output.png** with only the digits without the numbers.

Enjoy ;)

Created on Mar 30 2018 

@author: Mingju Li

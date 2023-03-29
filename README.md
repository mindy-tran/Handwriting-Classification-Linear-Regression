# Handwriting-Classification-Linear-Regression
Use linear regression to classify handwriting images in the MNIST dataset.

Mindy Tran
Project for Machine Learning for Data Science, Winter 2023

In this project you will learn about the MNIST dataset, a classic benchmark dataset in the field of machine
learning.
You will use the machinery you developed in the abalone linear regression project to classify images
in the MNIST dataset. The MNIST dataset is a collection of handwritten digits and their associated values. The handwritten
digits have been converted to pure grayscale, and the colors inverted so that the background is in
black (coded as 0 values) while the strokes are in white (coded as non-0 values).




*********************************************************************************
Task 0, Set up:

1. Install the required libraries for this project by running the following commands:

pip install scikit-learn

pip install scipy

pip install numpy

pip install matplotlib

2. Unzip the mnist_test_train.zip into your directory to get the mnist_test.csv and mnist_train.csv files.


*********************************************************************************
Task 1:  Run main.py with exp = Exp01() uncommented in the main function

This is a working pipeline which reads in the data for a digit, reshapes the data, and visualizes it using matplotlib. It visualizes samples indexes 0, 42, 156 of the training data and samples 0, 42, 542 of the testing data.
*********************************************************************************
Task 2:  Run main.py with exp = Exp02() uncommented in the main function

This uses linear regression to classify images in the MNIST dataset with the scikit-learn library and evaluates the model.
This includes the Mean Error Rate for the train and test data.

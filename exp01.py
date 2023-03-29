import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime


class Exp01:

    @staticmethod
    def load_train_test_data(file_path_prefix=""):
        """
        This method loads the training and testing data
        :param file_path_prefix: Any prefix needed to correctly locate the files.
        :return: x_train, y_train, x_test, y_test, which are to be numpy arrays.
        x_train and x_test should have a .shape value of (#samples, 28, 28)
        y_train and y_test should have a shape of (#samples, ) or (#samples, 1)
        """
        x_train, y_train, x_test, y_test = None, None, None, None

        mnist_test = pd.read_csv('mnist_test.csv', header=None)
        mnist_train = pd.read_csv('mnist_train.csv', header=None)

        # Test set
        # all pixels into an array for each row
        pixels_test = mnist_test.loc[:, 1:].values
        # reshape so each row of pixels is (28,28)
        pixels_test = [pixels_test[i].reshape(28, 28) for i in range(len(pixels_test))]
        # isolate the column: digit_the_image_represents
        mnist_test = mnist_test.loc[:, :0]
        # put pixels in second column, making an ordered pair
        mnist_test[1] = pixels_test

        # Train set
        # all pixels into an array for each row
        pixels_train = mnist_train.loc[:, 1:].values
        # reshape so each row of pixels is (28,28)
        pixels_train = [pixels_train[i].reshape(28, 28) for i in range(len(pixels_train))]
        # isolate the column: digit_the_image_represents
        mnist_train = mnist_train.loc[:, :0]
        # put pixels in second column, making an ordered pair
        mnist_train[1] = pixels_train

        # set up x, y test and train
        x_test = mnist_test[1]
        y_test = mnist_test[0]

        x_train = mnist_train[1]
        y_train = mnist_train[0]

        return x_train, y_train, x_test, y_test

    @staticmethod
    def display_image(x_image_data, y_label_value):
        """
        Display an image using matplotlib.
        :param x_image_data: A numpy array of shape (28, 28)
        :param y_label_value: An integer label.
        :return: None; displays a matplotlib plot.
        """
        plt.imshow(x_image_data)
        plt.title("Label = " + y_label_value)
        plt.show()

    def run(self):
        start_time = datetime.now()
        print("Running Exp: ", self.__class__, "at", start_time)

        print("Loading Data")
        x_train, y_train, x_test, y_test = Exp01.load_train_test_data()

        print("Displaying Certain Figures")

        Exp01.display_image(x_train[0], str(y_train[0]) + " (train)")
        Exp01.display_image(x_train[42], str(y_train[42]) + " (train)")
        Exp01.display_image(x_train[156], str(y_train[156]) + " (train)")

        Exp01.display_image(x_test[0], str(y_test[0]) + " (test)")
        Exp01.display_image(x_test[42], str(y_test[42]) + " (test)")
        Exp01.display_image(x_test[542], str(y_test[542]) + " (test)")

        # End and report time.
        end_time = datetime.now()
        print("Exp is over; completed at", datetime.now())
        total_time = end_time - start_time
        print("Total time to run:", total_time)




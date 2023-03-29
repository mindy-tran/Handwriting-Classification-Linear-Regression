import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.linear_model import LinearRegression


class Exp02:

    @staticmethod
    def load_train_test_data(file_path_prefix=""):
        """
        This method loads the training and testing data
        :param file_path_prefix: Any prefix needed to correctly locate the files.
        :return: x_train, y_train, x_test, y_test, which are to be numpy arrays.
        x_train and x_test should have a shape value of (#samples, 28, 28)
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
        x_test = np.array(mnist_test[1].tolist())
        x_test = x_test.reshape(x_test.shape[0], -1)
        y_test = np.array(mnist_test[0].tolist())

        x_train = np.array(mnist_train[1].tolist())
        x_train = x_train.reshape(x_train.shape[0], -1)
        y_train = np.array(mnist_train[0].tolist())

        return x_train, y_train, x_test, y_test

    @staticmethod
    def compute_mean_error_rate(true_y_values, predicted_y_values):
        """
        Computes the
        :param true_y_values:
        :param predicted_y_values:
        :return: The mean error rate of true values vs predicted values.
        """
        n = len(true_y_values)
        total_sum = 0
        # result = 0
        for i in range(n):
            pred = int(np.round(predicted_y_values[i]))
            total_sum += (true_y_values[i] != pred)

        return total_sum/n

    @staticmethod
    def print_error_report(trained_model, x_train, y_train, x_test, y_test):
        print("\tEvaluating on Training Data")
        # Evaluating on training data is a less effective as an indicator of
        # accuracy in the wild. Since the model has already seen this data
        # before, it is a less realistic measure of error when given novel/unseen
        # inputs.
        #
        # The utility is in its use as a "sanity check" since a trained model
        # which preforms poorly on data it has seen before/used to train
        # indicates underlying problems (either more data or data preprocessing
        # is needed, or there may be a weakness in the model itself.

        y_train_pred = trained_model.predict(x_train)

        mean_error_rate_train = Exp02.compute_mean_error_rate(y_train, y_train_pred)

        print("\tMean Error Rate (Training Data):", mean_error_rate_train)
        print()

        print("\tEvaluating on Testing Data")
        # Is a more effective as an indicator of accuracy in the wild.
        # Since the model has not seen this data before, so is a more
        # realistic measure of error when given novel inputs.

        y_test_pred = trained_model.predict(x_test)

        mean_error_rate_test = Exp02.compute_mean_error_rate(y_test, y_test_pred)

        print("\tMean Error Rate (Testing Data):", mean_error_rate_test)
        print()

    def run(self):
        start_time = datetime.now()
        print("Running Exp: ", self.__class__, "at", start_time)

        print("Loading Data")
        x_train, y_train, x_test, y_test = Exp02.load_train_test_data()

        print("Training Model...")

        #######################################################################
        # Complete this 2-step block of code using the variable name 'model' for
        # the linear regression model.
        # You can complete this by turning the given psuedocode to real code
        #######################################################################

        # (1) Initialize model; model = NameOfLinearRegressionClassInScikitLearn()
        model = LinearRegression()

        # (2) Train model using the function 'fit' and the variables 'x_train'
        # and 'y_train'
        model.fit(x_train.reshape(x_train.shape[0], -1), y_train)  # Fix this line

        print("Training complete!")
        print()

        print("Evaluating Model")
        Exp02.print_error_report(model, x_train, y_train, x_test, y_test)

        # End and report time.
        end_time = datetime.now()
        print("Exp is over; completed at", datetime.now())
        total_time = end_time - start_time
        print("Total time to run:", total_time)





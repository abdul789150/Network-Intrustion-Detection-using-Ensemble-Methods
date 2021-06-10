import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import learning_curve
from sklearn.model_selection import validation_curve
from sklearn.model_selection import ShuffleSplit
from sklearn.metrics import plot_confusion_matrix
import matplotlib.pyplot as plt
import time


def ModelLearning(X, y, number_of_estimators):
    # Create 10 cross-validation sets for training and testing
    cv = ShuffleSplit(n_splits = 10, test_size = 0.2, random_state = 0)

    # Generate the training set sizes increasing by 50
    train_sizes = np.rint(np.linspace(1, X.shape[0]*0.8 - 1, 9)).astype(int)

    # Create the figure window
    fig = plt.figure(figsize=(10,7))

    # Create different models based on max_depth
    for k, estimator in enumerate(number_of_estimators):

        # Create a Decision tree Classifier at max_depth = depth
        classifier = AdaBoostClassifier(n_estimators= estimator)

        # Calculate the training and testing scores
        sizes, train_scores, test_scores = learning_curve(classifier, X, y, cv = cv, train_sizes = train_sizes)

        # Find the mean and standard deviation for smoothing
        train_std = np.std(train_scores, axis = 1)
        train_mean = np.mean(train_scores, axis = 1)
        test_std = np.std(test_scores, axis = 1)
        test_mean = np.mean(test_scores, axis = 1)

        # Subplot the learning curve
        ax = fig.add_subplot(2, 2, k+1)
        ax.plot(sizes, train_mean, 'o-', color = 'r', label = 'Training Score')
        ax.plot(sizes, test_mean, 'o-', color = 'g', label = 'Testing Score')
        ax.fill_between(sizes, train_mean - train_std, \
            train_mean + train_std, alpha = 0.15, color = 'r')
        ax.fill_between(sizes, test_mean - test_std, \
            test_mean + test_std, alpha = 0.15, color = 'g')

        # Labels
        ax.set_title('Number of Estimators = %s'%(estimator))
        ax.set_xlabel('Number of Training Points')
        ax.set_ylabel('Score')
        ax.set_xlim([0, X.shape[0]*0.8])
        ax.set_ylim([-0.05, 1.05])

    # Visual aesthetics
    ax.legend(bbox_to_anchor=(1.05, 2.05), loc='lower left', borderaxespad = 0.)
    fig.suptitle('AdaBoost Model Learning Performances', fontsize = 16, y = 1.03)
    fig.tight_layout()
    fig.show()


def ModelComplexity(X, y):
    # Create 10 cross-validation sets for training and testing
    cv = ShuffleSplit(n_splits = 10, test_size = 0.2, random_state = 0)

    # Vary the max_depth parameter from 1 to 10
    n_estimators = np.arange(1,11)

    # Calculate the training and testing scores
    train_scores, test_scores = validation_curve(AdaBoostClassifier(), X, y,
        param_name = "n_estimators", param_range = n_estimators, cv = cv)

    # Find the mean and standard deviation for smoothing
    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    test_mean = np.mean(test_scores, axis=1)
    test_std = np.std(test_scores, axis=1)

    # Plot the validation curve
    plt.figure(figsize=(7, 5))
    plt.title('AdaBoost Classifier Model Complexity Performance')
    plt.plot(n_estimators, train_mean, 'o-', color = 'r', label = 'Training Score')
    plt.plot(n_estimators, test_mean, 'o-', color = 'g', label = 'Validation Score')
    plt.fill_between(n_estimators, train_mean - train_std, \
        train_mean + train_std, alpha = 0.15, color = 'r')
    plt.fill_between(n_estimators, test_mean - test_std, \
        test_mean + test_std, alpha = 0.15, color = 'g')

    # Visual aesthetics
    plt.legend(loc = 'lower right')
    plt.xlabel('Maximum Number of Estimators')
    plt.ylabel('Score')
    plt.ylim([-0.05,1.05])
    plt.show()




class AdaBoostModel():

    def __init__(self):
        pass

    def read_data(self):
        print("Reading Data!")
        dataframe = pd.read_csv("network_data.csv")
        dataframe = dataframe.drop(dataframe.columns[0], axis=1)

        features = dataframe.get(["Source IP", "Destination Port", "Forwarding status", "Packets exchanged"
               ,"Bytes exchanged"])

        label = dataframe.get('label')

        enc = OneHotEncoder()
        preprocessed_features = enc.fit_transform(features)

        return preprocessed_features, label

    def train_model(self, x_train, y_train):
        print("Started Training")
        self.classifier = AdaBoostClassifier(n_estimators=6)
        start = time.time()
        self.classifier.fit(x_train, y_train)
        stop = time.time()
        print("Time taken for Training: {} seconds.".format(round(stop - start)))

    def test_model(self, x_test, y_test):
        print("Started Model Testing")
        testing_accuracy = round(self.classifier.score(x_test, y_test), 3) * 100
        print("Testing Accuracy: {}%".format(testing_accuracy) )

    def plot_confusion_matrix(self, x_test, y_test):
        fig, ax = plt.subplots(figsize=(8, 8))
        plot_confusion_matrix(self.classifier, x_test, y_test, normalize='true', cmap=plt.cm.Blues, ax=ax)
        plt.show()

    def plot_learning_curves(self, features, label):
        ModelLearning(features, label, [1, 3, 6, 10])

    def plot_model_complexity(self, features, label):
        ModelComplexity(features, label)
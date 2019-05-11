# Imports
import numpy as np

from sklearn import svm, model_selection as cross_validation, neighbors
from sklearn.ensemble import VotingClassifier, RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from collections import Counter

from keras.models import Sequential
from keras.layers import Dense, Embedding, LSTM, SpatialDropout1D, Dropout, Activation, Conv1D, \
    GlobalMaxPooling1D, BatchNormalization
from keras.regularizers import l2



# Functions

def train_rf(df_tweets, tokenized_tweets, test_size: float = 0.25, n_estimators: int = 500):
    """This function trains a randomforest to distinguish between positive, neutral and negative tweets

    Args:
        df_tweets (pandas.core.frame.DataFrame): Preprocessed data frame with labelled tweets
        tokenized_tweets (numpy.ndarray): Tokenized tweets
        test_size (float): How large should the test_size be in the train-test split

    Return:
        forest_mod_acc (float): Accuracy of the model
        forest_mod (sklearn.ensemble.forest.RandomForestClassifier): The randomforest model

    """

    # Get list of labels
    my_labels = df_tweets['label']

    # Create train - test split
    x_train, x_test, y_train, y_test = train_test_split(tokenized_tweets, my_labels,
                                                        train_size=1-test_size, test_size=test_size,
                                                        random_state=1)

    # Create random forest
    forest_mod = RandomForestClassifier(criterion='gini', n_estimators=n_estimators, verbose=1)
    forest_mod.fit(x_train, y_train)

    # Predict validation data and compute accuracy
    forest_mod_acc = accuracy_score(y_test, forest_mod.predict(x_test))
    print('Random forest accuracy:', forest_mod_acc)

    return forest_mod_acc, forest_mod, x_train, x_test, y_train, y_test


def confusion_matrix (my_pred, my_ground_truth, num_class):
    """This function returns the confusion matrix for further analysis

        Args:
            x_test (numpy.ndarray): The test part of the train/test split
            y_test (numpy.ndarray): The "ground truth"
            model (sklearn.ensemble.forest.RandomForestClassifier): The trained model

        Return:
            conf_mat (numpy.ndarray): The confusion matrix

        """

    conf_mat = np.zeros((num_class, num_class))
    for i in range(len(my_ground_truth)):
        conf_mat[my_ground_truth[i]][my_pred[i]] += 1

    conf_mat.astype(int)
    print('Confusion Matrix: ')
    print(conf_mat)

    return conf_mat


def compute_metrics (conf_mat):
    """This function returns different metrics calculated with the confusion matrix

        Args:
            conf_mat (numpy.ndarray): Confusion matrix (2x2)

        Return:
            precision (numpy.float64): The precision of the model
            recall (numpy.float64): The recall of the model
            accuracy (numpy.float64): The accuracy of the model

        """
    # To avoid division by zero.
    if conf_mat[0, 1] == 0 and conf_mat[1, 0] == 0:

        precision = recall = accuracy = 1.0

    else:

        precision = conf_mat[1][1] / (conf_mat[1][1] + conf_mat[0][1])
        recall = conf_mat[1][1] / (conf_mat[1][1] + conf_mat[1][0])
        accuracy = (conf_mat[1][1] + conf_mat[0][0]) / (
                    conf_mat[1][1] + conf_mat[0][0] + conf_mat[0][1] + conf_mat[1][0])
    print('Precision: ' + str(precision))
    print('Recall: ' + str(recall))
    print('Accuracy: ' + str(accuracy))
    return precision, recall, accuracy


def get_base_cnn(max_features: int, embedding_dims: int, maxlen: int,
                 num_conv_filters: int, kernel_size: int, num_hidden_dims: int):
    """This function creates a basic cnn with the parameters

       """

    model = Sequential()

    model.add(Embedding(max_features,
                        embedding_dims,
                        input_length = maxlen))
    model.add(BatchNormalization())

    model.add(Conv1D(num_conv_filters,
                     kernel_size,
                     padding = 'valid',
                     activation = 'relu',
                     strides = 1,
                     kernel_regularizer = l2()))
    model.add(BatchNormalization())
    model.add(GlobalMaxPooling1D())

    model.add(Dense(num_hidden_dims))
    model.add(BatchNormalization())
    model.add(Activation('relu'))

    model.add(Dense(1))
    model.add(Activation('sigmoid'))

    return model


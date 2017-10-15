import numpy as np

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Activation
import keras
import re


# TODO: fill out the function below that transforms the input series 
# and window-size into a set of input/output pairs for use with our RNN model
def window_transform_series(series, window_size):
    # containers for input/output pairs
    X = []
    y = []

    # getting the max index to be able to fit window and window_size for input/output pairs
    len_minus_window_size = len(series) - window_size

    # compute input/output pairs
    for i in range(0, len_minus_window_size):
        j = i + window_size
        X.append(series[i:j])
        y.append(series[j])

    # reshape each 
    X = np.asarray(X)
    X.shape = (np.shape(X)[0:2])
    y = np.asarray(y)
    y.shape = (len(y), 1)

    return X, y

# TODO: build an RNN to perform regression on our time series input/output data
def build_part1_RNN(window_size):
    # Building RNN
    # layer 1 uses an LSTM module with 5 hidden units (note here the input_shape = (window_size,1))
    # layer 2 uses a fully connected module with one unit
    return Sequential([
        LSTM(5, input_shape = (window_size, 1)),
        Dense(1)
    ])

### TODO: return the text input with only ascii lowercase and the punctuation given below included.
def cleaned_text(text):
    punctuation = ['!', ',', '.', ':', ';', '?']

    # Using a regular expression to keep any letter and valid punctuation and remove everything else
    invalid_characters = "[^a-zA-Z" + ''.join(punctuation) + "]"

    pattern = re.compile(invalid_characters);
    text = pattern.sub(' ', text)

    return text

### TODO: fill out the function below that transforms the input text and window-size into a set of input/output pairs for use with our RNN model
def window_transform_text(text, window_size, step_size):
    # containers for input/output pairs
    inputs = []
    outputs = []

    # getting the max index to be able to fit window and window_size for input/output pairs
    len_minus_window_size = len(text) - window_size

    # Making sure range takes into account the step_size for each increment
    for i in range(0, len_minus_window_size, step_size):
        j = i + window_size
        inputs.append(text[i:j])
        outputs.append(text[j])

    return inputs, outputs

# TODO build the required RNN model: 
# a single LSTM hidden layer with softmax activation, categorical_crossentropy loss 
def build_part2_RNN(window_size, num_chars):
    # Buildng RNN
    # layer 1 should be an LSTM module with 200 hidden units --> note this should have input_shape = (window_size,len(chars)) where len(chars) = number of unique characters in your cleaned text
    # layer 2 should be a linear module, fully connected, with len(chars) hidden units --> where len(chars) = number of unique characters in your cleaned text
    # layer 3 should be a softmax activation (since we are solving a multiclass classification)
    return Sequential([
        LSTM(200, input_shape = (window_size, num_chars)),
        Dense(num_chars),
        Activation('softmax')
    ])

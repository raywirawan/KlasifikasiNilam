import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import MinMaxScaler
from scipy.linalg import pinv2

# Modified from the following tutorial:
# https://github.com/glenngara/tutorials/blob/master/elm/elm-tutorial.ipynb

def getPrediction(TRAINING_FILE, TESTING_FILE, HIDDEN_NODES_SIZE):
    
    print("Starting ELM procedure...")

    TRAINING_FEATURES = pd.read_csv(TRAINING_FILE, header=None)
    TESTING_FEATURES = pd.read_csv(TESTING_FILE, header=None)

    onehotencoder = OneHotEncoder(categories='auto')
    scaler = MinMaxScaler()

    X_train = scaler.fit_transform(TRAINING_FEATURES.values[:,2:])
    y_train = onehotencoder.fit_transform(TRAINING_FEATURES.values[:,:1]).toarray()

    X_test = scaler.fit_transform(TESTING_FEATURES.values[:,2:])
    y_test = onehotencoder.fit_transform(TESTING_FEATURES.values[:,:1]).toarray()

    input_size = X_train.shape[1]

    input_weights = np.random.normal(size=[input_size,HIDDEN_NODES_SIZE])
    biases = np.random.normal(size=[HIDDEN_NODES_SIZE])

    output_weights = np.dot(pinv2(\
        hidden_nodes(X_train, input_weights, biases)\
        ), y_train)

    prediction = predict(X_test, hidden_nodes, input_weights, biases, output_weights)

    """ #debugging part
    for value in prediction:
        print("Prediksi", ":", max(value), "-", TRAINING_FEATURES[1][np.argmax(value)]) """

    #get the last image in the testing csv
    value = prediction[-1:,:]
    #if you uploaded more than 1 image then only the last image's prediction will be returned

    return_value = []
    return_value.append(TRAINING_FEATURES[1][np.argmax(value)]) #returns the predicted class of the input
    confidence_value = value[0][np.argmax(value)]
    
    #for display purpose, preventing confidence from being more than 100%
    confidence_value = confidence_value if (confidence_value <= 1.0) else 1.0
    
    return_value.append("{:.2%}".format(confidence_value))
    return_value.append(str(value[0][np.argmax(value)]))

    print("Prediction completed!")

    return return_value

def relu(x):
   return np.maximum(x, 0, x)

def hidden_nodes(X, input_weights, biases):
    G = np.dot(X, input_weights)
    G = G + biases
    H = relu(G)
    return H

def predict(X, hidden_nodes, input_weights, biases, output_weights):
    out = hidden_nodes(X, input_weights, biases)
    out = np.dot(out, output_weights)
    return out

if __name__ == "__main__":
    
    TRAINING_FILE = "training_features.csv"
    TESTING_FILE = "testing_features.csv"
    HIDDEN_NODES_SIZE = 1000
    prediction = getPrediction(TRAINING_FILE, TESTING_FILE, HIDDEN_NODES_SIZE)

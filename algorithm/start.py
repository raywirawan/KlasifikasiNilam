from algorithm import feature_extraction, ELM
from pathlib import Path

TRAINING_DATASET_PATH   = "dataset/training/*.jpg"
TRAINING_FILE           = "dataset/csv/training_features.csv"

def train():
    feature_extraction.extract(TRAINING_DATASET_PATH, TRAINING_FILE)

TESTING_DATASET_PATH    = "dataset/testing/"
TESTING_FILE            = "dataset/csv/testing_features.csv"

def test(filename):
    TESTFILE_PATH       = TESTING_DATASET_PATH+filename
    feature_extraction.extract(TESTFILE_PATH, TESTING_FILE)
    
    #remove the file after extraction to save storage space
    Path(TESTFILE_PATH).unlink()

HIDDEN_NODES_SIZE       = 10000      #please refer to hidden nodes in ELM algorithm in the internet

def predict():
    prediction = ELM.getPrediction(TRAINING_FILE, TESTING_FILE, HIDDEN_NODES_SIZE)
    return prediction
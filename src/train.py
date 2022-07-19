# Modules
import os
import pickle

from sklearn.datasets import fetch_20newsgroups
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

import config
from MNB_model import my_MNB

def run_training():
    """
    Training ML model 
    """

    # Load in data
    news = fetch_20newsgroups(subset="all")
    
    # Extract the messages and topic labels, and view the topic labels
    text = news["data"]
    target = news["target"]

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(text, target, random_state=0)
    
    # Create feature representation -- here we use TF-IDF
    tfidfer = TfidfVectorizer()
    tfidfer.fit(X_train)
    X_train_tfidf = tfidfer.transform(X_train)
    
    # Path to save model
    model_path = f"{config.MODEL_DIR}/"

    # Checking the folder exist
    if not os.path.exists(model_path):
        os.makedirs(model_path)
    
    # Train a single model
    mnb = my_MNB()
    mnb.fit(X_train_tfidf, y_train)
    
    # save the model to disk
    filename = 'MNB_trained.sav'
    pickle.dump(mnb, open(f"{model_path}/{filename}", 'wb'))


if __name__ == "__main__":
    run_training()
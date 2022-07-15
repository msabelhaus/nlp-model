# Modules
from sklearn.datasets import fetch_20newsgroups
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer


def run_training():
    """
    Training ML model 
    """

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
    X_test_tfidf = tfidfer.transform(X_test)
    
    # Load the classifier
    


if __name__ == "__main__":
    run_training("LSTM")
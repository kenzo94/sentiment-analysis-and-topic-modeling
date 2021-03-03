from sklearn.feature_extraction.text import TfidfVectorizer, TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn import svm
from sklearn.svm import SVC
import time
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import pandas as pd
from sklearn.preprocessing import MaxAbsScaler, MinMaxScaler
from yellowbrick.model_selection import FeatureImportances
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import cross_val_score
from nltk.corpus import stopwords
import numpy as np

stopwords = stopwords.words("german")

def get_tfidf_vector(corpus, ngram):
    vectorizer = TfidfVectorizer(min_df=5, max_df=0.8, ngram_range=ngram, stop_words=stopwords)
    vectorizer.fit(corpus)
    print(vectorizer.vocabulary_, "\n") # Zeig wörter und ihre einzigartige id
    print(vectorizer.get_feature_names(), "\n") # zeig alle vorhandenen features
    print(f"vocab length: {len(vectorizer.vocabulary_)}\n")
    return vectorizer

def get_bow(corpus, ngram, stopwords):
    vectorizer = CountVectorizer(min_df=5, ngram_range=ngram, stop_words=stopwords, binary=True)
    vectorizer.fit(corpus)
    print(vectorizer.vocabulary_, "\n") # Zeig wörter und ihre einzigartige id
    print(vectorizer.get_feature_names(), "\n") # zeig alle vorhandenen features
    print(f"vocab length: {len(vectorizer.vocabulary_)}")
    return vectorizer

def svm_clf(X_train, X_test, y_train, y_test):
    linear_svm = svm.LinearSVC(class_weight="balanced")
    linear_svm.fit(X_train, y_train)

    # predict class y
    predictions = linear_svm.predict(X_test)
    print(f"Confusion Matrix:\n{confusion_matrix(y_test, predictions)}\n")
    print(f"Classification report:\n{classification_report(y_test, predictions)}\n")
    print(f"Accuracy: \n{accuracy_score(y_test, predictions)}")

#random forest
def rf_clf(X_train, X_test, y_train, y_test):
    #random_forest = RandomForestClassifier(n_estimators=200, random_state=0)

    #viz = FeatureImportances(random_forest)
    #viz.fit(X_train, y_train)
    #viz.show()

    #random_forest.fit(X_train, y_train)
    #print(random_forest.feature_importances_)
    # model = SelectFromModel(random_forest, prefit=True, threshold="mean")
    #X_new = model.transform(X_train)
    #print(X_new)
    clf = Pipeline([
        ("feature_selection", SelectFromModel(RandomForestClassifier(n_estimators=900, random_state=0, max_features=12, class_weight="balanced"))),
        ("classification", RandomForestClassifier(n_estimators=900, random_state=0, max_features=12, class_weight="balanced"))
    ])
    clf.fit(X_train, y_train)

    print(X_train.shape)
    # predict class y
    predictions = clf.predict(X_test)
    print(confusion_matrix(y_test, predictions))
    print(classification_report(y_test, predictions))
    print(accuracy_score(y_test, predictions))

#Multinomial Naive Bayes
def mnb_clf(X_train, X_test, y_train, y_test):
    # training naive bayes model
    NB_model = MultinomialNB()
    NB_model.fit(X_train, y_train)

    #predict class y
    predictions = NB_model.predict(X_test)
    print(f"Confusion Matrix:\n{confusion_matrix(y_test, predictions)}\n")
    print(f"Classification report:\n{classification_report(y_test, predictions)}\n")
    print(f"Accuracy: \n{accuracy_score(y_test, predictions)}")

if __name__ == "__main__":

    df = pd.read_csv("resources/corpus/train_corpus.csv", sep=",")
    df_meta = pd.read_csv("resources/corpus/train_corpus_meta.csv", delimiter=",", index_col=False, header=0)

    data_meta = df_meta.loc[:, "anzahl_wort":"sentiWS_score"]
    #print(data_meta["anzahl_neg_tri"])
    #vectorize_corpus = get_tfidf_vector(df["tweet_processed"], (1, 1))
    vectorize_corpus = get_bow(df["tweet_processed"], (1, 1), stopwords)
    data_n = vectorize_corpus.transform(df["tweet_processed"]).toarray()
    #print(data_meta)

    feature_vec = np.concatenate((data_n, data_meta), axis=1)
    print(feature_vec)

    # avoid overfitting, split into training and test data

    tweets_train, tweets_test, y_train, y_test = train_test_split(feature_vec, df["sentiment"],
                                                                  test_size=0.20, random_state=1, shuffle=True,
                                                                  stratify=df["sentiment"])

    print("Shape of tweets_train:", tweets_train.shape)
    print("Shape of tweets_test:", tweets_test.shape)
    print("Shape of sentiment_train:", y_train.shape)
    print("Shape of sentiment_test:", y_test.shape, "\n")

    """
    print("[Multinomial Naive Bayes]\n")
    mms = MinMaxScaler()
    t0mnb = time.time()
    mnb_clf(mms.fit_transform(tweets_train), mms.transform(tweets_test), y_train, y_test)
    t1mnb = time.time()
    time_linear_train_mnb = t1mnb - t0mnb
    # results
    print(f"Training time: {time_linear_train_mnb}\n")

    print("[Support Vector Machine]\n")
    sc = MaxAbsScaler()
    t0svm = time.time()
    svm_clf(sc.fit_transform(tweets_train), sc.transform(tweets_test), y_train, y_test)
    t1svn = time.time()
    time_linear_train_svm = t1svn - t0svm
    # result
    print(f"Training time: {time_linear_train_svm}\n")
    """

    print("[Random Forest]\n")
    t0rf = time.time()
    rf_clf(tweets_train, tweets_test, y_train, y_test)
    t1rf = time.time()
    time_linear_train_rt = t1rf - t0rf
    # results
    print(f"Training time: {time_linear_train_rt}\n")

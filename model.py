import pandas as pd
import time
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer, TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn import svm
from sklearn.svm import SVC
from sklearn.preprocessing import MaxAbsScaler, MinMaxScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from yellowbrick.text import TSNEVisualizer
from yellowbrick.text import FreqDistVisualizer
from yellowbrick.target import ClassBalance
from yellowbrick.features import PCA
from yellowbrick.model_selection import FeatureImportances
from yellowbrick.classifier import ConfusionMatrix
from collections import defaultdict
import numpy as np
import spacy
import pickle

# In der Main wird das Skript gesteuert
# Quellen:
# https://machinelearningmastery.com/cost-sensitive-svm-for-imbalanced-classification/
# https://stackabuse.com/python-for-nlp-sentiment-analysis-with-scikit-learn/#disqus_thread
# https://www.pluralsight.com/guides/building-a-twitter-sentiment-analysis-in-python
# https://scikit-learn.org/stable/auto_examples/model_selection/grid_search_text_feature_extraction.html#sphx-glr-auto-examples-model-selection-grid-search-text-feature-extraction-py
# https://scikit-learn.org/stable/auto_examples/model_selection/plot_grid_search_digits.html#sphx-glr-auto-examples-model-selection-plot-grid-search-digits-py
# https://machinelearningmastery.com/machine-learning-in-python-step-by-step/
# https://towardsdatascience.com/3-things-you-need-to-know-before-you-train-test-split-869dfabb7e50
# https://machinelearningmastery.com/save-load-machine-learning-models-python-scikit-learn/
# augerufen 08.12.2020 um 19:39

train_path = "resources/corpus/train_corpus.csv"
train_path_meta = "resources/corpus/train_corpus_meta.csv"
pred_path = "resources/corpus/pred_corpus_meta.csv"

nlp = spacy.load('de_core_news_lg', disable=['parser', 'ner'])

all_features = ["anzahl_wort", "anzahl_caps", "anzahl_punc", "anzahl_qm",
                 "anzahl_em", "lange_seq_em_qm", "lange_seq_qm_em", "lange_r_letters", "lange_r_punc","last_punc", "last_em",
                 "last_qm", "anzahl_slangs", "anzahl_emoji", "anzahl_pos_emoji", "anzahl_neg_emoji", "anzahl_pos_smiley",
                 "anzahl_neg_smiley", "last_emoji_smiley", "anzahl_verb", "anzahl_nomen", "anzahl_pron", "anzahl_adj",
                 "anzahl_intj", "anzahl_cconj", "anzahl_negation", "anzahl_verstarker", "anzahl_shift_pos",
                 "anzahl_shift_neg", "anzahl_shift_gen", "anzahl_neutral_uni", "anzahl_pos_uni", "anzahl_pos_bi",
                 "anzahl_pos_tri", "anzahl_neg_uni", "anzahl_neg_bi", "anzahl_neg_tri", "sentiWS_score"]

top_10_features = ["anzahl_wort", "anzahl_caps" , "anzahl_nomen", "anzahl_neutral_uni", "anzahl_punc", "anzahl_slangs",
                   "anzahl_verb", "anzahl_adj", "anzahl_pron", "anzahl_neg_uni"]

top_20_features = ["anzahl_wort", "anzahl_caps" , "anzahl_nomen", "anzahl_neutral_uni", "anzahl_punc", "anzahl_slangs",
                   "anzahl_verb", "anzahl_adj", "anzahl_pron", "anzahl_neg_uni", "anzahl_pos_smiley", "anzahl_em",
                   "sentiWS_score", "anzahl_pos_uni", "anzahl_shift_gen", "anzahl_negation", "lange_r_punc",
                   "anzahl_qm", "last_punc", "anzahl_verstarker"]

#nltk - 232
stopwords = stopwords.words("german")

#spacy - 543
stop_words = nlp.Defaults.stop_words

# Print absolut number of sentiments
def explore_data(path):
    df = pd.read_csv(path)
    vis = ClassBalance(labels=["negativ", "neutral", "positive"], colors=["red", "yellow", "green"])
    vis.fit(df["sentiment"])
    vis.show()

# print percentage of sentiments
def show_sentiment(path):
    df = pd.read_csv(path)
    df.sentiment.value_counts().plot(kind='pie', autopct='%1.0f%%', colors=["yellow", "green", "red"])
    plt.show()

# show statistics of words over whole corpus
def show_word_statistics(path, ngram):
    df = pd.read_csv(path, sep=",")

    vectorizer = CountVectorizer()
    train_vectors = vectorizer.fit(df.tweet_processed)
    print(train_vectors.vocabulary_, "\n")  # Zeig wörter und ihre einzigartige id
    print(train_vectors.get_feature_names(), "\n")  # zeig alle vorhandenen features
    print(f"vocab length: {len(train_vectors.vocabulary_)}")

    # Create a dict to map target labels to documents of that category
    tweets = defaultdict(list)
    for text, label in zip(df.tweet_processed, df.sentiment):
        tweets[label].append(text)

    for cat in ["neutral", "positive", "negative"]:
        vec = CountVectorizer(ngram_range=ngram, stop_words=stopwords)
        docs = vec.fit_transform(text for text in tweets[cat])
        features = vec.get_feature_names()

        visualizer = FreqDistVisualizer(
            features=features,
            title=cat,
            n=10 # top 10 freq words, change for more
        )
        visualizer.fit(docs)
        visualizer.show()

def print_statistics():
    for path in [train_path, pred_path]:
        explore_data(path)
        show_sentiment(path)
        show_word_statistics(path, (1, 1))
        df = pd.read_csv(path)
        print(df.sentiment.value_counts())

# max_features: build a vocabulary that only consider the top max_features ordered by term frequency across the corpus.
# max_df: ignore terms that have more frequency than the threshhold
# min_df: ignore terms that have lower frequency than the threshhold
def get_tfidf_vector(corpus, ngram, stopwords):
    vectorizer = TfidfVectorizer(min_df=5, ngram_range=ngram, stop_words=stopwords)
    vectorizer.fit(corpus)
    print(vectorizer.vocabulary_, "\n") # Zeig wörter und ihre einzigartige id
    print(vectorizer.get_feature_names(), "\n") # zeig alle vorhandenen features
    print(f"vocab length: {len(vectorizer.vocabulary_)}\n")
    return vectorizer

def get_bow(corpus, ngram, stopwords):
    vectorizer = CountVectorizer(min_df=10, ngram_range=ngram, stop_words=stopwords)
    vectorizer.fit(corpus)
    print(vectorizer.vocabulary_, "\n") # Zeig wörter und ihre einzigartige id
    print(vectorizer.get_feature_names(), "\n") # zeig alle vorhandenen features
    print(f"vocab length: {len(vectorizer.vocabulary_)}")
    return vectorizer

def svm_clf(X_train, X_test, y_train, y_test):
    linear_svm = svm.LinearSVC(class_weight="balanced", random_state=1, C=1)
    linear_svm.fit(X_train, y_train)

    # predict class y
    predictions = linear_svm.predict(X_test)
    print(f"Confusion Matrix:\n{confusion_matrix(y_test, predictions)}\n")
    print(f"Classification report:\n{classification_report(y_test, predictions)}\n")
    print(f"Accuracy: \n{accuracy_score(y_test, predictions)}")

    vis_svm = ConfusionMatrix(linear_svm)
    vis_svm.fit(X_train, y_train)
    vis_svm.score(X_test, y_test)
    vis_svm.show()

#random forest
def rf_clf(X_train, X_test, y_train, y_test):
    random_forest = RandomForestClassifier(random_state=1, n_estimators=900, max_features=12, class_weight="balanced")
    random_forest.fit(X_train, y_train)

    # predict class y
    predictions = random_forest.predict(X_test)
    print(f"Confusion Matrix:\n{confusion_matrix(y_test, predictions)}\n")
    print(f"Classification report:\n{classification_report(y_test, predictions)}\n")
    print(f"Accuracy: \n{accuracy_score(y_test, predictions)}")

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

# features are vectorized with tfidf
def model_tfidf(ngram, stopwords):

    df = pd.read_csv(train_path, sep=",")
    #df.drop(df.query("sentiment == \"neutral\"").sample(frac=.3).index, inplace=True)

    # avoid overfitting, split into training and test data
    tweets_train, tweets_test, y_train, y_test = train_test_split(df["tweet_processed"], df["sentiment"],
                                                        test_size=0.20, random_state=1, shuffle=True, stratify=df["sentiment"])

    print("Shape of tweets_train:", tweets_train.shape)
    print("Shape of tweets_test:", tweets_test.shape)
    print("Shape of sentiment_train:", y_train.shape)
    print("Shape of sentiment_test:", y_test.shape, "\n")

    # fit should be only to the training set, better for corpus generalisation
    vectorize_corpus = get_tfidf_vector(tweets_train, ngram, stopwords)
    X_train = vectorize_corpus.transform(tweets_train)
    X_test = vectorize_corpus.transform(tweets_test)

    # visualize dataset
    #tsne = TSNEVisualizer()
    #tsne.fit(X_train, y_train)
    #tsne.show()

    print(f"Document term matrix :\n\n{X_train.toarray()}\n")
    print(f"Nr. of Doc: {X_train.shape[0]} Nr. of Feature: {X_train.shape[1]}\n")
    dtm_min_max = np.array(X_train.toarray())
    print(f"min value in matrix: {dtm_min_max.min()} max value in matrix: {dtm_min_max.max()}")


    print("[Multinomial Naive Bayes]\n")
    t0mnb = time.time()
    mnb_clf(X_train, X_test, y_train, y_test)
    t1mnb = time.time()
    time_linear_train_mnb = t1mnb - t0mnb
    # results
    print(f"Training time: {time_linear_train_mnb}\n")

    print("[Support Vector Machine]\n")
    t0svm = time.time()
    svm_clf(X_train, X_test, y_train, y_test)
    t1svn = time.time()
    time_linear_train_svm = t1svn - t0svm
    # result
    print(f"Training time: {time_linear_train_svm}\n")

    print("[Random Forest]\n")
    t0rf = time.time()
    rf_clf(X_train, X_test, y_train, y_test)
    t1rf = time.time()
    time_linear_train_rt = t1rf - t0rf
    # results
    print(f"Training time: {time_linear_train_rt}\n")

# feature are vectorized with bow
def model_bow(ngram, stopwords):
    df = pd.read_csv(train_path, sep=",")
    # df.drop(df.query("sentiment == \"neutral\"").sample(frac=.3).index, inplace=True)

    # avoid overfitting, split into training and test data
    tweets_train, tweets_test, y_train, y_test = train_test_split(df["tweet_processed"], df["sentiment"],
                                                                  test_size=0.20, random_state=1, shuffle=True,
                                                                  stratify=df["sentiment"])

    print("Shape of tweets_train:", tweets_train.shape)
    print("Shape of tweets_test:", tweets_test.shape)
    print("Shape of sentiment_train:", y_train.shape)
    print("Shape of sentiment_test:", y_test.shape, "\n")

    # fit should be only to the training set, better for corpus generalisation
    vectorize_corpus = get_bow(tweets_train, ngram, stopwords)
    X_train = vectorize_corpus.transform(tweets_train)
    X_test = vectorize_corpus.transform(tweets_test)

    # visualize dataset
    #tsne = TSNEVisualizer()
    #tsne.fit(X_train, y_train)
    #tsne.show()

    print(f"Document term matrix :\n\n{X_train.toarray()}\n")
    print(f"Nr. of Doc: {X_train.shape[0]} Nr. of Feature: {X_train.shape[1]}\n")
    dtm_min_max = np.array(X_train.toarray())
    print(f"min value in matrix: {dtm_min_max.min()} max value in matrix: {dtm_min_max.max()}")

    print("[Multinomial Naive Bayes]\n")
    t0mnb = time.time()
    mnb_clf(X_train, X_test, y_train, y_test)
    t1mnb = time.time()
    time_linear_train_mnb = t1mnb - t0mnb
    # results
    print(f"Training time: {time_linear_train_mnb}\n")

    print("[Support Vector Machine]\n")
    ms = MaxAbsScaler()
    t0svm = time.time()
    svm_clf(ms.fit_transform(X_train), ms.transform(X_test), y_train, y_test)
    t1svn = time.time()
    time_linear_train_svm = t1svn - t0svm
    # result
    print(f"Training time: {time_linear_train_svm}\n")

    print("[Random Forest]\n")
    t0rf = time.time()
    rf_clf(X_train, X_test, y_train, y_test)
    t1rf = time.time()
    time_linear_train_rt = t1rf - t0rf
    # results
    print(f"Training time: {time_linear_train_rt}\n")

# features are a collection of metafeatures
def model_meta():
    df = pd.read_csv(train_path_meta, sep=",")
    # df.drop(df.query("sentiment == \"neutral\"").sample(frac=.3).index, inplace=True)

    # avoid overfitting, split into training and test data
    tweets_train, tweets_test, y_train, y_test = train_test_split(df.loc[:, "anzahl_wort":"sentiWS_score"],
                                                                  df["sentiment"],
                                                                  test_size=0.20, random_state=1, shuffle=True,
                                                                  stratify=df["sentiment"])

    print("Shape of tweets_train:", tweets_train.shape)
    print("Shape of tweets_test:", tweets_test.shape)
    print("Shape of sentiment_train:", y_train.shape)
    print("Shape of sentiment_test:", y_test.shape, "\n")

    # visualize dataset
    #classes = ['neutral', 'negative', "positive"]
    #le = preprocessing.LabelEncoder()
    #visualizer = PCA(scale=True, projection=3, classes=classes)
    #visualizer.fit_transform(tweets_train, le.fit_transform(y_train))
    #visualizer.show()
    feature_importance(tweets_train, y_train)

    print("[Multinomial Naive Bayes]\n")
    mms = MinMaxScaler()
    t0mnb = time.time()
    mnb_clf(mms.fit_transform(tweets_train), mms.transform(tweets_test), y_train, y_test)
    t1mnb = time.time()
    time_linear_train_mnb = t1mnb - t0mnb
    # results
    print(f"Training time: {time_linear_train_mnb}\n")

    print("[Support Vector Machine]\n")
    ms = MaxAbsScaler()
    t0svm = time.time()
    svm_clf(ms.fit_transform(tweets_train), ms.transform(tweets_test), y_train, y_test)
    t1svn = time.time()
    time_linear_train_svm = t1svn - t0svm
    # result
    print(f"Training time: {time_linear_train_svm}\n")

    print("[Random Forest]\n")
    t0rf = time.time()
    rf_clf(tweets_train, tweets_test, y_train, y_test)
    t1rf = time.time()
    time_linear_train_rt = t1rf - t0rf
    # results
    print(f"Training time: {time_linear_train_rt}\n")

# features are a concat featurevector of metafeatures and tfidf
def model_meta_tfidf(ngram, stopwords, feature_space):

    df_meta = pd.read_csv(train_path_meta, delimiter=",", index_col=False, header=0)

    # avoid overfitting, split into training and test data
    tweets_train, tweets_test, y_train, y_test = train_test_split(df_meta.loc[:, "tweet_processed":"sentiWS_score"],
                                                                  df_meta["sentiment"],
                                                                  test_size=0.20, random_state=1, shuffle=True,
                                                                  stratify=df_meta["sentiment"])

    print("Shape of tweets_train:", tweets_train.shape)
    print("Shape of tweets_test:", tweets_test.shape)
    print("Shape of sentiment_train:", y_train.shape)
    print("Shape of sentiment_test:", y_test.shape, "\n")

    vectorize_train = get_tfidf_vector(tweets_train["tweet_processed"], ngram, stopwords)
    transform_train = vectorize_train.transform(tweets_train["tweet_processed"]).toarray()
    transform_test = vectorize_train.transform(tweets_test["tweet_processed"]).toarray()

    X_train = np.concatenate((transform_train, tweets_train[feature_space]), axis=1)
    X_test = np.concatenate((transform_test, tweets_test[feature_space]), axis=1)
    print(f"Featureraum Train\n:{X_train}\n")
    print(f"Featureraum Test\n:{X_test}\n")

    print("[Multinomial Naive Bayes]\n")
    mms = MinMaxScaler()
    t0mnb = time.time()
    mnb_clf(mms.fit_transform(X_train), mms.transform(X_test), y_train, y_test)
    t1mnb = time.time()
    time_linear_train_mnb = t1mnb - t0mnb
    # results
    print(f"Training time: {time_linear_train_mnb}\n")

    print("[Support Vector Machine]\n")
    sc = MaxAbsScaler()
    t0svm = time.time()
    svm_clf(sc.fit_transform(X_train), sc.transform(X_test), y_train, y_test)
    t1svn = time.time()
    time_linear_train_svm = t1svn - t0svm
    # result
    print(f"Training time: {time_linear_train_svm}\n")

    print("[Random Forest]\n")
    t0rf = time.time()
    rf_clf(X_train, X_test, y_train, y_test)
    t1rf = time.time()
    time_linear_train_rt = t1rf - t0rf
    # results
    print(f"Training time: {time_linear_train_rt}\n")

def model_meta_bow(ngram, stopwords, feature_space):

    df_meta = pd.read_csv(train_path_meta, delimiter=",", index_col=False, header=0)

    # avoid overfitting, split into training and test data
    tweets_train, tweets_test, y_train, y_test = train_test_split(df_meta.loc[:, "tweet_processed":"sentiWS_score"],
                                                                  df_meta["sentiment"],
                                                                  test_size=0.20, random_state=1, shuffle=True,
                                                                  stratify=df_meta["sentiment"])

    print("Shape of tweets_train:", tweets_train.shape)
    print("Shape of tweets_test:", tweets_test.shape)
    print("Shape of sentiment_train:", y_train.shape)
    print("Shape of sentiment_test:", y_test.shape, "\n")

    vectorize_train = get_bow(tweets_train["tweet_processed"], ngram, stopwords)
    transform_train = vectorize_train.transform(tweets_train["tweet_processed"]).toarray()
    transform_test = vectorize_train.transform(tweets_test["tweet_processed"]).toarray()

    X_train = np.concatenate((transform_train, tweets_train[feature_space]), axis=1)
    X_test = np.concatenate((transform_test, tweets_test[feature_space]), axis=1)
    print(f"Featureraum Train\n:{X_train}\n")
    print(f"Featureraum Test\n:{X_test}\n")

    print("[Multinomial Naive Bayes]\n")
    mms = MinMaxScaler()
    t0mnb = time.time()
    mnb_clf(mms.fit_transform(X_train), mms.transform(X_test), y_train, y_test)
    t1mnb = time.time()
    time_linear_train_mnb = t1mnb - t0mnb
    # results
    print(f"Training time: {time_linear_train_mnb}\n")

    print("[Support Vector Machine]\n")
    sc = MaxAbsScaler()
    t0svm = time.time()
    svm_clf(sc.fit_transform(X_train), sc.transform(X_test), y_train, y_test)
    t1svn = time.time()
    time_linear_train_svm = t1svn - t0svm
    # result
    print(f"Training time: {time_linear_train_svm}\n")

    print("[Random Forest]\n")
    t0rf = time.time()
    rf_clf(X_train, X_test, y_train, y_test)
    t1rf = time.time()
    time_linear_train_rt = t1rf - t0rf
    # results
    print(f"Training time: {time_linear_train_rt}\n")

def rf_model_selection():

    df = pd.read_csv(train_path, sep=",")

    # avoid overfitting, split into training and test data
    tweets_train, tweets_test, y_train, y_test = train_test_split(df["tweet_processed"], df["sentiment"],
                                                                  test_size=0.20, random_state=1, shuffle=True,
                                                                  stratify=df["sentiment"])

    print("Shape of tweets_train:", tweets_train.shape)
    print("Shape of tweets_test:", tweets_test.shape)
    print("Shape of sentiment_train:", y_train.shape)
    print("Shape of sentiment_test:", y_test.shape, "\n")

    pipeline = Pipeline([
        ('vect', CountVectorizer(stop_words=stopwords, min_df=5)),
        ('tfidf', TfidfTransformer()),
        ('clf', RandomForestClassifier()),
    ])
    # Set the parameters by cross-validation
    # n_estimators - number of trees
    # max_depth - maxdepth of trees
    # min_samples_split - The minimum number of samples required to split an internal node:
    # min_samples _leaf - default
    # criterion : gini, entrophy
    parameters = {
        "clf__n_estimators": [200, 500, 900],
        "clf__max_features": [11, 12, 13],
        "clf__criterion": ["gini", "entropy"]
    }


    clf = GridSearchCV(
            pipeline, parameters, scoring='f1', cv=5, n_jobs=-1, verbose=10
    )
    # print(clf.get_params().keys())
    clf.fit(tweets_train, y_train)

    print("Best parameters set found on development set:")
    print()
    print(clf.best_params_)
    print()
    print("Grid scores on development set:")
    print()
    means = clf.cv_results_['mean_test_score']
    stds = clf.cv_results_['std_test_score']
    for mean, std, params in zip(means, stds, clf.cv_results_['params']):
        print("%0.3f (+/-%0.03f) for %r"
                % (mean, std * 2, params))
    print()

    print("Detailed classification report:")
    print()
    print("The model is trained on the full development set.")
    print("The scores are computed on the full evaluation set.")
    print()
    y_true, y_pred = y_test, clf.predict(tweets_test)
    print(classification_report(y_true, y_pred))
    print()

def svm_model_selection():
    df = pd.read_csv(train_path, sep=",")

    # avoid overfitting, split into training and test data
    tweets_train, tweets_test, y_train, y_test = train_test_split(df["tweet_processed"], df["sentiment"],
                                                                  test_size=0.20, random_state=1, shuffle=True,
                                                                  stratify=df["sentiment"])

    print("Shape of tweets_train:", tweets_train.shape)
    print("Shape of tweets_test:", tweets_test.shape)
    print("Shape of sentiment_train:", y_train.shape)
    print("Shape of sentiment_test:", y_test.shape, "\n")

    pipeline = Pipeline([
        ('vect', CountVectorizer(stop_words=stopwords, min_df=5)),
        ('tfidf', TfidfTransformer()),
        ('clf', SVC()),
    ])

    # Set the parameters by cross-validation
    # Small Weight: Smaller C value, larger penalty for misclassified examples.
    # Larger Weight: Larger C value, smaller penalty for misclassified examples.
    # Gamma für rbf

    parameters = [
        {
            'clf__kernel': ['rbf'],
            'clf__gamma': [0.0001, 0.001, 0.01],
            'clf__C': [1, 10, 100]
         },
        {
            'clf__kernel': ['linear'],
            'clf__C': [1, 10, 100]
        }
    ]

    clf = GridSearchCV(
        pipeline, parameters, scoring='f1', cv=5, n_jobs=-1, verbose=10
    )
    # print(clf.get_params().keys())
    clf.fit(tweets_train, y_train)

    print("Best parameters set found on development set:")
    print()
    print(clf.best_params_)
    print()
    print("Grid scores on development set:")
    print()
    means = clf.cv_results_['mean_test_score']
    stds = clf.cv_results_['std_test_score']
    for mean, std, params in zip(means, stds, clf.cv_results_['params']):
        print("%0.3f (+/-%0.03f) for %r"
              % (mean, std * 2, params))
    print()

    print("Detailed classification report:")
    print()
    print("The model is trained on the full development set.")
    print("The scores are computed on the full evaluation set.")
    print()
    y_true, y_pred = y_test, clf.predict(tweets_test)
    print(classification_report(y_true, y_pred))
    print()

def feature_importance(X_train, y_train):
    feature = X_train.columns.tolist()

    random_forest = RandomForestClassifier(random_state=1, n_estimators=900, max_features=12, class_weight="balanced")
    random_forest.fit(X_train, y_train)
    importances = random_forest.feature_importances_
    indices = np.argsort(importances)[::-1] #slice array in desc order

    # Print the feature ranking
    print("Feature ranking:")
    for f in range(X_train.shape[1]):
        print("{}. feature {} {}".format(f + 1, feature[indices[f]], importances[indices[f]]))

    viz = FeatureImportances(random_forest)
    viz.fit(X_train, y_train)
    viz.show()

def pickle_model(ngram, stopwords, feature_space):
    df_meta = pd.read_csv(train_path_meta, delimiter=",", index_col=False, header=0)

    # avoid overfitting, split into training and test data
    tweets_train, tweets_test, y_train, y_test = train_test_split(df_meta.loc[:, "tweet_processed":"sentiWS_score"],
                                                                  df_meta["sentiment"],
                                                                  test_size=0.20, random_state=1, shuffle=True,
                                                                  stratify=df_meta["sentiment"])

    print("Shape of tweets_train:", tweets_train.shape)
    print("Shape of tweets_test:", tweets_test.shape)
    print("Shape of sentiment_train:", y_train.shape)
    print("Shape of sentiment_test:", y_test.shape, "\n")

    vectorize_train = get_bow(tweets_train["tweet_processed"], ngram, stopwords)
    transform_train = vectorize_train.transform(tweets_train["tweet_processed"]).toarray()
    transform_test = vectorize_train.transform(tweets_test["tweet_processed"]).toarray()

    X_train = np.concatenate((transform_train, tweets_train[feature_space]), axis=1)
    X_test = np.concatenate((transform_test, tweets_test[feature_space]), axis=1)
    print(f"Featureraum Train:\n{X_train}\nshape:{X_train.shape}\n")
    print(f"Featureraum Test:\n{X_test}\nshape:{X_train.shape}\n")

    print("[Support Vector Machine]\n")
    sc = MaxAbsScaler()
    linear_svm = svm.LinearSVC(class_weight="balanced", random_state=1, C=1)
    linear_svm.fit(sc.fit_transform(X_train), y_train)

    # predict class y
    predictions = linear_svm.predict(sc.transform(X_test))
    print(f"Confusion Matrix:\n{confusion_matrix(y_test, predictions)}\n")
    print(f"Classification report:\n{classification_report(y_test, predictions)}\n")
    print(f"Accuracy: \n{accuracy_score(y_test, predictions)}\n")

    print("Save the Model")
    # pickling the vectorizer
    pickle.dump(vectorize_train, open('model/bow_vectorizer.sav', 'wb'))
    print("Vectorizer Done")
    # pickling the model
    pickle.dump(linear_svm, open('model/sentiment_classifier.sav', 'wb'))
    print("SVM Done")

def pred_new_data():

    # load
    df = pd.read_csv("resources/corpus/pred_corpus_meta.csv", sep=",")
    vectorizer = pickle.load(open("model/bow_vectorizer.sav", 'rb'))
    svm_model = pickle.load(open("model/sentiment_classifier.sav", 'rb'))

    # vectorizer
    transform = vectorizer.transform(df["tweet_processed"])
    concat = np.concatenate((transform.toarray(), df[all_features]), axis=1)
    print(f"Matrix: \n{concat}\n")

    #pred
    print("Pred new tweets")
    sc = MaxAbsScaler()
    df["sentiment"] = svm_model.predict(sc.fit_transform(concat))
    df.to_csv("resources/corpus/pred_corpus_meta.csv", encoding='utf-8', index=False, sep=",")
    print("Done")


if __name__ == "__main__":

    # print statistics of train and pred corpus
    print("Show some Statistics")
    print_statistics()

    # uncomment, but will take a long time
    #print("Hyperparam tuning ....")
    #svm_model_selection()
    #rf_model_selection()

    print("Analyse Model Performance...")
    model_meta()
    model_tfidf((1, 1), stopwords)
    model_bow((1, 1), stopwords)
    model_meta_bow((1, 2), stopwords, all_features)
    model_meta_tfidf((1, 1), stopwords, all_features)

    # uncomment to check functionality, model already saved in model/*
    print("Save Final Model")
    # pickle_model((1, 2), stopwords, all_features)

    # uncomment to check functionality, model already saved in resources/corpus/pred*
    print("Pred new data")
    #pred_new_data()

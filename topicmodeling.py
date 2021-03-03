import re
import pandas as pd
from pprint import pprint

# Gensim
from gensim.corpora import Dictionary
from gensim.utils import simple_preprocess
from gensim.models import CoherenceModel, LdaModel
from gensim.models.phrases import Phrases, Phraser

# to see log of the model
#import logging
#logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

# spacy for postag
import spacy

# Plotting tools
import pyLDAvis
import pyLDAvis.gensim
import matplotlib.pyplot as plt
from wordcloud import WordCloud

# LDA: Jedes Dokument ist eine Sammlung von Topics und jedes Topic ist eine Ansammlung von Keywords
# In der Main wird das Skript gesteuert
# Quelle: https://www.machinelearningplus.com/nlp/topic-modeling-gensim-python/
# https://www.machinelearningplus.com/nlp/topic-modeling-visualization-how-to-present-results-lda-models/
# https://medium.com/@osas.usen/topic-extraction-from-tweets-using-lda-a997e4eb0985
# aufgerufen  11.12.2020, 19:00

nlp = spacy.load('de_core_news_sm', disable=['parser', 'ner'])
stop_words = nlp.Defaults.stop_words
allowed_postags = ['NOUN', 'VERB']

# Import Tweets
df = pd.read_csv("resources/corpus/pred_corpus_meta.csv")
neutral_tweets = df[(df.sentiment == "neutral")].tweet_processed.values.tolist()
negative_tweets = df[(df.sentiment == "negative")].tweet_processed.values.tolist()
positive_tweets = df[(df.sentiment == "positive")].tweet_processed.values.tolist()

# Tweets aus Berlin
df_new = df[df["user_location"].isin(["Berlin", "Berlin, Deutschland"])]
df_be_neutral = df_new[(df_new.sentiment == "neutral")].tweet_processed.values.tolist()
df_be_neg = df_new[(df_new.sentiment == "negative")].tweet_processed.values.tolist()
df_be_pos = df_new[(df_new.sentiment == "positive")].tweet_processed.values.tolist()

def find_mentioned(tweet):
    return re.findall('(@[A-Za-z]+[A-Za-z0-9-_]+)', tweet)

def find_hashtags(tweet):
    return re.findall('(#[A-Za-z]+[A-Za-z0-9-_]+)', tweet)

def count_mentions_hashtags():
    df['mentioned'] = df.tweet.apply(find_mentioned)
    df['hashtags'] = df.tweet.apply(find_hashtags)

    print(df.head())
    # take the rows from the hashtag columns where there are actually hashtags
    hashtags_list_df = df.loc[df.hashtags.apply(lambda hashtags_list: hashtags_list != []), ['hashtags']]
    print("Hashtag in the corpus:")
    print(hashtags_list_df, "\n")
    # create dataframe where each use of hashtag gets its own row
    flattened_hashtags_df = pd.DataFrame([hashtag for hashtags_list in hashtags_list_df.hashtags for hashtag in hashtags_list], columns=['hashtag'])
    #print(flattened_hashtags_df.head())
    # number of unique hashtags
    print(f"Nr. of Unique hashtags:{flattened_hashtags_df['hashtag'].unique().size}\n")
    popular_hashtags = flattened_hashtags_df.groupby('hashtag').size().reset_index(name='counts').sort_values('counts', ascending=False)\
                                            .reset_index(drop=True)
    print(f"Top 10 Popular hashtags:\n{popular_hashtags[:10]}\n")

    # take the rows from the hashtag columns where there are actually hashtags
    mentioned_list_df = df.loc[df.mentioned.apply(lambda hashtags_list: hashtags_list != []), ['mentioned']]
    print("Mentioned in the corpus:")
    print(mentioned_list_df, "\n")
    # create dataframe where each use of hashtag gets its own row
    flattened_mentioned_df = pd.DataFrame([mentioned for mentioned_list in mentioned_list_df.mentioned for mentioned in mentioned_list],
        columns=['mentioned'])
    #print(flattened_mentioned_df.head())
    print(f"Nr. of Unique Mentions: {flattened_mentioned_df['mentioned'].unique().size}\n")
    # count of appearances of each hashtag
    popular_mentioned = flattened_mentioned_df.groupby('mentioned').size().reset_index(name='counts').sort_values('counts', ascending=False) \
        .reset_index(drop=True)
    print(f"Top 10 Popular mention:\n{popular_mentioned[:10]}")

def tokenize(tweets):
    tokenize_tweet = [simple_preprocess(str(tweet)) for tweet in tweets]
    return tokenize_tweet

# Join uni, bi and trigrams together, remove stopwords and only allow specified tags
def process_words(tweets, stop_words, postags):
    # Build the bigram and trigram models, phraser better memory usage
    bigram = Phrases(tweets, min_count=10)
    trigram = Phrases(bigram[tweets])
    bigram_mod = Phraser(bigram)
    trigram_mod = Phraser(trigram)

    # See trigram example, will be appended in teh next function
    print(f"Example of phrases: {trigram_mod[bigram_mod[tweets[0]]]}")

    texts = [[word for word in tweet if word not in stop_words] for tweet in tweets] # remove stopwords
    texts = [bigram_mod[tweet] for tweet in texts]
    texts = [trigram_mod[bigram_mod[tweet]] for tweet in texts]
    texts_out = []
    for sent in texts:
        doc = nlp(" ".join(sent))
        texts_out.append([token.text for token in doc if token.pos_ in postags])
    return texts_out

def make_gensim_corpus(tweets):
    # 1.make gensim corpus (list of list)
    tweets_words = tokenize(tweets)
    print(f"Tokenize tweet: {tweets_words[:1]}")

    # 2. make gensim corpus with phrases
    tweets_ready = process_words(tweets_words, stop_words, allowed_postags)
    print(f"Tweet with Phrases, nouns and verbs: {tweets_ready[:1]}")

    # 3.Create Input for LDA: A dictionary and the Term frequency
    # Create Dictionary for input to LDA
    dictionary = Dictionary(tweets_ready)

    # Create Corpus: Term Document Frequency for input into LDA
    corpus = [dictionary.doc2bow(tweet) for tweet in tweets_ready]

    return tweets_ready, dictionary, corpus

def model(text_corpus, corpus, dictionary, num_topics, chunksize):

    # View the words and its ID in the dictionary

    word_counts = [[(dictionary[id], count) for id, count in line] for line in corpus]
    print(f"View the dictionary: {corpus[:1]}, {word_counts[:1]}")
    print('Number of unique words: %d' % len(dictionary))
    print('Number of Tweets: %d' % len(corpus))

    # LDA Model
    # Build LDA model
    lda_model = LdaModel(corpus=corpus,  # term frequency
                         id2word=dictionary,  # dictionary
                         num_topics=num_topics,  # k number of topics
                         random_state=100,
                         chunksize=chunksize,  # number of documents to process each time
                         passes=30,
                         iterations=400,
                         eval_every=1,
                         alpha='auto',
                         eta="auto",
                         per_word_topics=True)

    # Print the 10 most significant topics in a nice format with pprint
    compute_coherence(lda_model, text_corpus, dictionary)
    pprint(lda_model.print_topics(num_topics=-1, num_words=10))
    print("\nTop Topics:\n")
    pprint(lda_model.print_topics(num_topics=10, num_words=10))
    print("\nTopics ordered by Coherence c_v:\n")
    pprint(lda_model.top_topics(texts=text_corpus, dictionary=dictionary, coherence="c_v", topn=10))

    # visualize
    word_cloud(lda_model)
    if num_topics > 1:
        vis = pyLDAvis.gensim.prepare(lda_model, corpus, dictionary, sort_topics=False)
        pyLDAvis.show(vis)

def topics_modelling(tweets, topics_nr, chunksize):
    "make gensim input and give it to model"
    tweets_ready, dictionary, corpus = make_gensim_corpus(tweets)
    model(tweets_ready, corpus, dictionary, topics_nr, chunksize)

def compute_coherence(model, data, dictionary):
    # Compute Coherence Score
    coherence_model_lda = CoherenceModel(model=model, texts=data, dictionary=dictionary, coherence='c_v')
    coherence_lda = coherence_model_lda.get_coherence()
    print('\nCoherence Score: ', coherence_lda)

def compute_coherence_values(dictionary, corpus, texts, limit, start, step):
    "compute coherence to specify optimal nr of topics"
    coherence_values = []
    for num_topics in range(start, limit, step):
        model = LdaModel(corpus=corpus, id2word=dictionary, num_topics=num_topics, random_state=100)
        coherencemodel = CoherenceModel(model=model, texts=texts, dictionary=dictionary, coherence='c_v')
        coherence_values.append(coherencemodel.get_coherence())

    x = range(start, limit, step)
    plt.plot(x, coherence_values)
    plt.xlabel("Num Topics")
    plt.ylabel("Coherence score")
    plt.legend(("coherence_values"), loc='best')
    plt.show()

# compute coherence for each sentiment of tweets
def analyse_topic_nr():
    for tweets in [neutral_tweets, negative_tweets, positive_tweets]:
        tweets_ready, dictionary, corpus = make_gensim_corpus(tweets)
        compute_coherence_values(dictionary=dictionary, corpus=corpus, texts=tweets_ready, start=2, limit=96, step=8)


def word_cloud(model):

    cloud = WordCloud(background_color='white',
                      width=2500,
                      height=1800,
                      max_words=10,
                      prefer_horizontal=1.0)

    for t in range(model.num_topics):
        plt.figure()
        plt.imshow(cloud.fit_words(dict(model.show_topic(t, 10))))
        plt.axis("off")
        plt.title("Topic #" + str(t+1))
        plt.show()

if __name__ == "__main__":

    count_mentions_hashtags()
    analyse_topic_nr()

    # Deutschland
    topics_modelling(neutral_tweets, topics_nr=30, chunksize=30000)
    topics_modelling(negative_tweets, topics_nr=20, chunksize=8000)
    topics_modelling(positive_tweets, topics_nr=1, chunksize=200)

    # Berlin
    topics_modelling(df_be_neutral, topics_nr=2, chunksize=1000)
    topics_modelling(df_be_neg, topics_nr=3, chunksize=100)
    topics_modelling(df_be_pos, topics_nr=1, chunksize=10)
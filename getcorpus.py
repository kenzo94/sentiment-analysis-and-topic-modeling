#import libraries
import pandas as pd
import csv
import tweepy
import glob
from tweepy import OAuthHandler

# In der Main wird das Skript gesteuert

#twitter credentials
consumer_key = "pvjiXZ4ojM4lf6fkMNDDWCfMl"
consumer_secret = "VRps2wCrowQtt3PsVRm2fXt2K4OUA3MXhvONex7fdPapHyxlj3"
access_token = "1298262098124787713-0ahmPtCpMAFtenOcv1gEHnlGDexTFq"
access_secret = "2dCzzBJRl0I6C9fl8zTFhqBIybeI1n6uu1uNK0oNK7EDT"

#Authenticate credentials
auth = OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_secret)

api = tweepy.API(auth,wait_on_rate_limit=True)


def get_corpus_sb10k(file_path_in,file_path_out):

    corpus_file = open(file_path_in, "r+", newline="")
    read_tsv = csv.reader(corpus_file, delimiter="\t")

    list = []

    for row in read_tsv:
        try:
            tweet = api.get_status(int(row[0]), tweet_mode="extended")
            row[0] = tweet.full_text.replace("\n", "")
            list.append(row[0:2])
        except tweepy.error.TweepError as e:
            print(e)

    corpus_file.close()

    with open(file_path_out, 'wt', encoding='utf-8', newline="") as out_file:
        tsv_writer = csv.writer(out_file, delimiter=",")
        tsv_writer.writerow(["tweet", "sentiment"])
        for row in list:
            tsv_writer.writerow(row)


def get_corpus_GTTC(file_path_in):
    df = pd.read_csv(file_path_in, usecols=["ID", "Stance"])
    list = []
    for row in df["ID"]:
        try:
            status = api.get_status(row, tweet_mode="extended")
            list.append(status.full_text.replace("\n", ""))
        except tweepy.error.TweepError as e:
            print(f"Problems with API: {e}")
            list.append(None)
            continue
    df["ID"] = pd.Series(list)
    return df

def get_german_twitter_sentiment_corpus(file_path_in):
    df = pd.read_csv(file_path_in, usecols=["TweetID", "HandLabel"])
    list = []
    for row in df["TweetID"]:
        try:
            status = api.get_status(row, tweet_mode="extended")
            list.append(status.full_text.replace("\n", ""))
        except tweepy.error.TweepError as e:
            print(f"Problems with API: {e}")
            list.append(None)
            continue
    df["TweetID"] = pd.Series(list)
    return df

def get_dai_corpus(file_path_in, file_path_out):
    with open(file_path_in, 'r', encoding='utf-8', newline="") as in_file, \
            open(file_path_out, 'w', encoding='utf-8', newline="") as out_file:
        tsv_reader = csv.DictReader(in_file, delimiter="\t")
        tsv_writer = csv.writer(out_file, delimiter="\t")
        tsv_writer.writerow(["tweet", "sentiment"])
        for row in tsv_reader:
            tsv_writer.writerow([row["tweet"], row["sentiment"]])

def get_germeval_corpus(file_path_in, file_path_out):
    with open(file_path_in, 'r', encoding='utf-8', newline="") as in_file, \
            open(file_path_out, "w", encoding='utf-8', newline="") as out_file:
        tsv_reader = csv.reader(in_file, delimiter="\t")
        tsv_writer = csv.writer(out_file, delimiter="\t")
        tsv_writer.writerow(["tweet", "relevance", "sentiment"])
        for row in tsv_reader:
            if "twitter" in row[0]:
                tsv_writer.writerow(row[1:4])

def get_pred_corpus(path):
    files = glob.glob("tweets\*.csv")
    dfs = [pd.read_csv(f, header=0, sep=",") for f in files]

    df = pd.concat(dfs, ignore_index=True)
    print(df.shape)
    df.to_csv(path, sep=",", index=False)

if __name__ == "__main__":

    print("Getting the Tweets ....")

    #Get German Twitter Titling Corpus
    df = get_corpus_GTTC("resources/raw_corpus/GTTC.csv")
    df.to_csv("./resources/corpus/gttc_final.csv", encoding='utf-8', index=False)

    # Get SB-10K Corpus
    get_corpus_sb10k("./resources/raw_corpus/corpus_v1.0.tsv", "resources/corpus/SB-10k.csv")

    # Get German Twitter Sentiment Corpus
    df2 = get_german_twitter_sentiment_corpus("resources/raw_corpus/German_Twitter_sentiment.csv")
    df2.to_csv("./resources/corpus/German_Twitter_sentiment_final.csv", encoding="utf-8", index=False)

    #get dai corpus
    get_dai_corpus("resources/raw_corpus/de_sentiment_agree2.tsv", "resources/corpus/de_sentiment_agree2_final.tsv")

    #get germeval2017 corpus
    get_germeval_corpus("resources/raw_corpus/train_v1.4.tsv", "resources/corpus/germeval_2017.tsv")

    #get pred corpus
    get_pred_corpus("resources/raw_corpus/tweets.csv")
    df = pd.read_csv("resources/raw_corpus/tweets.csv", sep=",", header=0)
    print(df.count(0))


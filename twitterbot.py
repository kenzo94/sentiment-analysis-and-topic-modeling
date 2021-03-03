import tweepy
from datetime import datetime
import csv
from tweepy import OAuthHandler

# In der Main wird das Skript gesteuert
'''
Quelle: https://www.pluralsight.com/guides/building-a-twitter-bot-with-python
http://docs.tweepy.org/en/latest/index.html
'''

# twitter credentials
consumer_key = "pvjiXZ4ojM4lf6fkMNDDWCfMl"
consumer_secret = "VRps2wCrowQtt3PsVRm2fXt2K4OUA3MXhvONex7fdPapHyxlj3"
access_token = "1298262098124787713-0ahmPtCpMAFtenOcv1gEHnlGDexTFq"
access_secret = "2dCzzBJRl0I6C9fl8zTFhqBIybeI1n6uu1uNK0oNK7EDT"


def initiate_api():
    try:
        # Authenticate credentials
        auth = OAuthHandler(consumer_key, consumer_secret)
        auth.set_access_token(access_token, access_secret)
        api = tweepy.API(auth, wait_on_rate_limit=True, wait_on_rate_limit_notify=True)
        return api
    except tweepy.error.TweepError as e:
        print(f"Problems with API: {e}")
        return None


# rate limit: 450 request/ 15min
def get_tweets_with_search(api, query, count):
    tweets = []
    try:
        for status in tweepy.Cursor(
                api.search,
                q=query,
                lang="de",
                tweet_mode="extended"
        ).items(count):
                tweets.append([status.id, status.created_at.strftime("%d-%m-%Y"), status.user.screen_name,
                               status.full_text.replace("\n", ""), status.source, status.retweet_count,
                               status.user.location, status.place, status.coordinates
                               ])
    except tweepy.error.TweepError as e:
        print(f"Problems with API: {e}")
        return None
    return tweets

#write tweets to a csv
def write_to_csv(tweets):
    today = datetime.today().strftime("%Y-%m-%d")
    with open("tweets/" + today + "-tweets.csv", "a+", encoding="utf-8", newline="") as file_tweets:
        writer = csv.writer(file_tweets)
        writer.writerow(["tweet_id", "date", "username", "tweet", "source", "retweet_count", "user_location",
                         "user_place", "coordinates"])
        for tweet in tweets:
            writer.writerow(tweet)

# https://data.gesis.org/tweetscov19/#dataset -> important entities: Wuhan, Hydroxychloroquine, Corona, Covid-19
# https://data.gesis.org/tweetscov19/keywords.txt -> keywords: coronavirusdeutschland, covid tests, covid19 epidemic

if __name__ == "__main__":
    # print all possible things from the tweet
    print("Get Tweets...")
    query = "corona OR covid-19 OR covid19 OR coronavirusdeutschland OR lockdown -filter:retweets AND -filter:replies"
    write_to_csv(get_tweets_with_search(initiate_api(), query, 27))
    print("Done!")
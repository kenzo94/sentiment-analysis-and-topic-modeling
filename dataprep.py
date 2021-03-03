import pandas as pd
import re
import string
import spacy
import csv
import emoji
from spacy_sentiws import spaCySentiWS
from spacymoji import Emoji
from itertools import groupby
from nltk.corpus import stopwords
from nltk import ngrams
from HanTa import HanoverTagger as ht

# In der Main wird das Skript gesteuert

# 5 train corpus var
gttc_path = "resources/corpus/gttc_final.csv"
sb10k_path = "resources/corpus/SB-10k.csv"
gts_path = "resources/corpus/German_Twitter_sentiment_final.csv"
dai_path = "resources/corpus/de_sentiment_agree2_final.tsv"
germeval_path = "resources/corpus/germeval_2017.tsv"

df_cols = ["tweet", "sentiment"]

df_final_cols = ["tweet", "sentiment", "tweet_processed", "anzahl_wort", "anzahl_caps", "anzahl_punc", "anzahl_qm",
                 "anzahl_em", "lange_seq_em_qm", "lange_seq_qm_em", "lange_r_letters", "lange_r_punc","last_punc", "last_em",
                 "last_qm", "anzahl_slangs", "anzahl_emoji", "anzahl_pos_emoji", "anzahl_neg_emoji", "anzahl_pos_smiley",
                 "anzahl_neg_smiley", "last_emoji_smiley", "anzahl_verb", "anzahl_nomen", "anzahl_pron", "anzahl_adj",
                 "anzahl_intj", "anzahl_cconj", "anzahl_negation", "anzahl_verstarker", "anzahl_shift_pos",
                 "anzahl_shift_neg", "anzahl_shift_gen", "anzahl_neutral_uni", "anzahl_pos_uni", "anzahl_pos_bi",
                 "anzahl_pos_tri", "anzahl_neg_uni", "anzahl_neg_bi", "anzahl_neg_tri", "sentiWS_score"]

df_pred_col = ["date", "user_location", "tweet", "tweet_processed", "anzahl_wort", "anzahl_caps", "anzahl_punc", "anzahl_qm",
                 "anzahl_em", "lange_seq_em_qm", "lange_seq_qm_em", "lange_r_letters", "lange_r_punc","last_punc", "last_em",
                 "last_qm", "anzahl_slangs", "anzahl_emoji", "anzahl_pos_emoji", "anzahl_neg_emoji", "anzahl_pos_smiley",
                 "anzahl_neg_smiley", "last_emoji_smiley", "anzahl_verb", "anzahl_nomen", "anzahl_pron", "anzahl_adj",
                 "anzahl_intj", "anzahl_cconj", "anzahl_negation", "anzahl_verstarker", "anzahl_shift_pos",
                 "anzahl_shift_neg", "anzahl_shift_gen", "anzahl_neutral_uni", "anzahl_pos_uni", "anzahl_pos_bi",
                 "anzahl_pos_tri", "anzahl_neg_uni", "anzahl_neg_bi", "anzahl_neg_tri", "sentiWS_score"]
sep_tsv = "\t"
sep_csv = ","

# lexicons var path
emoticon_path = "resources/lexicon/emoticons.txt"
smiley_path = "resources/lexicon/smiley.txt"
verstarker_lex_path = "resources/lexicon/verstärker.txt"
slang_lex_path = "resources/lexicon/internet_slang.txt"
shifter_lex_pos_path = "resources/lexicon/shifter_positiv.txt"
shifter_lex_gen_path = "resources/lexicon/shifter_general.txt"
shifter_lex_neg_path = "resources/lexicon/shifter_negativ.txt"
negation_lex_path = "resources/lexicon/negation_words.txt"
neutral_lex_path = "resources/lexicon/GermanPolarityClues-Neutral-Lemma-21042012.tsv"
pos_uni_lex_path = "resources/lexicon/pos-unigram.txt"
pos_bi_lex_path = "resources/lexicon/pos-bigram.txt"
pos_tri_lex_path = "resources/lexicon/pos-trigram.txt"
neg_uni_lex_path = "resources/lexicon/neg-unigram.txt"
neg_bi_lex_path = "resources/lexicon/neg-bigram.txt"
neg_tri_lex_path = "resources/lexicon/neg-trigram.txt"
sentiWS_path = "resources/raw_lexicons/"

#train corpus
train_corpus_path = "resources/corpus/train_corpus.csv"
train_meta_corpus_path = "resources/corpus/train_corpus_meta.csv"

#pred corpus
pred_corpus_path = "resources/corpus/pred_corpus.csv"
pred_meta_corpus_path = "resources/corpus/pred_corpus_meta.csv"

#spacy model for tokenizing, Emoji detection
nlp = spacy.load('de_core_news_lg', disable=['ner'])
sentiws = spaCySentiWS(sentiws_path=sentiWS_path)
nlp.add_pipe(sentiws)
spacy_emoji = Emoji(nlp, merge_spans=True)
nlp.add_pipe(spacy_emoji, first=True)

#nltk stopword list better
stopwords = stopwords.words("german")

#load smiley dictionary: dictionaries {key:value}; line has to be strip because of multiple whitespace
def get_smiley_dict(path):
    smiley_dic = {}
    with open(path, encoding="utf-8") as in_file:
        for line in in_file:
            values = line.strip().split()
            if values:
                smiley_dic[values[0]] = "".join(values[1:])
    return smiley_dic

#load emoticon dictionary, dictionaries {key:value}
def get_emoticon_dict(path):
    emoji_dic = {}
    with open(path, encoding="utf-8") as in_file:
        for line in in_file:
            key, val = line.split()
            emoji_dic[emoji.emojize(key)] = val
    return emoji_dic

# load the datasets from resource
def load_dataset(path, sep, cols):
    if path in [gts_path, sb10k_path, gttc_path]:
        df = pd.read_csv(path, sep)
        df.columns = cols
        df.sentiment = df.sentiment.str.lower()
        # remove na, NaN, duplicates columns, change neither to neutral
        df = df[(df.sentiment != "na")]
        df.dropna(subset=["tweet"], inplace=True)
        df.loc[df.sentiment == "neither", ["sentiment"]] = "neutral"
        df.drop_duplicates(subset=["tweet"], inplace=True)
    else:
        df = pd.read_csv(path, sep, usecols=df_cols)
        df.sentiment = df.sentiment.str.lower()
        # remove na, NaN, duplicates columns
        df = df[(df.sentiment != "na")]
        df.dropna(subset=["tweet"], inplace=True)
        df.drop_duplicates(subset=["tweet"], inplace=True)
    return df

#concat all datas in a dataframe
def concat_data(*datasets):
    frames = [dataset for dataset in datasets]
    result = pd.concat(frames, ignore_index=True)
    return result

# return all concat datasets and do some preprocessing: sentiment anpassen, doppelte tweets löschen
# Anzahl Daten:
def all_dataset():
    #concat all 5 corpus
    all_data = concat_data(load_dataset(gttc_path, sep_csv, df_cols), load_dataset(sb10k_path, sep_csv, df_cols),
                           load_dataset(dai_path, sep_tsv, df_cols), load_dataset(germeval_path, sep_tsv, df_cols),
                           load_dataset(gts_path, sep_csv, df_cols))
    all_data.sentiment = all_data.sentiment.str.lower()
    return all_data

# make a doc spacy object
def spacy(tweet):
    doc = nlp(tweet)
    return doc

# tokenizer with spacy
def tokenize(tweet):
    #print("tokenize")
    doc = nlp(tweet)
    word = [words.text for words in doc]
    #print(word)
    return word

# lemmatizer with HanTa
def lemmatize(tweet):
    # HANTA: load model from github, anaylyze sentence with tag_sent, return the lemmas from the sentence
    #print("lemmatize")
    lemmatized_sentence = ""
    if tweet != "":
        tagger = ht.HanoverTagger('morphmodel_ger.pgz')
        tags = tagger.tag_sent(tokenize(tweet), taglevel=1)
        #print(tags)
        lemmatized_sentence = ["".join(lemma[1]) for lemma in tags]
        #print(lemmatized_sentence)
    else:
        print("Tweet ist leer.")
    return " ".join(lemmatized_sentence)

# make grams to generate lex features
def make_gram(tweet, n):
    ngram = ngrams(tweet.split(), n)  # ngrams(list, number of ngram)
    return [" ".join(gram) for gram in ngram]

# zählt anzahl wörter, caps, punc, questionmarks, exclamation, sequenz em qm,
# sequenz qm em, repeated letters, repeated punc
# Featureset: 13 Features
def get_microblog_surface_feature(tweet):
    slang_lex = pd.read_csv(slang_lex_path, header=None)
    slang, r_letter, r_punc, last_token_qm, last_token_em= 0, 0, 0, 0, 0

    word = len(re.findall('[a-zA-ZÄäÖöÜüß]', tweet))
    cap = len(re.findall("[A-ZÄÖÜß]", tweet))
    punc = len(re.findall("[.]", tweet))
    qm = len(re.findall("[?]", tweet))
    em = len(re.findall("[!]", tweet))
    seq_em_qm = len(re.findall("([!])[?]+", tweet))
    seq_qm_em = len(re.findall("([?])[!]+", tweet))

    group_punc = [len(list(group)) for key, group in
                  groupby(tweet) if
                  key in ["!", "?", "."]]  # group by elements and get the number of elements > 1
    group_letter = [len(list(group)) for key, group in
                    groupby(tweet.lower()) if
                    key in string.ascii_letters.__add__("äüöß")]  # group by elements and get the number of elements > 3
    for element in group_letter:
        if element > 3:
            r_letter = r_letter + element
    for element in group_punc:
        if element > 1:
            r_punc = r_punc + element

    if spacy(tweet)[-1].is_punct:
        last_token_punc = 1
        if spacy(tweet)[-1].text == "!":
            last_token_em = 1
        elif spacy(tweet)[-1].text == "?":
            last_token_qm = 1
    else:
        last_token_punc = 0

    for slang_w in slang_lex[0]:  # element in slang dictionary ?
        tweet = re.sub(r"(\w)(\1{2,})", r"\1", tweet) # In case of repeated letters > 2
        for token in nlp(tweet):
            if token.text.lower() in slang_w.lower():
                slang = slang + 1

    return word, cap, punc, qm, em, seq_em_qm, seq_qm_em, r_letter, r_punc, last_token_punc, \
           last_token_em, last_token_qm, slang

# zählt emoticon, pos emoticon, pos smiley, neg smiley, neg smiley, last emo or smiley
# Featureset: 6 Features
def get_emoticon_features(tweet):
    pos_emo, neg_emo, pos_smiley, neg_smiley = 0, 0, 0, 0
    tokenize_tweet = tokenize(tweet)
    emoticon_dict = get_emoticon_dict(emoticon_path)
    smiley_dict = get_smiley_dict(smiley_path)

    #anzahl
    emoji = len(spacy(tweet)._.emoji)
    for index, word in enumerate(tokenize_tweet): # make a tuple (index ,word) to iterate
        if word in emoticon_dict.keys():
            if emoticon_dict[word] == "Positive": # if value of emoticon_dict[word] (key) is pos
                pos_emo = pos_emo + 1
            elif emoticon_dict[word] == "Negative":
                neg_emo = neg_emo + 1
    for index, word in enumerate(tweet.split()):
        if word in smiley_dict.keys():
            if smiley_dict[word] == "Positive":
                pos_smiley = pos_smiley + 1
            elif smiley_dict[word] == "Negative":
                neg_smiley = neg_smiley + 1

    #check last token
    tweet = re.sub(r"(\w)(\1{3,})", r"\1", tweet) # in case of :=)))) > 3
    if spacy(tweet)[-1]._.is_emoji or tweet.split()[-1] in smiley_dict.keys():
        last_emo_smiley = 1
    else:
        last_emo_smiley = 0

    return emoji, pos_emo, neg_emo, pos_smiley, neg_smiley, last_emo_smiley

# zählt postag (Verb, Noun, Pron, Adj, intj, kokom), negation, verstärker, shifter pos, shifter neg, shifter gen,
# neutral wörter uni, pos uni, pos bi, pos tri, neg uni, neg bi, neg tri
# Featureset: 19 Features
def get_lexicon_features(tweet):
    negation_lex = pd.read_csv(negation_lex_path, header=None)
    verstarker_lex = pd.read_csv(verstarker_lex_path, header=None)
    shifter_lex_pos = pd.read_csv(shifter_lex_pos_path, header=None)
    shifter_lex_neg = pd.read_csv(shifter_lex_neg_path, header=None)
    shifter_lex_gen = pd.read_csv(shifter_lex_gen_path, header=None)

    neutral_lex = pd.read_csv(neutral_lex_path, delimiter="\t", header=None)
    pos_uni_lex = pd.read_csv(pos_uni_lex_path)
    pos_bi_lex = pd.read_csv(pos_bi_lex_path)
    pos_tri_lex = pd.read_csv(pos_tri_lex_path)
    neg_uni_lex = pd.read_csv(neg_uni_lex_path)
    neg_bi_lex = pd.read_csv(neg_bi_lex_path)
    neg_tri_lex = pd.read_csv(neg_tri_lex_path)

    verb, nomen, pr, adj, intj, cconj, negs, emps, shift_pos, shift_neg, \
    shift_gen, neutral_uni, pos_uni, pos_bi, pos_tri, \
    neg_uni, neg_bi, neg_tri, sentiscore = 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0

    tweet = re.sub(r"(\w)(\1{3,})", r"\1", tweet) # handle repeated char > 3

    # For unigram, compare each token with given condition
    for token in spacy(tweet):
        if token.pos_ == "VERB":
            verb = verb + 1
        elif token.pos_ == "NOUN":
            nomen = nomen + 1
        elif token.pos_ == "PRON":
            pr = pr + 1
        elif token.pos_ == "ADJ":
            adj = adj + 1
        elif token.pos_ == "INTJ":
            intj = intj + 1
        elif token.tag_ == "KOKOM":
            cconj = cconj + 1

        for negationen in negation_lex[0]: # element in negation dict ?
            if token.text.lower() == negationen.lower():
                negs = negs + 1

        for verstarker in verstarker_lex[0]:
            if token.text.lower() == verstarker.lower():
                emps = emps + 1

        for pos in shifter_lex_pos[0]:
            if token.text.lower() == pos.lower():
                shift_pos = shift_pos + 1

        for neg in shifter_lex_neg[0]:
            if token.text.lower() == neg.lower():
                shift_neg = shift_neg + 1

        for gen in shifter_lex_gen[0]:
            if token.text.lower() == gen.lower():
                shift_gen = shift_gen + 1

        for neu_uni in neutral_lex[0]:
            if token.text.lower() == neu_uni.lower():
                neutral_uni = neutral_uni + 1

        for posi_uni in pos_uni_lex["Wort"]:
            if token.text.lower() == posi_uni.lower():
                print(f"pos: {token.text}")
                pos_uni = pos_uni + 1

        for nega_uni in neg_uni_lex["Wort"]:
            if token.text.lower() == nega_uni.lower():
                print(f"neg: {token.text}")
                neg_uni = neg_uni + 1

        if token._.sentiws is not None:
            sentiscore = sentiscore + token._.sentiws
        else:
            sentiscore = 0

    # for bigram, compare each token with given condition
    for token in make_gram(tweet, 2):
        for bi_pos in pos_bi_lex["Wort"]:
            if token.lower() == bi_pos.lower():
                pos_bi = pos_bi + 1
                print("po bi:" + token)
        for bi_neg in neg_bi_lex["Wort"]:
            if token.lower() == bi_neg.lower():
                neg_bi = neg_bi + 1
                print("neg bi:" + token)

    # for trigram, compare each token with given condition
    for token in make_gram(tweet, 3):
        for tri_pos in pos_tri_lex["Wort"]:
            if token.lower() == tri_pos.lower():
                pos_tri = pos_tri + 1
                print("po tri:" + token)
        for tri_neg in neg_tri_lex["Wort"]:
            if token.lower() == tri_neg.lower():
                neg_tri = neg_tri + 1
                print("neg tri:" + token)

    return verb, nomen, pr, adj, intj, cconj, negs, emps, shift_pos, shift_neg, shift_gen,\
           neutral_uni, pos_uni, pos_bi, pos_tri, neg_uni, neg_bi, neg_tri, sentiscore

# remove punctuation but keep emoticons and smileys as words
# accuracy of smiley depend on tokenizer accuracy
def remove_punctuation(tweet):
    tokenize_tweet = tokenize(tweet)
    emoticon_dict = get_emoticon_dict(emoticon_path)
    smiley_dict = get_smiley_dict(smiley_path)
    #print("remove punc")
    for index, word in enumerate(tokenize_tweet): # make a tuple (index ,word) to iterate
        if word in emoticon_dict.keys():
            if emoticon_dict[word] == "Positive": # if value of emoticon_dict[word] (key) is pos
                tokenize_tweet[index] = "positiveemoticon"
            elif emoticon_dict[word] == "Negative":
                tokenize_tweet[index] = "negativeemoticon"
        elif word in smiley_dict.keys():
            if smiley_dict[word] == "Positive":
                tokenize_tweet[index] = "positivesmiley"
            elif smiley_dict[word] == "Negative":
                tokenize_tweet[index] = "negativesmiley"
    return " ".join(tokenize_tweet).translate(str.maketrans("", "", string.punctuation))

def tweet_preprocessing(tweet):
    # all to lower case
    tweet = tweet.lower()

    #remove URLs
    tweet = re.sub(r"(http://\S+|https://\S+)", "", tweet)
    tweet = re.sub(r"www\.\S+\.com", '', tweet)

    #remove user @ reference and '#' from tweet
    tweet = re.sub(r"\@\w+|\#\w+", "", tweet)

    #remove rt and cc
    tweet = re.sub(r"rt|cc", "", tweet)

    #remove html tags
    tweet = re.sub(r"<.*?>", "", tweet)

    #handle repeated characters > 3
    tweet = re.sub(r"(\w)(\1{3,})", r"\1", tweet)

    # remove puctuation
    tweet = remove_punctuation(tweet)

    # Remove all the special characters eg. newline
    tweet = re.sub(r'\W', " ", tweet)

    # transform multi spaces into singl space
    tweet = re.sub(r'\s+', ' ', tweet, flags=re.I)

    # lemmatize
    tweet = lemmatize(tweet.strip())

    # remove numbers
    tweet = re.sub(r"[0-9]", "", tweet)

    #transform multi spaces into singl space
    tweet = re.sub(r'\s+', ' ', tweet, flags=re.I)

    tweet = " ".join(char for char in tweet.split() if len(char) > 1)

    print(tweet)

    return "".join(tweet)

#load all into one corpus
def write_big_corpus_to_file(file_path):
    df = all_dataset()
    print("Printing file...")
    df["tweet_processed"] = df["tweet"].apply(tweet_preprocessing)
    df.to_csv(file_path, encoding='utf-8', index=False)
    print("Done !")

#drop dup and na due to lemmatization
def drop_dup_in_big_file(path):
    df = pd.read_csv(path, sep=sep_csv)
    if path == train_corpus_path:
        df.drop_duplicates(subset="tweet_processed", inplace=True)
        df.dropna(subset=["tweet_processed"], inplace=True)
        df.to_csv(path, sep=sep_csv, index=False)
    elif path == pred_meta_corpus_path:
        df.dropna(subset=["tweet_processed"], inplace=True)
        df.to_csv(path, sep=sep_csv, index=False)

def write_feature_for_big_file(path_in, path_out):
    with open(path_in, encoding="utf-8", newline="", mode="r") as in_file, \
            open(path_out, encoding="utf-8", newline="", mode="w") as out_file:
        content = csv.reader(in_file, delimiter=",")
        next(content) # ignore first line
        ce = csv.writer(out_file, delimiter=",")
        ce.writerow(df_final_cols) # write header

        for rows in content:
            print(rows)
            #13
            word, cap, punc, qm, em, seq_em_qm, seq_qm_em, r_letter, r_punc, last_token_punc, \
            last_token_em, last_token_qm, slang = get_microblog_surface_feature(rows[0])

            # 6
            emoji, pos_emo, neg_emo, pos_smiley, neg_smiley, last_emo_smiley = get_emoticon_features(rows[0])

            #19
            verb, nomen, pr, adj, intj, cconj, negs, emps, shift_pos, shift_neg, shift_gen, \
            neutral_uni, pos_uni, pos_bi, pos_tri, neg_uni, neg_bi, neg_tri, sentiscore = get_lexicon_features(rows[2])

            ce.writerow([rows[0], rows[1], rows[2], word, cap, punc, qm, em, seq_em_qm, seq_qm_em, r_letter, r_punc
                         , last_token_punc, last_token_em, last_token_qm, slang, emoji, pos_emo, neg_emo, pos_smiley,
                         neg_smiley, last_emo_smiley, verb, nomen, pr, adj, intj, cconj, negs,
                         emps, shift_pos, shift_neg, shift_gen, neutral_uni, pos_uni, pos_bi, pos_tri, neg_uni, neg_bi,
                         neg_tri, sentiscore])

def write_pred_corpus(path_in, path_out):
    df = pd.read_csv(path_in, sep=sep_csv, header=0)
    print("Printing file...")
    df["tweet_processed"] = df["tweet"].apply(tweet_preprocessing)
    df.to_csv(path_out, encoding='utf-8', index=False)
    print("Done !")

def write_feature_to_pred(path_in, path_out):
    with open(path_in, encoding="utf-8", newline="", mode="r") as in_file, \
            open(path_out, encoding="utf-8", newline="", mode="w") as out_file:
        content = csv.reader(in_file, delimiter=",")
        next(content) # ignore first line
        ce = csv.writer(out_file, delimiter=",")
        ce.writerow(df_pred_col) # write header

        for rows in content:
            print(rows)
            #13
            word, cap, punc, qm, em, seq_em_qm, seq_qm_em, r_letter, r_punc, last_token_punc, \
            last_token_em, last_token_qm, slang = get_microblog_surface_feature(rows[3])

            # 6
            emoji, pos_emo, neg_emo, pos_smiley, neg_smiley, last_emo_smiley = get_emoticon_features(rows[3])

            #19
            verb, nomen, pr, adj, intj, cconj, negs, emps, shift_pos, shift_neg, shift_gen, \
            neutral_uni, pos_uni, pos_bi, pos_tri, neg_uni, neg_bi, neg_tri, sentiscore = get_lexicon_features(rows[9])

            ce.writerow([rows[1], rows[6], rows[3], rows[9], word, cap, punc, qm, em, seq_em_qm, seq_qm_em, r_letter, r_punc
                         , last_token_punc, last_token_em, last_token_qm, slang, emoji, pos_emo, neg_emo, pos_smiley,
                         neg_smiley, last_emo_smiley, verb, nomen, pr, adj, intj, cconj, negs,
                         emps, shift_pos, shift_neg, shift_gen, neutral_uni, pos_uni, pos_bi, pos_tri, neg_uni, neg_bi,
                         neg_tri, sentiscore])

if __name__ == "__main__":

    # 1. Preprocessing training corpus
    print("Preprocessing Train Corpus....")
    write_big_corpus_to_file(train_corpus_path)
    drop_dup_in_big_file(train_corpus_path)
    write_feature_for_big_file(train_corpus_path, train_meta_corpus_path)
    print("Done")

    # 2. Write pred corpus
    print("Preprocessing Pred Corpus...")
    write_pred_corpus("resources/raw_corpus/tweets.csv", pred_corpus_path)
    write_feature_to_pred(pred_corpus_path, pred_meta_corpus_path)
    drop_dup_in_big_file(pred_meta_corpus_path)
    print("Done")
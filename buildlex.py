import requests
import json
import spacy
import pandas as pd
import re
import string
import time
from HanTa import HanoverTagger as ht

# In der Main wird das Skript gesteuert

# init var
nlp = spacy.load('de_core_news_lg')
neutral_clues = pd.read_csv("resources/lexicon/GermanPolarityClues-Neutral-Lemma-21042012.tsv", delimiter="\t", header=None)
positive_sws = pd.read_csv("resources/raw_lexicons/SentiWS_v2.0_Positive.txt", delim_whitespace=True, header=None)
negative_sws = pd.read_csv("resources/raw_lexicons/SentiWS_v2.0_Negative.txt", delim_whitespace=True, header=None)
positive_clues = pd.read_csv("resources/raw_lexicons/GermanPolarityClues-Positive-Lemma-21042012.tsv", delimiter="\t", header=None)
negative_clues = pd.read_csv("resources/raw_lexicons/GermanPolarityClues-Negative-Lemma-21042012.tsv", delimiter="\t", header=None)


# tokenizer with spacy
def tokenize(token):
    doc = nlp(token)
    word = [words.text for words in doc]
    return word

# lemmatizer with HanTa -> to get lemmatize lex
def lemmatize(token):
    # HANTA: load model from github, anaylyze sentence with tag_sent, return the lemmas from the sentence
    tagger = ht.HanoverTagger('morphmodel_ger.pgz')
    tags = tagger.tag_sent(tokenize(token), taglevel=1)
    lemmatized_sentence = ["".join(lemma[1]) for lemma in tags]
    return " ".join(lemmatized_sentence).translate(str.maketrans("", "", string.punctuation)).strip()

# get the raw shiter lexicon and seperate the file in shifter pos, shifter neg, shifter general
def get_shifter_lexicon(path_in, path_g, path_n, path_p):
    with open(path_in, encoding="utf-8", mode="r") as in_file, \
            open(path_g, encoding="utf-8", mode="w", newline="") as out_file_g,\
            open(path_n, encoding="utf-8", mode="w", newline="") as out_file_n,\
            open(path_p, encoding="utf-8", mode="w", newline="") as out_file_p:
        content = in_file.readlines()
        for rows in content:
            string = [row for row in rows.split()]
            if string[1] == "g":
                out_file_g.write(f"{string[0]}\n")
            elif string[1] == "n":
                out_file_n.writelines(f"{string[0]}\n")
            elif string[1] == "p":
                out_file_p.write(f"{string[0]}\n")

# function to explore the json format of openthesaurus
def jprint(obj):
    #create a formatted string of the Python JSON object
    text = json.dumps(obj, sort_keys=True, indent= 4)
    print(text)

#preprocessing the raw sentiws lexicon
def sentiWS(df, sentiment):
    list = []
    for lines in df.iloc[:, 0]:
        string = [row for row in lines.split()]
        line = re.sub(r"\|.*$", "", string[0]) # schneide alles ab | weg
        list.append(line)
    df = pd.DataFrame(list)
    df.to_csv(f"resources/raw_lexicons/SentiWS_{sentiment}.txt", header=["Term"], index=False)

def get_dict(df, sentiment):
    synset_uni = []
    synset_bi = []
    synset_tri = []

    for word in df.values:
        response = requests.get(f"https://www.openthesaurus.de/synonyme/search?q={word}&format=application/json")
        #jprint(response)
        try:
            synsets = response.json()["synsets"]
            for word in synsets:
                terms = word["terms"]
                if terms is not None:
                    for term in terms:
                        term = re.sub(r"(\(([^\)]+)\))", "", term["term"]).translate(
                            str.maketrans("", "", string.punctuation)
                        ).strip() # remove everything in parenthesis
                        if len(term.split()) == 1:
                            synset_uni.append(lemmatize(term))
                        elif len(term.split()) == 2:
                            synset_bi.append(lemmatize(term))
                        elif len(term.split()) == 3:
                            synset_tri.append(lemmatize(term))

            print(synset_uni)
            print(synset_bi)
            print(synset_tri)
        except json.decoder.JSONDecodeError as e:
            print(e)
            time.sleep(60)
            continue

    print("Too csv... \n")
    df1 = pd.DataFrame(synset_uni)
    df2 = pd.DataFrame(synset_bi)
    df3 = pd.DataFrame(synset_tri)
    df1.to_csv(f"resources/raw_lexicons/GermanPolarityClues-SentiWS-{sentiment}-Synset-unigram.tsv", index=False)
    df2.to_csv(f"resources/raw_lexicons/GermanPolarityClues-SentiWS{sentiment}-Synset-bigram.tsv", index=False)
    df3.to_csv(f"resources/raw_lexicons/GermanPolarityClues-SentiWS{sentiment}-Synset-trigram.tsv", index=False)

# concat SentiWS and German polarity Clues into one Dataframe of negative terms
def union_dict_negativ():
    df_negativ = pd.read_csv("resources/raw_lexicons/SentiWS_negative.txt")
    all_negative = pd.concat([df_negativ["Term"], negative_clues[0]], ignore_index=True)
    all_negative.drop_duplicates(inplace=True)
    return all_negative

# concat SentiWS and German polarity Clues into one Dataframe of positive terms
def union_dict_positiv():
    df_positiv = pd.read_csv("resources/raw_lexicons/SentiWS_positive.txt")
    all_positive = pd.concat([df_positiv["Term"], positive_clues[0]], ignore_index=True)
    all_positive.drop_duplicates(inplace=True)
    return all_positive

#check for duplicates in all GPC and SentiWS Synset Lexicon and write a new lexicon
def check_all_dict():

    pos_uni = pd.read_csv("resources/raw_lexicons/GermanPolarityClues-SentiWS-positiv-Synset-unigram.tsv", delimiter="\t", header=0)
    pos_uni.drop_duplicates(subset="0", inplace=True)

    neg_uni = pd.read_csv("resources/raw_lexicons/GermanPolarityClues-SentiWS-negativ-Synset-unigram.tsv", delimiter="\t", header=0)
    neg_uni.drop_duplicates(subset="0", inplace=True)

    pos_bi = pd.read_csv("resources/raw_lexicons/GermanPolarityClues-SentiWSpositiv-Synset-bigram.tsv", delimiter="\t", header=0)
    pos_bi.drop_duplicates(subset="0", inplace=True)

    neg_bi = pd.read_csv("resources/raw_lexicons/GermanPolarityClues-SentiWSnegativ-Synset-bigram.tsv", delimiter="\t", header=0)
    neg_bi.drop_duplicates(subset="0", inplace=True)

    pos_tri = pd.read_csv("resources/raw_lexicons/GermanPolarityClues-SentiWSpositiv-Synset-trigram.tsv", delimiter="\t", header=0)
    pos_tri.drop_duplicates(subset="0", inplace=True)

    neg_tri = pd.read_csv("resources/raw_lexicons/GermanPolarityClues-SentiWSnegativ-Synset-trigram.tsv", delimiter="\t", header=0)
    neg_tri.drop_duplicates(subset="0", inplace=True)

    # type :numpy arrays
    pos_list_uni = pos_uni.values.ravel()
    neg_list_uni = neg_uni.values.ravel()
    neutral_list_uni = neutral_clues[0].values.ravel()
    pos_list_bi = pos_bi.values.ravel()
    neg_list_bi = neg_bi.values.ravel()
    pos_list_tri = pos_tri.values.ravel()
    neg_list_tri = neg_tri.values.ravel()

    # lösche duplicate zwischen neutral und positiv in positiv
    while list(set(pos_list_uni).intersection(set(neutral_list_uni))) is not None:
        print("Lösche duplicate zwischen neutral und positiv")
        print(f"Länge neutral: {len(neutral_list_uni)} und pos uni: {len(pos_list_uni)}.")
        duplicate = list(set(pos_list_uni).intersection(set(neutral_list_uni))) # type List, finde die duplikate in beiden sets
        print(f"Lösche {len(duplicate)} Duplicate")

        pos_list_uni = [word for word in pos_list_uni if word not in duplicate]

        if list(set(pos_list_uni).intersection(set(neutral_list_uni))):
            print("Immernoch Duplicate.")
        else:
            print(f"Keine Duplicate mehr zwischen neutral({len(neutral_list_uni)}) und pos uni({len(pos_list_uni)}).")
            break

    # lösche zwischen neutral und negative in negativ
    while list(set(neg_list_uni).intersection(set(neutral_list_uni))) is not None:
        print("Lösche zwischen neutral und negativ")
        print(f"Länge neutral: {len(neutral_list_uni)} und neg uni: {len(neg_list_uni)}.")
        duplicate = list(set(neg_list_uni).intersection(set(neutral_list_uni)))  # type List
        print(f"Lösche {len(duplicate)} Duplicate")

        neg_list_uni = [word for word in neg_list_uni if word not in duplicate]

        if list(set(neg_list_uni).intersection(set(neutral_list_uni))):
            print("Immernoch Duplicate.")
        else:
            print(f"Keine Duplicate mehr zwischen neutral({len(neutral_list_uni)}) und pos({len(neg_list_uni)}).")
            break

    # lösche zwischen pos und neg uni
    while list(set(pos_list_uni).intersection(set(neg_list_uni))) is not None:
        print("Duplicate zwischen neg uni und pos uni.")
        print(f"Länge neg:{len(neg_list_uni)} und pos uni: {len(pos_list_uni)}.")
        duplicate = list(set(neg_list_uni).intersection(set(pos_list_uni)))  # type List
        print(f"Lösche {len(duplicate)} Duplicate")

        neg_list_uni = [word for word in neg_list_uni if word not in duplicate]
        pos_list_uni = [word for word in pos_list_uni if word not in duplicate]

        if list(set(neg_list_uni).intersection(set(pos_list_uni))):
            print("Immernoch Duplicate.")
        else:
            print(f"Keine Duplicate mehr zwischen neutral({len(pos_list_uni)}) und pos uni({len(neg_list_uni)}).")
            break

    # lösche zwischen pos und neg bi
    while list(set(pos_list_bi).intersection(set(neg_list_bi))) is not None:
        print("Duplicate zwischen neg und pos bi.")
        print(f"Länge pos bi: {len(pos_list_bi)} und neg bi: {len(neg_list_bi)}.")
        duplicate = list(set(pos_list_bi).intersection(set(neg_list_bi)))  # type List
        print(f"Lösche {len(duplicate)} Duplicate")

        neg_list_bi = [" ".join(word.split()) for word in neg_list_bi if word not in duplicate] # remove whitespace !!!
        pos_list_bi = [" ".join(word.split()) for word in pos_list_bi if word not in duplicate]

        if list(set(neg_list_bi).intersection(set(pos_list_bi))):
            print("Immernoch Duplicate. Lösche ...")
            df = pd.DataFrame(pos_list_bi, columns=["Wort"])
            df.drop_duplicates(subset="Wort", inplace=True)
            df.to_csv("resources/lexicon/pos-bigram.txt", index=False)
            df2 = pd.DataFrame(neg_list_bi, columns=["Wort"])
            df2.drop_duplicates(subset="Wort", inplace=True)
            df2.to_csv("resources/lexicon/neg-bigram.txt", index=False)
        else:
            print(f"Keine Duplicate mehr zwischen pos bi({len(pos_list_bi)}) und neg bi({len(neg_list_bi)}).")
            break

    # lösche zwischen pos und neg tri
    while list(set(pos_list_tri).intersection(set(neg_list_tri))) is not None:
        print("Duplicate zwischen neg und pos tri.")
        print(f"Länge pos tri: {len(pos_list_tri)} und neg tri: {len(neg_list_tri)}.")
        duplicate = list(set(pos_list_tri).intersection(set(neg_list_tri)))  # type List
        print(f"Lösche {len(duplicate)} Duplicate")

        neg_list_tri = [" ".join(word.split()) for word in neg_list_tri if word not in duplicate] # remove whitespace
        pos_list_tri = [" ".join(word.split()) for word in pos_list_tri if word not in duplicate]

        if list(set(neg_list_tri).intersection(set(pos_list_tri))):
            print("Immernoch Duplicate. Lösche ...")
            df = pd.DataFrame(pos_list_tri, columns=["Wort"])
            df.drop_duplicates(subset="Wort", inplace=True)
            df.to_csv("resources/lexicon/pos-trigram.txt", index=False)
            df2 = pd.DataFrame(neg_list_tri, columns=["Wort"])
            df2.drop_duplicates(subset="Wort", inplace=True)
            df2.to_csv("resources/lexicon/neg-trigram.txt", index=False)
        else:
            print(f"Keine Duplicate mehr zwischen pos tri({len(pos_list_tri)}) und neg tri({len(neg_list_tri)}).")
            break

        df = pd.DataFrame(pos_list_uni, columns=["Wort"])
        df.drop_duplicates(subset="Wort", inplace=True)
        df.to_csv("resources/lexicon/pos-unigram.txt", index=False)
        df2 = pd.DataFrame(neg_list_uni, columns=["Wort"])
        df2.drop_duplicates(subset="Wort", inplace=True)
        df2.to_csv("resources/lexicon/neg-unigram.txt", index=False)


if __name__ == "__main__":
    #1. Get sentiment Shifter
    print("Building Shifter Lexicon ...")
    get_shifter_lexicon("resources/raw_lexicons/sentimentshifter.txt", "resources/lexicon/shifter_general.txt",
                        "resources/lexicon/shifter_negativ.txt", "resources/lexicon/shifter_positiv.txt")
    print("Done!")

    #2. Preprocess SentiWS to be able to concat with German Polarity Clues
    print("Preprocess SentiWS Lexicon ...")
    sentiWS(negative_sws, "negative")
    sentiWS(positive_sws, "positive")
    print("Done!")

    #3. With the concat SentiWS and GPL, build a lexicon with synsets with Openthesaurus for Feature Extraction
    print("Getting Synsets from Openthesaurus ...")
    get_dict(union_dict_positiv(), "positiv")
    get_dict(union_dict_negativ(), "negativ")
    print("Printing .. Done!")

    #4. Check for Duplicates and make the final dictionary
    print("Check for duplicates.")
    check_all_dict()
    print("Done.")



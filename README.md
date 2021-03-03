**Opinion Mining und Topic Modelling von Tweets:**
*Eine Stimmungsanalyse über die aktuelle Situation der Covid-19 Pandemie in Deutschland mit Hilfe von Machine Learning auf Twitter*

Entwickelt mit Python 3.8 (siehe requirements.txt für Abhängigkeiten)

Autor: Hung Anh Le

---

---

## Quellen

https://machinelearningmastery.com/cost-sensitive-svm-for-imbalanced-classification/

https://stackabuse.com/python-for-nlp-sentiment-analysis-with-scikit-learn/#disqus_thread

https://www.pluralsight.com/guides/building-a-twitter-sentiment-analysis-in-python

https://scikit-learn.org/stable/auto_examples/model_selection/grid_search_text_feature_extraction.html#sphx-glr-auto-examples-model-selection-grid-search-text-feature-extraction-py

https://scikit-learn.org/stable/auto_examples/model_selection/plot_grid_search_digits.html#sphx-glr-auto-examples-model-selection-plot-grid-search-digits-py

https://machinelearningmastery.com/machine-learning-in-python-step-by-step/

https://towardsdatascience.com/3-things-you-need-to-know-before-you-train-test-split-869dfabb7e50

https://machinelearningmastery.com/save-load-machine-learning-models-python-scikit-learn/

https://www.machinelearningplus.com/nlp/topic-modeling-gensim-python/

https://www.machinelearningplus.com/nlp/topic-modeling-visualization-how-to-present-results-lda-models/

https://medium.com/@osas.usen/topic-extraction-from-tweets-using-lda-a997e4eb0985


Trainingskorpus:
van den Berg, E., Korfhage, K., Ruppenhofer, J., Wiegand, M., and Markert, K. (2020). Doctor who? Framing through names and titles in German. In Proceedings of the 12th Conference on Language Resources and Evaluation, May 11-16, 2020, Marseille, France. [abgerufen 10.10.2020]

Cieliebak, Mark & Deriu, Jan & Egger, Dominic & Uzdilli, Fatih. (2017). A Twitter Corpus and Benchmark Resources for German Sentiment Analysis. Social NLP @ EACL. 10.18653/v1/W17-1106. https://spinningbytes.com/resources/germansentiment/ [abgerufen 10.10.2020]

Mozetič, Igor; Grčar, Miha and Smailović, Jasmina, 2016, Twitter sentiment for 15 European languages, Slovenian language resource repository CLARIN.SI, http://hdl.handle.net/11356/1054. [abgerufen 10.10.2020]

Narr, Sascha, Hülfenhaus, Michael & Sahin, Albayrak 2012. Language-Independent Twitter Sentiment Analysis. DAI-Labor, Technical University Berlin . http://www.dai-labor.de/fileadmin/files/publications/narr-twittersentiment-KDML-LWA-2012.pdf [abgerufen 10.10.2020]

Wojatzki, Michael u. a. 2017. Proceedings of the GermEval 2017-Shared Task on As-pect-based Sentiment in Social Media Customer Feedback. https://sites.google.com/view/germeval2017-absa/data?authuser=0 [Stand 2021-02-19].


Lexika:
Mario Sänger, Ulf Leser, Steffen Kemmerer, Peter Adolphs, and Roman Klinger. SCARE – The Sentiment Corpus of App Reviews with Fine-grained Annotations in German. In Proceedings of the Tenth International Conference on Language Resources and Evaluation (LREC’16), Portorož, Slovenia, May 2016. European Language Resources Association (ELRA) https://www.romanklinger.de/scare/ [abgerufen 19.01.2021].

R. Remus, U. Quasthoff & G. Heyer: SentiWS - a Publicly Available German-language Resource for Sentiment Analysis. In: Proceedings of the 7th International Language Resources and Evaluation (LREC'10), pp. 1168-1171, 2010. https://wortschatz.uni-leipzig.de/en/download. [abgerufen 19.01.2021]

Ulli Waltinger (2010). Sentiment Analysis Reloaded: A Comparative Study On Sentiment Polarity Identification Combining Machine Learning And Subjectivity Features. In Proceedings of the 6th International Conference on Web Information Systems and Technologies (WEBIST '10), April 7-10, 2010, Valencia, 2010. http://www.ulliwaltinger.de/sentiment/. [abgerufen 19.01.2021]

Michael Wiegand, Maximilian Wolf and Josef Ruppenhofer "Negation Modeling for German Polarity Classification", in Proceedings of the German Society for Computational Linguistics and Language Technology (GSCL), Potsdam, Germany. 2017. https://github.com/artificial-max/polcla/tree/master/polcla/src/main/resources/dictionaries [abgerufen 19.01.2021]

Michael Wiegand et al. "Saarland University’s participation in the German sentiment analysis shared task (GESTALT)." , Workshop Proceedings of the 12th KONVENS. 2014. https://github.com/artificial-max/polcla/tree/master/polcla/src/main/resources/dictionaries [abgerufen 19.01.2021]

---

## Hotfix
Merge span bug from spacymoji, change init.py of spacymoji according to: https://github.com/ines/spacymoji/pull/8/files 

import twitter

# initialize api instance
twitter_api = twitter.Api(consumer_key='',
                          consumer_secret='',
                          access_token_key='',
                          access_token_secret='')

# test authentication



# ------------------------------------------------------------------------

def buildTestSet(search_keyword):
    try:
        tweets_fetched = twitter_api.GetSearch(search_keyword, count=100, lang="id")

        print("Fetched " + str(len(tweets_fetched)) + " tweets for the term " + search_keyword)

        return [{"text": status.text, "label": None} for status in tweets_fetched]
    except:
        print("Unfortunately, something went wrong..")
        return None


# ------------------------------------------------------------------------

search_term = input("Enter a search keyword: ")
testDataSet = buildTestSet(search_term)

print(testDataSet[0:4])


# ------------------------------------------------------------------------

def buildTrainingSet(corpusFile, tweetDataFile):
    import csv
    import time

    corpus = []

    with open(corpusFile, 'r+', encoding="utf-8") as csvfile:
        lineReader = csv.reader(csvfile, delimiter=',', quotechar="\"")
        for row in lineReader:
            corpus.append({"tweet_id": row[0], "label": row[3], "topic": row[4]})


    trainingDataSet = []

    for tweet in corpus:
        try:
            status = twitter_api.GetStatus(tweet["tweet_id"])
            tweet["text"] = status.text
            trainingDataSet.append(tweet)
        except:
            continue
    # Now we write them to the empty CSV file
    with open(tweetDataFile, 'w', encoding="utf-8") as csvfile:
        linewriter = csv.writer(csvfile, delimiter=',', quotechar="\"")
        for tweet in trainingDataSet:
            try:
                linewriter.writerow([tweet["tweet_id"], tweet["text"], tweet["label"], tweet["topic"]])
            except Exception as e:
                print(e)
    return trainingDataSet


# ------------------------------------------------------------------------

corpusFile = "corpus.csv"
tweetDataFile = "tweetDataFile.csv"

trainingData = buildTrainingSet(corpusFile, tweetDataFile)

# ------------------------------------------------------------------------


import re
import string
from nltk.tokenize import word_tokenize
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory, StopWordRemover, ArrayDictionary
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory

class PreProcessTweets:
    factory = StopWordRemoverFactory()
    get_stop_words = factory.get_stop_words()
    factory1 = StemmerFactory()
    stemmer = factory1.create_stemmer()

    def __init__(self):
        self._stopwords = StopWordRemoverFactory.get_stop_words(self)
    #  self._stopwords = set(stopwords.words('indonesian') + list(punctuation) + ['AT_USER', 'URL'])
    def processTweets(self, list_of_tweets):
        processedTweets = []
        for tweet in list_of_tweets:
            processedTweets.append((self._processTweet(tweet["text"]), tweet["label"]))
        return processedTweets


    def _processTweet(self, tweet):
        punctuations = '''!()-![]{};:+'"\,<>./?@#$%^&*_~'''
        tweet = tweet.lower()  # convert text to lower-case
        tweet = re.sub('((www\.[^\s]+)|(https?://[^\s]+))', '', tweet)  # remove URLs
        tweet = re.sub('@[^\s]+', '', tweet)  # remove usernames
        tweet = re.sub(r'#([^\s]+)', r'\1', tweet)  # remove the # in #hashtag
        tweet = "".join((char for char in tweet if char not in string.punctuation))
        tweet = re.sub('\s+', ' ', tweet).strip()
        tweet = re.sub(r"\d", "", tweet)
        # Ambil Stopword bawaan
        stop_factory = StopWordRemoverFactory().get_stop_words()
        more_stopword = open("stopword.txt", "r").read().split()
        # Merge stopword
        data = stop_factory + more_stopword
        dictionary = ArrayDictionary(data)
        str = StopWordRemover(dictionary)

        factory1 = StemmerFactory() #stemming factory
        stemmer = factory1.create_stemmer() #buat stemming
        #
        tweet = str.remove(tweet)
        # tweet = stemmer.stem(tweet)  # stemming tweet
        tweet = word_tokenize(tweet)  # remove repeated characters (helloooooooo into hello)
        # return [word for word in tweet if word not in self._stopwords]
        return tweet


tweetProcessor = PreProcessTweets()
preprocessedTrainingSet = tweetProcessor.processTweets(trainingData)
preprocessedTestSet = tweetProcessor.processTweets(testDataSet)



# ------------------------------------------------------------------------
# list_kasar = []
# with open("kasar.txt","r",encoding="utf-8") as file:
#     data_file = file.readline()
#     for data in data_file:
#         list_kasar.append(data)
# file.close()
#
# for tweet in preprocessedTestSet:
#     for kata_kasar in list_kasar:
#
#
# #------------------------------------------------
import nltk



def buildVocabulary(preprocessedTrainingData):
    all_words = []

    for (words, sentiment) in preprocessedTrainingData:
        all_words.extend(words)

    wordlist = nltk.FreqDist(all_words)
    word_features = wordlist.keys()
    print(" ")
    print("############################")
    print(word_features)
    print(len(word_features))
    print(wordlist)
    print(len(wordlist))
    print("############################")
    print(" ")
    return word_features


# ------------------------------------------------------------------------

def extract_features(tweet):
    tweet_words = set(tweet)
    features = {}


    for word in word_features:
        features['contains(%s)' % word] = (word in tweet_words)

    return features

# ------------------------------------------------------------------------


# Now we can extract the features and train the classifier
word_features = buildVocabulary(preprocessedTrainingSet)

trainingFeatures = nltk.classify.apply_features(extract_features, preprocessedTrainingSet)
# ------------------------------------------------------------------------

NBayesClassifier = nltk.NaiveBayesClassifier.train(trainingFeatures)
# ------------------------------------------------------------------------
NBResultLabels = [NBayesClassifier.classify(extract_features(tweet[0])) for tweet in preprocessedTestSet]
# ------------------------------------------------------------------------

# get the majority vote
if NBResultLabels.count('positive') > NBResultLabels.count('negative'):
    for data, labels in zip(testDataSet, NBResultLabels):
        print(data, labels)
    # print(" ")
    # print("Sebelum Stopword Removal :")
    # print(testDataSet[-1], NBResultLabels[-1])
    # print("Hasil Setelah Stopword Removal :")
    # print(preprocessedTestSet[-1])
    # print(" ")
    # print("Sebelum Stopword Removal :")
    # print(testDataSet[-2], NBResultLabels[-2])
    # print("Hasil Setelah Stopword Removal :")
    # print(preprocessedTestSet[-2])
    # print(" ")
    # print("Sebelum Stopword Removal :")
    # print(testDataSet[-3], NBResultLabels[-3])
    # print("Hasil Setelah Stopword Removal :")
    # print(preprocessedTestSet[-3])
    # print(" ")


    print("Hasil Training Dengan Testing Di analisis dengan pencarian " + (search_term))
    print("Overall Positive Sentiment")
    print("Positive Sentiment Percentage = " + str(100 * NBResultLabels.count('positive') / len(NBResultLabels)) + "%")
    print("Negative Sentiment Percentage = " + str(100 * NBResultLabels.count('negative') / len(NBResultLabels)) + "%")

else:
    for data, labels in zip(testDataSet, NBResultLabels):
        print(data, labels)
    # print(" ")
    # print("Sebelum Stopword Removal :")
    # print(testDataSet[-1], NBResultLabels[-1])
    # print("Hasil Setelah Stopword Removal :")
    # print(preprocessedTestSet[-1])
    # print(" ")
    # print("Sebelum Stopword Removal :")
    # print(testDataSet[-2], NBResultLabels[-2])
    # print("Hasil Setelah Stopword Removal :")
    # print(preprocessedTestSet[-2])
    # print(" ")
    # print("Sebelum Stopword Removal :")
    # print(testDataSet[-3], NBResultLabels[-3])
    # print("Hasil Setelah Stopword Removal :")
    # print(preprocessedTestSet[-3])
    # print(" ")
    print("Hasil Training Dengan Testing Di analisis dengan pencarian " + (search_term))
    print("Overall Negative Sentiment")
    print("Positive Sentiment Percentage = " + str(100 * NBResultLabels.count('positive') / len(NBResultLabels)) + "%")
    print("Negative Sentiment Percentage = " + str(100 * NBResultLabels.count('negative') / len(NBResultLabels)) + "%")
# ------------------------------------------------------------------------
def buildTestSet(search_keyword):
    try:
        tweets_fetched = twitter_api.GetSearch(search_keyword, count=200)
        with open("out1.csv", "w+", encoding='utf-8') as f:
            f.write("date,user,is_retweet,is_quote,text,quoted_text\n")
        print("Fetched " + str(len(tweets_fetched)) + " tweets for the term " + search_keyword)
        print(preprocessedTestSet)

        return [{"text": status.text} for status in tweets_fetched]

    except:
        print("Unfortunately, something went wrong..")
        return None




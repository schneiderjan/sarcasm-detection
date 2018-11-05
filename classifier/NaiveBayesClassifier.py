from __future__ import division
import numpy as np
import pandas as pd
import time
import nltk
import sklearn as sk
from time import gmtime, strftime
from pprint import pprint
from sklearn import cross_validation
from sklearn.cross_validation import train_test_split
from sklearn.externals import joblib

# Properties
py_file = "NaiveBayesClassifier.py"
date_file = "data/tweets_90-10.csv"
start_time = time.time()
testd_size = 0.35


# Print Properties
print "Start Time  : ",time.strftime("%Y-%m-%d %H:%M:%S")
print "Python File : ",py_file
print "Data File   : ",date_file
print "Test Size   : ",testd_size


# Read Data Set
df = pd.read_csv(date_file, delimiter='|', encoding="utf-8", quotechar='|', header=None, names=['ID', 'Tweet','Status'])
#df = df.dropna()
print(len(df))

# Create Train and Test Set
train_df, test_df = train_test_split(df, test_size = testd_size)
print "\nTweets Train Set : ",len(train_df)
print "Tweets Test  Set : ",len(test_df)

# Train Set - Tweets
print "\n\nStart Training :", len(train_df), " Tweets (%s Seconds)" % (time.time() - start_time)
tweets_train = []
for row in train_df.itertuples():
    words_filtered = [e.lower() for e in row[2].split() if len(e) >= 3] 
    tweets_train.append((words_filtered, row[3]))

# Test Set - Tweets
tweets_test = []
for row in test_df.itertuples():
    words_filtered = [e.lower() for e in row[2].split() if len(e) >= 3] 
    tweets_test.append((words_filtered, row[3]))
    
# Get Unique Word from Word List
def get_word_features(wordlist):
    wordlist = nltk.FreqDist(wordlist)
    word_features = wordlist.keys()
    return word_features

# Get the word lists of Tweets
def get_words_in_tweets(tweets):
    all_words = []
    for (words, sentiment) in tweets:
      all_words.extend(words)
    return all_words

word_features = get_word_features(get_words_in_tweets(tweets_train))

# Feature Extractor
def extract_features(document):
    document_words = set(document)
    features = {}
    for word in word_features:
      features['contains(%s)' % word] = (word in document_words)
    return features

# Build Training and Test Set
training_set = nltk.classify.util.apply_features(extract_features, tweets_train)
test_set = nltk.classify.util.apply_features(extract_features, tweets_test)

# Create Classifier
classifier = nltk.NaiveBayesClassifier.train(training_set)

# NLTK Train Method
def train(labeled_featuresets, estimator=nltk.probability.ELEProbDist):
    
    # Create the P(label) distribution
    label_probdist = estimator(label_freqdist)
    
    # Create the P(fval|label, fname) distribution
    feature_probdist = {}
    return NaiveBayesClassifier(label_probdist, feature_probdist)

# Display
print "\n\nStart Testing 1 :", len(test_df), " Tweets (%s Seconds)" % (time.time() - start_time)
print "Wrong Predicted Tweets: "
count = 0
for row in test_df.itertuples():
    a = classifier.classify(extract_features(row[2].split()))
    if row[3] != a:
        #print(row[2])
        #print("Label:      ", row[3])
        #print("Prediction: ", a)
        #print("\n")
        count += 1
accuracy_test1 = (len(test_df)-count)/len(test_df)
print "Prediction : %d wrong from %d" %(count, len(test_df))
print "Accuracy   :", (accuracy_test1)

# Save Classifier
filename = 'classifier/classifier%s.pkl' % (str(time.strftime("%Y-%m-%d_%H:%M:%S")))
_ = joblib.dump(classifier, filename, compress=9)

# Print Output
print "\n\nStart Testing 2 :", len(test_df), " Tweets (%s Seconds)" % (time.time() - start_time)
accuracy_test2 = nltk.classify.util.accuracy(classifier, test_set)
print "Accuracy:", accuracy_test2
pprint(classifier.most_informative_features(25))


# Check Accuracy Tests
if accuracy_test1>accuracy_test2:
	print "\nTest 1 is more accurate"
elif accuracy_test1<accuracy_test2:
	print "\nTest 2 is more accurate"
else:
	print "\nResults are the same"


# End
print("\nTotal Time : %s Seconds" % (time.time() - start_time))
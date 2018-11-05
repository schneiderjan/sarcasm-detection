import itertools
from pprint import pprint
from sklearn.externals import joblib

tweet = "Heb me in mijn shirt gehesen voor een rondje rennen. Op t progstaan 38 minuten maar ik ga ns kijken hoever ik echt kan"

classifier = joblib.load("classifier/classifier2015-10-14_04:34:12.pkl")




# Feature Extractor
def extract_features(document):
	document_words = set(document)
	features = {}
	for x, y in classifier.most_informative_features():
		features[x] = (x[9:len(x)-1] in document_words)
	return features


#pprint(features)
pprint(classifier.most_informative_features(25))
print classifier.classify(extract_features(tweet.split()))
import numpy as np
import pandas as pd
import re

# tokenization
from nltk.tokenize import word_tokenize

# stem data
from nltk.stem.lancaster import LancasterStemmer
from nltk.stem import WordNetLemmatizer
from nltk.stem.porter import PorterStemmer
from nltk.stem.snowball import SnowballStemmer

lemmatizer = WordNetLemmatizer()
lancaster_stemmer = LancasterStemmer()

def _stemmer_word(word):
	if type(word) == list:
		return stemmer_words(word)
	# lem word as if a noun
	lem = lemmatizer.lemmatize(word,'n')
	# if unchanged
	if lem == word:
		#lem word as if a verb
		lem = lemmatizer.lemmatize(word,'v')
	# if still no change - time to be aggressive
	if lem == word:
		# CUT CUT CUT
		lem = lancaster_stemmer.stem(word)
	return lem

def _stemmer_words(words):
	# lem word as if a noun
	for i in range(len(words)):
		lem = lemmatizer.lemmatize(words[i],'n')
		# if unchanged
		if lem == words[i]:
			#lem word as if a verb
			lem = lemmatizer.lemmatize(words[i],'v')
		# if still no change - time to be aggressive
		if lem == words[i]:
			# CUT CUT CUT
			lem = lancaster_stemmer.stem(words[i])
		words[i]=lem
	return words
'''
def steves_stemmer(words):
	res = pd.DataFrame()
	res = res.assign(Orig=list_o_words)
	Porter=[PorterStemmer().stem(w) for w in words],
	Snowball=[SnowballStemmer('english').stemp(w) for w in words],
	Lancaster=[LancasterStemmer().stem(w) for w in words],
	WordNetLemmatizer=[WordNetLemmatizer().lemmatize(w) for w in words])

	return res
'''
def pre_process(data,tokenize=False, stop_words = []):   
	
	# variable to store final values
	proc_data = []
	
	# loop through all sentences in data
	for sentence in data:
		# store pre-processed sentence
		new_sentence = []
		
		# loop through words in sentence
		for word in word_tokenize(sentence):
			# remove punctuation
			word = re.sub(r'[^\w\w]', '', word)
			
			# remove non-unicode

			
			# remove digits
			word = re.sub(r'\d+', '', word)
			
			# remove stop words
			if word.lower() not in stop_words and word.lower() != '':
				# add lowercase words to new sentence
				new_sentence.append(word.lower())
		if tokenize:
			# Keep split
			proc_data.append(new_sentence)
		else:
			#join
			proc_data.append((" ").join(new_sentence))
	
	return proc_data
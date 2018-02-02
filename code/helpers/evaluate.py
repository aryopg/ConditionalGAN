from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import sys
import csv
import datetime
try:
    import cPickle as pickle
except:
    import pickle
import getopt
import string
import re
import unicodedata
import numpy as np
from shutil import rmtree
from os import listdir, mkdir
from os.path import join, isfile, isdir, dirname, basename, normpath, abspath, exists

from nltk.tokenize import sent_tokenize, word_tokenize

################################################################################################################
# candidate : String of candidate sentence
# motion : string of motion
#
# HOW TO USE : 
# from helpers.evaluate import BleuScore
#
# candidates = ['This is a test','This is a test','This is a test','This is a test','This is a test','This is a test','This is a test']
#
# motions = ['This house would embrace multiculturalism',
# 'This house would ban gambling',
# 'This house would ban boxing',
# 'This house believes in the use of affirmative action',
# 'This house would permit the use of performance enhancing drugs in professional sports',
# 'This house supports the one-child policy of the republic of China',
# 'This house would build the Keystone XL pipeline']
#
# BleuScore(candidate=candidates, motion=motions)
################################################################################################################

class BleuScore():
	def __init__(self, candidate, motion):
		self.smoothing = SmoothingFunction().method2
		self.claims_list = '../dataset/claims.txt'
		# self.candidate = word_tokenize(candidate.lower())
		# self.motion = motion.lower()
		self.candidate = candidate
		self.motion = motion
		
		self.reference = {}
		# self.lexicon_dictionary = pickle.load(open('../data_histo_no_article/lexicon-dict.pkl', 'rb'))
		self.lexicon_dictionary = pickle.load(open('lexicon-dict.pkl', 'rb'))

		# Make evaluation data
		fobj = csv.reader(open(self.claims_list, "rb"), delimiter = '\t')
		for idx, line in enumerate(fobj):
			if idx ==0: continue
			sentence_temp = unicodedata.normalize('NFKD', line[2].decode('utf-8')).encode('ascii', 'ignore')
			sentence_temp = sentence_temp.replace('.', '')
			sentence_temp = sentence_temp.replace('?', '')
			sentence_temp = sentence_temp.replace('"', '')
			sentence_temp = sentence_temp.replace('\'', '')
			sentence_temp = sentence_temp.replace('(', '')
			sentence_temp = sentence_temp.replace(')', '')
			sentence_temp = sentence_temp.replace('%', '')
			sentence_temp = sentence_temp.replace('$', '')
			sentence_temp = sentence_temp.replace(',', '')
			cleaned_claim = sentence_temp.replace('[REF]', '')
			cleaned_claim = cleaned_claim.lower()

			sentence_motion = unicodedata.normalize('NFKD', line[0].decode('utf-8')).encode('ascii', 'ignore')

			tokenized_claim = word_tokenize(cleaned_claim.lower())
			for idx,word in enumerate(tokenized_claim):
				if word not in self.lexicon_dictionary:
					print ("%s is not in dictionary"%word)
					tokenized_claim[idx] = '<unk>'
					# cleaned_claim = cleaned_claim.replace(word,'444')
			# tokenized_claim = word_tokenize(cleaned_claim)
			tokenized_claim.append('<eos>')



			# a=[1,2,3,4,5,1,2,3,4,5,1]
			# for n,i in enumerate(a):
			# 	if i==1:
			# 		a[n]=10

			print tokenized_claim
			# print cleaned_claim


			if len(tokenized_claim) < 5:
				continue
			else:
				if not sentence_motion.lower() in self.reference:
					self.reference[sentence_motion.lower()] = []
				self.reference[sentence_motion.lower()].append(word_tokenize(cleaned_claim.lower()))
	

		# Write to CSV
		with open('evaluation_'+ datetime.datetime.now().strftime("%Y-%m-%d:%H:%M:%S") +'.csv', 'wb') as csvfile:
			csvWriter = csv.writer(csvfile, delimiter='	')
			csvWriter.writerow(['No','Motion','Claim','Bleu Score','Unigram','Bigram','Trigram','4-Gram','Cumulative'])

			for i in range(len(candidate)):
				c = word_tokenize(self.candidate[i].lower())
				m = self.motion[i].lower()

				score1 = sentence_bleu(self.reference[m], c, smoothing_function=self.smoothing)
				score2 = sentence_bleu(self.reference[m], c, weights=(1,0,0,0), smoothing_function=self.smoothing)
				score3 = sentence_bleu(self.reference[m], c, weights=(0,1,0,0), smoothing_function=self.smoothing)
				score4 = sentence_bleu(self.reference[m], c, weights=(0,0,1,0), smoothing_function=self.smoothing)
				score5 = sentence_bleu(self.reference[m], c, weights=(0,0,0,1), smoothing_function=self.smoothing)
				score6 = sentence_bleu(self.reference[m], c, weights=(0.25,0.25,0.25,0.25), smoothing_function=self.smoothing)
				csvWriter.writerow([i+1, self.motion[i], self.candidate[i], score1, score2, score3, score4, score5, score6])

		print 'Evaluation Done! Saved To Disk!'

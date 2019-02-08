#!/usr/bin/env python

"""
* Create a corpus from previously downloaded biege book reports and use fastText to train word vectors
* 'vectorBeige.py' has code segment to download reports based on selection of regions, years and months.
"""

import glob # for unix style file paths with some patterns

import string
import re

from bs4 import BeautifulSoup

import fastText
from fastText import load_model
from fastText import util


#removing punctuation from strings courtesy SparkAndShine + ShadowRanger on
#https://stackoverflow.com/questions/265960/best-way-to-strip-punctuation-from-a-string-in-python
punc = string.punctuation
#len(punc) = 32  #these are:	'!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~'
#table for use to "translate". Since key is None (below), these chars will be deleted.
#This method is faster than string matching and replacement.
table = str.maketrans(dict.fromkeys(punc)) #table from the char to int
# {33: None, 34: None, 35: None, 36: None, 37: None, 38: None, 39: None, 40: None, 41: None,
#  42: None, 43: None, 44: None, 45: None, 46: None, 47: None, 58: None, 59: None, 60: None, 
#  61: None, 62: None, 63: None, 64: None, 91: None, 92: None, 93: None, 94: None, 95: None, 
#  96: None, 123: None, 124: None, 125: None, 126: None}
#'None' there were no keys, so defaulted to None in dict.fromkeys All of these chars will be replaced by None.

#set the path to reports
path2dir = './reports/*'   #directory with downloaded reports


#create a document corpus to train word vectors:
with open(file='./wordVec/corpus.txt', mode='w',newline=None) as corpus:
	for doc in glob.glob(path2dir):
		with open(doc,'r', newline=None) as file:
			soup = BeautifulSoup(file.read(), 'html.parser')		
			#The article content is in paras within the only <section> tags
			articleTxt = soup.section.get_text() #extract all text within the <section> tags
			#this includes \n and also 'text' contained as part of scripts
			
			#******* We don't want this stuff
			# <section class="article-content">
			# <script>
			#     var ref = document.referrer;
			#     var bBPath = 'beige-book-archive';
			#     function backToResults() {
			#         if (ref.indexOf(bBPath) > -1) {
			#             history.go(-1);
			#         }
			#         else {
			#             window.location.href = "./";
			#         }
			#     }
			# </script>
			# <p style="color:#fff;"><a href="#" onclick="backToResults();return false">‹ Back to Archive Search</a></p>
			# <h1>Beige Book Report: San Francisco</h1>
			# <p><strong>July 18, 2018</strong></p> *********** want stuff from here onwards
			
			#so we need to strip these away 
			remStr1 = soup.section.script.get_text()
			remStr2 = soup.section.find('p',string=re.compile('Back to Archive Search')).get_text()
			remStr3 = 'beige book report'
			remStr4 = 'summary of economic activity'
			
			articleTxt =  str.replace(articleTxt,remStr1,'')
			articleTxt =  str.replace(articleTxt,remStr2,'')
			
			#remove puncuation
			articleTxt = articleTxt.replace('\n',' ').lower().translate(table) #
			articleTxt =  str.replace(articleTxt,remStr3,'')
			articleTxt =  str.replace(articleTxt,remStr4,'')
			fname = file.name.replace('./reports/','')
			articleTxt = fname + articleTxt
			#fdata = re.sub(r'[0-9:]','',fdata)		#remove numbers (not okay in all contexts) 
			print(articleTxt,file = corpus)

#clean up
del(remStr1,remStr2, file, fname, soup,articleTxt, table,punc, doc)

#Train model using FastText:
#https://fasttext.cc/docs/en/options.html

f=fastText.FastText.train_unsupervised('./wordVec/corpus.txt',
	 lr=0.1, dim=100, ws=5, epoch=5, minCount=1, minCountLabel=0, minn=3, maxn=6, neg=5,
	  wordNgrams=2, loss='softmax', bucket=2000000, thread=2, lrUpdateRate=100, t=0.0001,
	   label='__label__', verbose=2, pretrainedVectors='')


#save the model for future use
f.save_model('./wordVec/saved_fastTextModel')

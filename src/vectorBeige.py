#!/usr/bin/env python
"""
Use previously trained fastText model to create document vectors for comparison. 
* Downloading reports from web as necessary per selected region, year, month
* Compute document vectors for full reports and also for specified sectors i.e. subsections
	* perform SVD to obtain 2D doc vectors and save the file to merged.csv
* Mean difference the vectors for each region (to simulate fiexed effects estimation in panel data settings)
	* perform SCD to obtain 2D doc vectors (mean differenced), save as mergedDiff.csv
* The intention is to use D3 for interactive plots. But as a check, matplotlib/seaborn plots can be created/saved right here.
"""

import glob

import string
import re

import urllib.request

from bs4 import BeautifulSoup

import fastText
from fastText import load_model
from fastText import util 

import numpy as np
import pandas as pd

from sklearn.decomposition import TruncatedSVD
#from sklearn.manifold import TSNE

import matplotlib.pyplot as plt
import seaborn as sns

#dictionary: Don't modify this. To limit selections, modify region (below).
districts = {
 			"at": 'Atlanta',
			"bo": 'Boston',
  			"ch": 'Chicago',
 			"cl": 'Cleveland',
 			"da": 'Dallas',
 			"kc": 'Kansas City',
 			"mi": 'Minneapolis',
  			"ny": 'New York',
			"ph": 'Philadelphia',
			"ri": 'Richmond',
 			"sf": 'San Francisco',
			"sl": 'St. Louis',
			"su": 'National Summary'
}

#List of sectors of the economy
sectors = ['comp_financ_bank_estate',	'comp_price',
'comp_empl_wage',		'comp_manuf_energ_trans',	'comp_retail_spend_service'] 

# Regions to compare. Same as region = list(districts) if selecting all. 
# those not needed should be commented out. Note: national summary has been commented out.
region = [			
 			"at", #Atlanta
			"bo", #Boston
  			"ch", #Chicago
 			"cl", #Cleveland
 			"da", #Dallas
 			"kc", #Kansas City
 			"mi", #Minneapolis
  			"ny", #New York
			"ph", #Philadelphia
			"ri", #Richmond
 			"sf", #San Francisco
			"sl", #St. Louis
			#"su"  #uncomment for 'national summary'
			]
#select years:
years = ['2007','2008','2009','2010','2011']#['2019','2018', '2017']

#select months: There are 8 reports each year. The months listed here appear consistenctly across all years (first report of each quarter).
#other months vary: e.g. either Aug or Sep , and Nov or Dec etc. 
months = [
		'01', #Jan
#		'04', #April
#		'07', #July
#		'10'  #October
		]

#Dictionary of months
mths  = {'01':'Jan','04':'Apr','07':'Jul','10':'Oct'}

#list of documents needed (to addi to base_url)
doclist = [y+'-'+m+'-'+r for y in years for m in months for r in region]

#base url
url_base = 'https://www.minneapolisfed.org/news-and-events/beige-book-archive/'
#directory with downloaded reports: use curl from that directory or move
	#base="https://www.minneapolisfed.org/news-and-events/beige-book-archive/"
	#curl -Os "$base[2014-2018]-{01,04,07,10}-{at,bo,ch,cl,da,kc,mi,ny,ph,ri,sf,sl,su}"
path2dir = './reports/*'

####Check if available in local dir
localReports = []
for doc in glob.glob(path2dir):
	localReports.append(doc.replace('./reports/',''))


#####Documents to download: 
docs2fetch = list(set(doclist) - set(localReports)) #convert set list

#download required files to local folder
for doc in docs2fetch:
	with open(file = './reports/'+doc,mode = 'w', newline=None) as docfile:
		url = url_base + doc
		response = urllib.request.urlopen(url)
		html_doc = response.read().decode('utf-8')
		print(html_doc, file = docfile)

#removing punctuation from strings courtesy SparkAndShine + ShadowRanger on
#https://stackoverflow.com/questions/265960/best-way-to-strip-punctuation-from-a-string-in-python
punc = string.punctuation
table = str.maketrans(dict.fromkeys(punc)) #table from the char to int

#Extract text from files to create corpus (first for full reports, then for selected sections)
#This is NOT for training the model! That is already done! (see trainBeige.py) 
#This creates a text file to serve as an input to the trained model (output will be doc-vectors).
def corpusFull(doclist):
	"""
	Create "comparefull.txt" where each line is an entire report as a single string
	That is a corpus for doc2Vec based on full (all sections) reports.
	"""
	with open(file='./wordVec/compareFull.txt', mode='w',newline=None) as corpus:
		for doc in doclist:
			with open(file='./reports/'+doc,mode='r', newline=None) as file:
				soup = BeautifulSoup(file.read().lower(), 'html.parser')
				#pList = soup.section.find_all('p')				
				articleTxt = soup.section.get_text() #extract all text within the <section> tags
				#this includes \n and also 'text' contained as part of scripts
		
				#******* don't want this stuff
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
				remStr2 = soup.section.find('p',string=re.compile('Back to Archive Search'.lower())).get_text()
				remStr3 = 'beige book report'
				remStr4 = 'summary of economic activity'
		
				articleTxt =  str.replace(articleTxt,remStr1,'')
				articleTxt =  str.replace(articleTxt,remStr2,'')
		
				#remove puncuation
				articleTxt = articleTxt.replace('\n',' ').lower().translate(table) #
				articleTxt =  str.replace(articleTxt,remStr3,'')
				articleTxt =  str.replace(articleTxt,remStr4,'')
				d = file.name[10:20]
				articleTxt = d + ' ' + articleTxt #add relevant file name			 
				print(articleTxt,file = corpus)			



#create "compare_(sectorName)_.txt" where each line is an entire report as a single string
def corpusSector(matchHeaders): 
	"""
	Create "compare_*_.txt" where each line is some section from each report as a single string
	i.e. corpora for doc2Vec based on relevant sectors.
	example: corpusSector('financ|bank|estate') will match these strings as one sector or section of a report.
	"""
	with open(file='./wordVec/comp_'+matchHeaders.replace('|','_')+'.txt', mode='w') as corpus:
		for doc in doclist:
			with open('./reports/'+doc,'r', newline=None) as file:
				soup = BeautifulSoup(file.read().lower(), 'html.parser')
				pList = soup.section.find_all('p')
				#match relevant paras
				matched = [i for i,x in enumerate(pList) if pList[i].find('strong', string=re.compile(matchHeaders.lower())) != None]
				#Some matched headers may have multiple paragraphs
				#find paras without section headers that should be part of a match
				for m in matched: #these are matches with header inside a <strong> tag in some para
					if m < len(pList)-1 : #don't exceed length of list of paragraphs in following steps.
						if pList[m+1].strong == None :	#para following a matched one with no <strong>, i.e a continuation
							matched.append(m+1) #add this paragraph to matched list. Since this is appended to end, following paras
							#will automatically be checked (bcos the loop for m in match is not yet done).
				#if there is at least one match, then 
				if len(matched) > 0:	
					tmp = '' #init. empty str
					for i in matched:
						tmp += pList[i].get_text()
						tmp = tmp.replace('\n',' ').lower().translate(table)
					
					d = file.name[10:20]
					tmp = d + ' ' + tmp #add relevant file name for reference
					print(tmp,file = corpus)
				else:
					print('No match for \''+ matchHeaders + '\' in \''+file.name[10:20]+ '\', skipping.')



#corpus for full reports
corpusFull(doclist=doclist)

#Create the sector-wise  corpus (one report each line)
for s in sectors:
	corpusSector(s.replace('comp_','').replace('_','|'))


#reload fasttext model saved earlier
f = load_model('./wordVec/saved_fastTextModel') 

############################### TODO #################
#instead of rewriting from scratch, modify to compute vectors for documents not seen before, then append to them
#only mean differencing, and SVD reduction need to be done computed each time, as per required docs.

#obtain document vectors (using sentence vector) for full reports and save to csv file
with open(file='./wordVec/doc2VecFull.csv',mode ='w') as docvec:
	with open(file='./wordVec/compareFull.txt', mode ='r', newline='\n') as d:
		for idx,line in enumerate(d):
			sv = f.get_sentence_vector(line.replace('\n',' '))
			dvStr = str(list(sv)).replace('[',"").replace(']',"")
			print(line[:10]+','+dvStr,file=docvec) #line[:10] is file id, e.g. 2018-01-sf
			
#Do the same for all sectors
for s in sectors:
		with open(file='./wordVec/doc2Vec' + s + '.csv',mode='w') as docvec:
			with open('./wordVec/'+s+'.txt', 'r', newline='\n') as d:
				for idx,line in enumerate(d):
					sv = f.get_sentence_vector(line.replace('\n',' '))
					dvStr = str(list(sv)).replace('[',"").replace(']',"")
					print(line[:10]+','+dvStr,file=docvec) #line[:10] is file id, e.g. 2018-01-sf


def svdDocVecFull(docVectors):
	"""
	Takes high dimension doc Vectors for full reports (e.g. #docs * 100), runs SVD, returns 2D vectors: 
	"""
	svd = TruncatedSVD(n_components=2).fit(docVectors)
	docVec2D_svd = svd.transform(docVectors)
	#docVec2D_svd.shape
	merged = pd.DataFrame(docVec2D_svd[:,0:2],columns=['x_full','y_full'])
	merged.index = docVectors.index
	merged['Group'] = merged.index
	merged.Group = merged.Group.str.replace(r'[0-9]+','')
	merged.Group = merged.Group.str.replace(r'-','')
	merged['Period'] = merged.index
	merged.Period = merged.Period.str.replace(r'-[a-z]+$','')
	merged['Year'] = merged.Period.str.replace(r'-[0-9]+','')
	merged['Month'] = merged.Period.str.replace(r'^[0-9]+-','')
	merged['District']='' #initialize empty column
	merged['Mon'] = ''
	#map the districts and months
	for i in range(merged.shape[0]):
		merged.District[i] = districts[merged.Group[i]]
		merged.Mon[i] = mths[str(merged.Month[i])]
	
	merged.Month = merged.Mon
	merged = merged.drop(['Period','Mon'],axis=1)
	return merged


#Compute doc vectors for each sector
def svdDocVecSec(sectors, merged):
	"""
	Takes high dimension doc Vectors for sector reports, runs SVD, returns 2D vectors:
	adds them to the one for full reports (which is another input)
	"""
	for s in sectors:
		#Viewing vectors in 2D space: SVD and T-SNE
		docVectors = pd.read_csv('wordVec/doc2Vec' + s + '.csv',index_col = 0, header=None) 	
		# from sklearn.decomposition import TruncatedSVD
		svd = TruncatedSVD(n_components=2).fit(docVectors)
		docVec2D_svd = svd.transform(docVectors)
		tmpDF = pd.DataFrame(docVec2D_svd[:,0:2],columns=['x_'+s,'y_'+s])
		tmpDF.index = docVectors.index
		merged = pd.concat([merged, tmpDF], axis=1, sort=False)
	return merged #careful about indentation, this should be outside the sector loop


def regDiff(origVecs):
	"""
	Obtain "mean-differenced" vectors for each region (to remove time-invariant features) from original higher dim vectors
	"""
	diffVectors = origVecs #orig higher dimension vector
	diffVectors['Group'] = diffVectors.index
	diffVectors.Group = diffVectors.Group.str.replace(r'[0-9]+','')
	diffVectors.Group = diffVectors.Group.str.replace(r'-','')	
	#use group to calculate df-df.mean() per each group. 'Time detrended,' instead of first-differenced.
	for r in region:
		b = diffVectors[diffVectors.Group==r].iloc[:,:100]
		b = b - b.mean() #subtract the average
		b['Group'] = r
		diffVectors[diffVectors.Group==r] = b
	return diffVectors


def svdDiffSec(sectors, merged):
	"""
	Performs mean-differencing (by region) then SVD 2D reduction for each sector, concats with 2D mean-diff for Full reports.
	Differences from 'svdDocVecSec()'
	* mean differencing (calls regDiff())
	* note that a string column is dropped whenn calling SVD
	"""
	for s in sectors:
		#read in doc vectors
		docVectors = pd.read_csv('wordVec/doc2Vec' + s + '.csv',index_col = 0, header=None)
		docVectors = regDiff(docVectors) #region wise mean differencing
		tmpDF = docVectors.drop(['Group'],axis=1)
		svd = TruncatedSVD(n_components=2).fit(tmpDF)
		docVec2D_svd = svd.transform(tmpDF)
		tmpDF = pd.DataFrame(docVec2D_svd[:,0:2],columns=['x_'+s,'y_'+s])
		tmpDF.index = docVectors.index
		merged = pd.concat([merged, tmpDF], axis=1, sort=False)
	return merged #careful about indentation, this should be outside the sector loop


#Dimensionality reduction 2D space (SVD)
origDocVecFull = pd.read_csv('wordVec/doc2VecFull.csv',index_col = 0, header=None) 

#2D vectors for full reports 
mergedOrig =  svdDocVecFull(docVectors=origDocVecFull)
# and then for all sectors
mergedOrig = svdDocVecSec(sectors = sectors, merged = mergedOrig)

# save for later use, e.g. feed to D3
mergedOrig.to_csv("./D3/merged.csv")

# Moving on to differences across time. 
#Dynamic case only, must have at least 2 periods.
#len(months) + len(years) # need > 1


# Read in higher dimension docVec for full reports:
#and obtain mean differenced version
diffFull = regDiff(origDocVecFull)
#Perform the singular value decomposition
#	remove the string valued column (to pass on to SVD)
mergedDiff =  svdDocVecFull(docVectors = diffFull.drop(['Group'],axis=1))
#obtain sector wise mean-differenced vectors and append/concat with meredDiff
mergedDiff = svdDiffSec(sectors = sectors, merged = mergedDiff)

mergedDiff.to_csv("./D3/mergedDiff.csv")

#Plotting the differenced and reduced document vectors
plotSec = ['full'] + sectors

#or to limit section to 
#sec = [s for i,s in enumerate(sectors) if i%2==0] #[e.g.'comp_financ_bank_estate', 'comp_empl_wage', 'comp_retail_spend_service']
#plotSec = ['full'] + sec

num_plots = len(plotSec)

if num_plots % 2 == 1:
	num_rows = int((num_plots+1)/2)
else:
	num_rows =  int(num_plots/2)


fig, axes = plt.subplots(num_rows,2, figsize=(8, 9))
for i in range(num_rows):
	for j in range(2):
		idx = 2*i+j
		if idx < num_plots:
			s = plotSec[idx]	
			sns.scatterplot(x="x_"+s, y="y_"+s, hue="District", data=mergedDiff, ax = axes[i,j])
			axes[i,j].set_title(s.replace('comp_',''))
			axes[i,j].set_xlabel('')
			axes[i,j].set_ylabel('')
			axes[i,j].set_yticklabels('')
			axes[i,j].set_xticklabels('')
			axes[i,j].get_legend().remove()

if num_plots % 2 == 1:
	axes[-1,-1].axis('off')

axes[0,0].legend(loc='upper center', bbox_to_anchor=(1, 1.5),
          ncol=5, fancybox=True, shadow=True)

#plt.show()
#axes[0,0].legend(loc='left center', bbox_to_anchor=(1.5, .5),
#          ncol=1, fancybox=True, shadow=True
plt.savefig('images/orig.png',bbox_inches="tight")
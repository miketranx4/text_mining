import csv
import urllib2
import numpy as np
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.cross_validation import cross_val_score
from sklearn.svm import SVC
import re

def StripInline(s, w1 = '<code', w2 = '</code>'):
	""" This function takes a string s as input, and removes all patterns that
	start with w1, and end with w2, but stays within a line (No /n in the middle)
	"""
	return re.sub(w1 + '((?!' + w1 + ').)*?' + w2, '', s, 
				flags = re.MULTILINE | re.S)

def StripBlock(s, w1 = '<pre', w2 = '</pre>'):
	""" This function takes a string s as input, and removes all patterns that
	start with w1 and end with w2, and spans multiple lines
	"""
	return re.sub('^' + w1 + '((?!' + w1 + ').)*' + w2 + '$', '', s, 
			flags = re.MULTILINE | re.S) 

def save_obj(obj, name ):
    with open(name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def process_text(text):
	del_sign_block = [('<pre>', '</pre>'), ('$$','$$'),('$','$'), ('<code>','</code>'), ('\begin{','\end{')]
	for sign in del_sign_block:
		x = sign[0]
		y = sign[1]
		text = StripInline(text, x, y)
		text = StripBlock(text, x, y)
	text = StripInline(text,'<','>')
	text = re.sub("[^a-zA-Z\s]",' ', text)
	text = re.sub('\\s+', ' ', text)
	text_list = [x.lower() for x in text.split() if x.lower() not in stopwords_list]
	return text_list

u = urllib2.urlopen("http://www.textfixer.com/resources/common-english-words.txt")
stopwords_list = u.read().split(',')

file_read = open('train.csv', 'r')
data = csv.reader(file_read)
data.next()
wordset_title = set()
wordset_big = set()
num_row = 0
for row in data:
	title = row[1]
	body = row[2]
	processed_title = set(process_text(title))
	wordset_title = wordset_title.union(processed_title)
	wordset_big = wordset_big.union(set(process_text(body)))
	wordset_big = wordset_big.union(processed_title)
	num_row += 1

num_words_big = len(wordset_big)
big_hash = dict()
i = 0
for word in wordset_big:
	big_hash[word] = i
	i += 1

file_read = open('train.csv', 'r')
data = csv.reader(file_read)
data.next()
count_matrix = np.zeros((num_row, num_words_big))
i = 0
for row in data:
	title = row[1]
	body = row[2]
	processed_title = process_text(title)
	processed_body = process_text(body)
	for word in processed_body+processed_title:
		count_matrix[i, big_hash[word]] += 1
	i += 1

del_col = list()
for i in range(0, count_matrix.shape[1]):
	total_count = sum(count_matrix[:,i])
	if(total_count <= 10):
		del_col.append(i)

rarewords = set()
for p in big_hash.items():
	if(p[1] in del_col):
		rarewords.add(p[0])

title_hash = dict()
wordset_title = wordset_title - rarewords 
num_words_title = len(wordset_title)
i = 0
for word in wordset_title:
	title_hash[word] = i
	i += 1

big_hash = dict()
wordset_big = wordset_big - rarewords 
num_words_big = len(wordset_big)
for word in wordset_big:
	big_hash[word] = i
	i += 1

del count_matrix
print "Doing histrogram filtering"

def create_histmatrix(filename, title_hash, big_hash):
	num_words_big = len(big_hash.keys())
	num_words_title = len(title_hash.keys())
	feat_matrix = np.zeros((num_row, num_words_title+num_words_big), dtype='float')
	file_read = open(filename, 'r')
	data = csv.reader(file_read)
	data.next()
	i = 0
	for row in data:
		title = row[1]
		body = row[2]
		processed_title = process_text(title)
		processed_body = process_text(body)
		for word in processed_title:
			if word in title_hash.keys():
				feat_matrix[i, title_hash[word]] += 1.0
		for word in processed_body + processed_title:
			if word in big_hash.keys():
				feat_matrix[i, big_hash[word]] += 1.0
		i += 1
	return feat_matrix


def create_featmatrix(filename, title_hash, big_hash):
	num_words_big = len(big_hash.keys())
	num_words_title = len(title_hash.keys())
	feat_matrix = np.zeros((num_row, num_words_title+num_words_big), dtype='float')
	file_read = open(filename, 'r')
	data = csv.reader(file_read)
	data.next()
	i = 0
	for row in data:
		title = row[1]
		body = row[2]
		processed_title = process_text(title)
		processed_body = process_text(body)
		for word in processed_title:
			if word in title_hash.keys():
				feat_matrix[i, title_hash[word]] += 1.0
		for word in processed_body+processed_title:
			if word in big_hash.keys():
				feat_matrix[i, big_hash[word]] += 1.0
		if(np.sum(feat_matrix[i,:num_words_title]) != 0):
			feat_matrix[i,:num_words_title] = feat_matrix[i,:num_words_title] / float(np.sum(feat_matrix[i,:num_words_title]))
		if(np.sum(feat_matrix[i,num_words_title:]) != 0):
			feat_matrix[i,num_words_title:] = feat_matrix[i,num_words_title:] / float(np.sum(feat_matrix[i,num_words_title:]))
		i += 1
	return feat_matrix

def create_featmatrix_power(filename, title_hash, big_hash):
	power_feat = ['\<\-','c\(', '[a-zA-Z0-9]\$[a-zA-Z0-9]', '\$\$', '\\%\\*\\%', '\\~', 'data\\.frame', 'library\\(', 'predict\\(','\\<code\\>', '[a-zA-Z0-9]\\.[a-zA-Z0-9]', 'return','begin\{', '\\int', '\\sum_', '\\prod_', '\\Phi', '\\delta', 'f\(x\)', 'np\.', 'import\\snumpy']
	num_words_big = len(big_hash.keys())
	num_words_title = len(title_hash.keys())
	num_power = len(power_feat)
	feat_matrix = np.zeros((num_row, num_words_title+num_words_big+num_power), dtype='float')
	file_read = open(filename, 'r')
	data = csv.reader(file_read)
	data.next()
	i = 0
	for row in data:
		title = row[1]
		body = row[2]
		processed_title = process_text(title)
		processed_body = process_text(body)
		for word in processed_title:
			if word in title_hash.keys():
				feat_matrix[i, title_hash[word]] += 1.0
		for word in processed_body+processed_title:
			if word in big_hash.keys():
				feat_matrix[i, big_hash[word]] += 1.0
		if(sum(feat_matrix[i,:num_words_title]) != 0):
			feat_matrix[i,:num_words_title] = feat_matrix[i,:num_words_title] / float(sum(feat_matrix[i,:num_words_title]))
		if(sum(feat_matrix[i,num_words_title:]) != 0):
			feat_matrix[i,num_words_title:] = feat_matrix[i,num_words_title:] / float(sum(feat_matrix[i,num_words_title:]))
		j = num_words_title+num_words_big
		for p in power_feat:
			feat_matrix[i,j] = len(re.findall(p, title+body))
			j += 1
		i += 1
	return feat_matrix

def create_featmatrix_poweronly(filename):
	power_feat = ['\<\-','c\(', '[a-zA-Z0-9]\$[a-zA-Z0-9]', '\$\$', '\\%\\*\\%', '\\~', 'data\\.frame', 'library\\(', 'predict\\(','\\<code\\>', '[a-zA-Z0-9]\\.[a-zA-Z0-9]', 'return','begin\{', '\\int', '\\sum_', '\\prod_', '\\Phi', '\\delta', 'f\(x\)', 'np\.', 'import\\snumpy']
	num_power = len(power_feat)
	feat_matrix = np.zeros((num_row, num_power), dtype='float')
	file_read = open(filename, 'r')
	data = csv.reader(file_read)
	data.next()
	i = 0
	for row in data:
		title = row[1]
		body = row[2]
		processed_title = process_text(title)
		processed_body = process_text(body)
		j = 0
		for p in power_feat:
			feat_matrix[i,j] = len(re.findall(p, title+body))
			j += 1
		i += 1
	return feat_matrix

hist_count = create_histmatrix('train.csv', title_hash, big_hash)
num_words_title = len(title_hash.keys())
num_words_big = len(big_hash.keys())

import matplotlib.pyplot as plt
plt.hist(hist_count.sum(axis = 0), bins = 1000)
plt.show()

file_read = open('train.csv', 'r')
data = csv.reader(file_read)
data.next()
label = []
for row in data:
	if 'r' in [x.lower() for x in row[3].split(' ')]:
		label.append(1)
	else:
		label.append(0)

#Picking Threshold#
lower = [25, 35, 50, 150, 200]
#higher = [5000, 8000, 10000, 12000, 15000, 21000]
#higher = [25000]
higher = [10000, 15000, 21000]
#higher = [38000]
cv_scores = []
for l in lower:
	for h in higher:
		del_hist_col = list()
		for i in range(num_words_title, hist_count.shape[1]):
			total_count = sum(hist_count[:,i])
			if(total_count < l or total_count > h):
				del_hist_col.append(i)
		num_column_bighash_delete = len(del_hist_col)
		delwords_hist = set()
		for p in big_hash.items():
			if(p[1] in del_hist_col):
				delwords_hist.add(p[0])
		num_column_title_delete = 0
		for pair in title_hash.items():
			if(pair[0] in delwords_hist):
				del_hist_col.append(pair[1])
				num_column_title_delete += 1
		new_col_index = [x for x in range(0, num_words_big+num_words_title) if x not in del_hist_col]
		print "done finding words hash"
		feat_matrix = hist_count[:,new_col_index]
		print l,h
		print feat_matrix.shape
		


##Using the best threshold#
#lower = 150, higher = 21000#

del_hist_col = list()
for i in range(num_words_title, hist_count.shape[1]):
	total_count = sum(hist_count[:,i])
	if(total_count < 300):
		del_hist_col.append(i)

rarewords_hist = set()
for p in big_hash.items():
	if(p[1] in del_hist_col):
		rarewords_hist.add(p[0])

title_hash_hist = dict()
wordset_title_hist = wordset_title - rarewords_hist 
num_words_title_hist = len(wordset_title_hist)
i = 0
for word in wordset_title_hist:
	title_hash_hist[word] = i
	i += 1

big_hash_hist = dict()
wordset_big_hist = wordset_big - rarewords_hist 
num_words_big_hist = len(wordset_big_hist)
for word in wordset_big_hist:
	big_hash_hist[word] = i
	i += 1

save_obj(title_hash_hist, 'title_hash')
save_obj(big_hash_hist, 'big_hash')
del hist_count
feat_matrix = create_featmatrix('train.csv', title_hash_hist, big_hash_hist)
feat_matrix_power = create_featmatrix_power('train.csv', title_hash_hist, big_hash_hist)
feat_matrix_poweronly = create_featmatrix_poweronly('train.csv')


#Cross Validation#
max_d = [None, 50, 100, 1000]
criterion = ["gini"], "entropy"]
max_features = ['sqrt', 'log2', 0.025]
cv_scores = []
cv_scores_power = []
cv_scores_poweronly = []

for m in max_d:
	for c in criterion:
		for f in max_features:
			rf_clf = RandomForestClassifier(n_estimators=10, criterion=c, max_features=f, max_depth=m)
			rf_score = cross_val_score(rf_clf, feat_matrix, label, cv=10, n_jobs=-1)
			rf_score_power = cross_val_score(rf_clf, feat_matrix_power, label, cv=10, n_jobs=-1)
			rf_score_power_only = cross_val_score(rf_clf, feat_matrix_poweronly, label, cv=10, n_jobs=-1)
			cv_scores.append(np.mean(rf_score))
			cv_scores_power.append(np.mean(rf_score_power))
			cv_scores_poweronly.append(np.mean(rf_score_power_only))
			print m, f

del hist_count
print "done finding words hash"



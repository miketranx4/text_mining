import csv
import urllib2
import numpy as np
import pickle

import re

def create_featmatrix(filename, title_hash, big_hash):
	num_words_big = len(big_hash.keys())
	num_words_title = len(title_hash.keys())
	file_read = open(filename, 'r')
	data = csv.reader(file_read)
	data.next()
	num_row = 0
	for row in data:
		num_row += 1
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
		if(sum(feat_matrix[i,:num_words_title]) != 0):
			feat_matrix[i,:num_words_title] = feat_matrix[i,:num_words_title] / float(sum(feat_matrix[i,:num_words_title]))
		if(sum(feat_matrix[i,num_words_title:]) != 0):
			feat_matrix[i,num_words_title:] = feat_matrix[i,num_words_title:] / float(sum(feat_matrix[i,num_words_title:]))
		i += 1
	return feat_matrix

def create_featmatrix_power(filename, title_hash, big_hash):
	power_feat = ['\<\-','c\(', '[a-zA-Z0-9]\$[a-zA-Z0-9]', '\$\$', '\\%\\*\\%', '\\~', 'data\\.frame', 'library\\(', 'predict\\(','\\<code\\>', '[a-zA-Z0-9]\\.[a-zA-Z0-9]', 'return','begin\{', '\\int', '\\sum_', '\\prod_', '\\Phi', '\\delta', 'f\(x\)', 'np\.', 'import\\snumpy']
	num_words_big = len(big_hash.keys())
	num_words_title = len(title_hash.keys())
	num_power = len(power_feat)
	file_read = open(filename, 'r')
	data = csv.reader(file_read)
	data.next()
	num_row = 0
	for row in data:
		num_row += 1
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
	num_row = 
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

file_read = open('train.csv', 'r')
data = csv.reader(file_read)
data.next()
label_r = []
for row in data:
	if 'r' in [x.lower() for x in row[3].split(' ')]:
		label_r.append(1)
	else:
		label_r.append(0)

file_read = open('train.csv', 'r')
data = csv.reader(file_read)
data.next()
label_stat = []
for row in data:
	if 'statistics' in [x.lower() for x in row[3].split(' ')]:
		label_stat.append(1)
	else:
		label_stat.append(0)

file_read = open('train.csv', 'r')
data = csv.reader(file_read)
data.next()
label_machine = []
for row in data:
	if 'machine-learning' in [x.lower() for x in row[3].split(' ')]:
		label_machine.append(1)
	else:
		label_machine.append(0)

file_read = open('train.csv', 'r')
data = csv.reader(file_read)
data.next()
label_math = []
for row in data:
	if 'math' in [x.lower() for x in row[3].split(' ')]:
		label_math.append(1)
	else:
		label_math.append(0)

file_read = open('train.csv', 'r')
data = csv.reader(file_read)
data.next()
label_numpy = []
for row in data:
	if 'numpy' in [x.lower() for x in row[3].split(' ')]:
		label_numpy.append(1)
	else:
		label_numpy.append(0)


def load_obj(name ):
    with open(name + '.pkl', 'rb') as f:
        return pickle.load(f)

title_hash = load_obj('title_hash')
big_hash = load_obj('big_hash')

rf_clf1 = RandomForestClassifier(n_estimators=10)
rf_clf2 = RandomForestClassifier(n_estimators=10)

feat_matrix = create_featmatrix('train.csv', title_hash, big_hash)
feat_matrix_power = create_featmatrix_power('train.csv', title_hash, big_hash)
feat_matrix_poweronly = create_featmatrix_poweronly('train.csv')

rf_score_r = cross_val_score(rf_clf, feat_matrix, label_r, cv=10, n_jobs=-1)
rf_score_stat = cross_val_score(rf_clf, feat_matrix, label_stat, cv=10, n_jobs=-1)
rf_score_machine = cross_val_score(rf_clf, feat_matrix, label_machine, cv=10, n_jobs=-1)
rf_score_math = cross_val_score(rf_clf, feat_matrix, label_math, cv=10, n_jobs=-1)
rf_score_numpy = cross_val_score(rf_clf, feat_matrix, label_numpy, cv=10, n_jobs=-1)

rf_score_r_pow = cross_val_score(rf_clf, feat_matrix_poweronly, label_r, cv=10, n_jobs=-1)
rf_score_stat_pow = cross_val_score(rf_clf, feat_matrix_poweronly, label_stat, cv=10, n_jobs=-1)
rf_score_machine_pow = cross_val_score(rf_clf, feat_matrix_poweronly, label_machine, cv=10, n_jobs=-1)
rf_score_math_pow = cross_val_score(rf_clf, feat_matrix_poweronly, label_math, cv=10, n_jobs=-1)
rf_score_numpy_pow = cross_val_score(rf_clf, feat_matrix_poweronly, label_numpy, cv=10, n_jobs=-1)

rf_score_r_comb = cross_val_score(rf_clf, feat_matrix_power, label_r, cv=10, n_jobs=-1)
rf_score_stat_comb = cross_val_score(rf_clf, feat_matrix_power, label_stat, cv=10, n_jobs=-1)
rf_score_machine_comb = cross_val_score(rf_clf, feat_matrix_power, label_machine, cv=10, n_jobs=-1)
rf_score_math_comb = cross_val_score(rf_clf, feat_matrix_power, label_math, cv=10, n_jobs=-1)
rf_score_numpy_comb = cross_val_score(rf_clf, feat_matrix_power, label_numpy, cv=10, n_jobs=-1)

rf_clf1 = RandomForestClassifier(n_estimators=200, max_features=0.025)
rf_clf2 = RandomForestClassifier(n_estimators=200, max_features=0.025)
rf_clf3 = RandomForestClassifier(n_estimators=200, max_features=0.025)
rf_clf4 = RandomForestClassifier(n_estimators=200, max_features=0.025)
rf_clf5 = RandomForestClassifier(n_estimators=200, max_features=0.025)

randomforest_fit_r = rf_clf1.fit(feat_matrix_power, label_r)
randomforest_fit_stat = rf_clf2.fit(feat_matrix_power, label_stat)
randomforest_fit_machine = rf_clf3.fit(feat_matrix_power, label_machine)
randomforest_fit_math = rf_clf4.fit(feat_matrix_power, label_math)
randomforest_fit_numpy = rf_clf5.fit(feat_matrix_power, label_numpy)

test_filename = "XtestKaggle2.csv"
kg_feat_matrix = create_featmatrix_power(test_filename, title_hash, big_hash)

pred_r = randomforest_fit_r.predict(kg_feat_matrix)
pred_stat = randomforest_fit_stat.predict(kg_feat_matrix)
pred_machine = randomforest_fit_machine.predict(kg_feat_matrix)
pred_math = randomforest_fit_math.predict(kg_feat_matrix)
pred_numpy = randomforest_fit_numpy.predict(kg_feat_matrix)


pred = np.vstack((pred_r,pred_stat,pred_machine,pred_math,pred_numpy)).transpose()

kaggle_submit = open('final_kaggle_pred2.csv', 'w')
writer = csv.writer(kaggle_submit, delimiter=',')
writer.writerow(['Id', 'statistics','machine-learning','r','numpy','math'])
i = 1
for p in pred:
	writer.writerow([i, p[1], p[2],p[0],p[4],p[3]])
	i = i+1

kaggle_submit.close()




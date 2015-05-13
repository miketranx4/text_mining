from sklearn.ensemble import RandomForestClassifier
from sklearn.cross_validation import cross_val_score
from sklearn.metrics import confusion_matrix

ind = np.random.permutation(range(0, 27425))
perm_feat_matrix = feat_matrix[ind, :]
perm_feat_matrix_poweronly = feat_matrix_poweronly[ind, :]
perm_feat_matrix_power = feat_matrix_power[ind, :]
perm_label = np.array(label)[ind]

#ROC word only model#
clf = RandomForestClassifier(n_estimators=10, max_depth=100, criterion='entropy', max_features=0.025)
randomforest_fit = clf.fit(perm_feat_matrix[:20000,:], perm_label[:20000])
conf_mat = confusion_matrix(perm_label[20000:], randomforest_fit.predict(perm_feat_matrix[20000:]))

from sklearn.metrics import roc_curve, auc
fpr, tpr, thresholds = roc_curve(perm_label[20000:], randomforest_fit.predict_proba(perm_feat_matrix)[20000:,1])
roc_auc = auc(fpr, tpr)
print "Area under the ROC curve : %f" % roc_auc

import pylab as pl
pl.clf()
pl.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % roc_auc)
pl.plot([0, 1], [0, 1], 'k--')
pl.xlim([0.0, 1.0])
pl.ylim([0.0, 1.0])
pl.xlabel('False Positive Rate')
pl.ylabel('True Positive Rate')
pl.title('ROC')
pl.legend(loc="lower right")
pl.show()

#ROC power only model#
clf = RandomForestClassifier(n_estimators=10, max_depth=50, criterion='entropy', max_features=0.025)
randomforest_fit = clf.fit(perm_feat_matrix_poweronly[:20000,:], perm_label[:20000])
conf_mat = confusion_matrix(perm_label[20000:], randomforest_fit.predict(perm_feat_matrix_poweronly[20000:]))

from sklearn.metrics import roc_curve, auc
fpr, tpr, thresholds = roc_curve(perm_label[20000:], randomforest_fit.predict_proba(perm_feat_matrix_poweronly)[20000:,1])
roc_auc = auc(fpr, tpr)
print "Area under the ROC curve : %f" % roc_auc

import pylab as pl
pl.clf()
pl.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % roc_auc)
pl.plot([0, 1], [0, 1], 'k--')
pl.xlim([0.0, 1.0])
pl.ylim([0.0, 1.0])
pl.xlabel('False Positive Rate')
pl.ylabel('True Positive Rate')
pl.title('ROC')
pl.legend(loc="lower right")
pl.show()


#ROC combined model#
clf = RandomForestClassifier(n_estimators=10, max_depth=50, criterion='entropy', max_features=0.025)
randomforest_fit = clf.fit(perm_feat_matrix_power[:20000,:], perm_label[:20000])
conf_mat = confusion_matrix(perm_label[20000:], randomforest_fit.predict(perm_feat_matrix_power[20000:]))

from sklearn.metrics import roc_curve, auc
fpr, tpr, thresholds = roc_curve(perm_label[20000:], randomforest_fit.predict_proba(perm_feat_matrix_power)[20000:,1])
roc_auc = auc(fpr, tpr)
print "Area under the ROC curve : %f" % roc_auc

import pylab as pl
pl.clf()
pl.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % roc_auc)
pl.plot([0, 1], [0, 1], 'k--')
pl.xlim([0.0, 1.0])
pl.ylim([0.0, 1.0])
pl.xlabel('False Positive Rate')
pl.ylabel('True Positive Rate')
pl.title('ROC')
pl.legend(loc="lower right")
pl.show()


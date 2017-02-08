#!/usr/bin/env python
from nltk.tokenize import RegexpTokenizer
from stop_words import get_stop_words
from nltk.stem.porter import PorterStemmer
from gensim import corpora, models
from datetime import datetime
import time
import gensim
import numpy as np
import matplotlib.pyplot as plt
from TweetUtility import TweetExtract
from sklearn.metrics import classification_report

from sklearn.ensemble import AdaBoostClassifier,RandomForestClassifier
from sklearn import svm, datasets
from sklearn.model_selection import cross_val_score,train_test_split
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import make_gaussian_quantiles
import time
import pickle

start_time = time.time()

topic_no=10
cv_no=10
wt=1000
legitimate_name=r"legitimate_large_50.txt"
  
legtimate = TweetExtract(legitimate_name,LDA_topic_number=topic_no,passNum=20,target=0)


pollute_name=r"social_honeypot_icwsm_2011/pollute_medium_50.txt"
  
pollute = TweetExtract(pollute_name,LDA_topic_number=topic_no,passNum=20,target=1)


f = open('legtimate_last_process.pckl', 'wb')
pickle.dump(legtimate, f)
f.close()

f = open('pollute_last_process.pckl', 'wb')
pickle.dump(pollute, f)
f.close()

X_tweet_topic_feature=legtimate.docTopic+pollute.docTopic
X_tweet_ratioURL_feature=legtimate.ratioURL+pollute.ratioURL
X_tweet_avgLength_feature=legtimate.avgLength+pollute.avgLength
X_tweet_avgTweetin_feature=legtimate.avgTweetin+pollute.avgTweetin
Y_tweet_target=legtimate.docTarget+pollute.docTarget

X_tweet_topic_feature=np.asarray(X_tweet_topic_feature)
X_tweet_ratioURL_feature=np.asarray(X_tweet_ratioURL_feature).reshape(-1,1)
X_tweet_avgLength_feature=np.asarray(X_tweet_avgLength_feature).reshape(-1,1)
X_tweet_avgTweetin_feature=np.asarray(X_tweet_avgTweetin_feature).reshape(-1,1)
X_tweet_Heuristic_feature=np.concatenate((X_tweet_ratioURL_feature,X_tweet_avgLength_feature),axis=1)

Y_tweet_target=np.asarray(Y_tweet_target)
print(X_tweet_topic_feature.shape)
print(X_tweet_ratioURL_feature.shape)
print(X_tweet_avgLength_feature.shape)
print(X_tweet_avgTweetin_feature.shape)
print(X_tweet_Heuristic_feature.shape)
print(Y_tweet_target.shape)

number_user=(Y_tweet_target.shape)[0]
print("Load Data  and Feature extraction Time--- %s seconds ---" % (time.time() - start_time))


Test_tr_ratio=[0.001,0.002,0.005,0.01,0.05,0.1,0.2,0.5,0.75,0.9]

#Test_tr_ratio=[0.5,0.75,0.85]

for tr_ratio in Test_tr_ratio:
	start_time_hy = time.time()
	trNum=number_user*tr_ratio
	print("train data is "+ str(tr_ratio*100)+"% of the total users and the total number is "+str(trNum))
	#X_tweet_hybrid_feature=np.concatenate((X_tweet_topic_feature*wt/trNum,X_tweet_ratioURL_feature,X_tweet_avgLength_feature),axis=1)
	X_tweet_hybrid_feature=np.concatenate((X_tweet_topic_feature,X_tweet_ratioURL_feature,X_tweet_avgLength_feature),axis=1)
	X_train, X_test, y_train, y_test = train_test_split(X_tweet_hybrid_feature, Y_tweet_target, train_size=tr_ratio, random_state=0)
	clf_svm_hy=svm.SVC(kernel='linear', C=1)
	clf_svm_hy.fit(X_train,y_train)
	y_prf_svm_hy=clf_svm_hy.predict(X_test)
	clf_AdaBoost_hy=AdaBoostClassifier(n_estimators=100)
	clf_AdaBoost_hy.fit(X_train,y_train)
	y_prf_AdaBoost_hy=clf_AdaBoost_hy.predict(X_test)
	clf_RF_hy = RandomForestClassifier(n_estimators=10)
	clf_RF_hy.fit(X_train,y_train)
	y_prf_RF_hy=clf_RF_hy.predict(X_test)
	target_names = ['human', 'bot']
	print("SVM:hybrid model with " + str(number_user*tr_ratio)+":")
	print(classification_report(y_test, y_prf_svm_hy, target_names=target_names))
	print("SVM:hybrid model Accuracy:"+str(accuracy_score(y_test, y_prf_svm_hy)))
	print("AdaBoost:hybrid model with " + str(number_user*tr_ratio)+":")
	print(classification_report(y_test, y_prf_AdaBoost_hy, target_names=target_names))
	print("AdaBoost:hybrid model Accuracy:"+str(accuracy_score(y_test, y_prf_AdaBoost_hy)))
	print("Random Forest:hybrid model with " + str(number_user*tr_ratio)+":")
	print(classification_report(y_test, y_prf_RF_hy, target_names=target_names))
	print("Random Forest:hybrid model Accuracy:"+str(accuracy_score(y_test, y_prf_RF_hy)))
	print("--- %s hybrid seconds ---" % (time.time() - start_time_hy))

	start_time_tp = time.time()
	X_train, X_test, y_train, y_test = train_test_split(X_tweet_topic_feature, Y_tweet_target, train_size=tr_ratio, random_state=0)
	clf_svm_tp=svm.SVC(kernel='linear', C=1)
	clf_svm_tp.fit(X_train,y_train)
	y_prf_svm_tp=clf_svm_tp.predict(X_test)
	clf_AdaBoost_tp=AdaBoostClassifier(n_estimators=100)
	clf_AdaBoost_tp.fit(X_train,y_train)
	y_prf_AdaBoost_tp=clf_AdaBoost_tp.predict(X_test)
	clf_RF_tp = RandomForestClassifier(n_estimators=10)
	clf_RF_tp.fit(X_train,y_train)
	y_prf_RF_tp=clf_RF_tp.predict(X_test)
	target_names = ['human', 'bot']
	print("SVM:topic model only with " + str(number_user*tr_ratio)+":")
	print(classification_report(y_test, y_prf_svm_tp, target_names=target_names))
	print("SVM:topic model only Accuracy:"+str(accuracy_score(y_test, y_prf_svm_tp)))
	print("AdaBoost:topic model only with " + str(number_user*tr_ratio)+":")
	print(classification_report(y_test, y_prf_AdaBoost_tp, target_names=target_names))
	print("AdaBoost:topic model only Accuracy:"+str(accuracy_score(y_test, y_prf_AdaBoost_tp)))
	print("Random Forest:topic model only with " + str(number_user*tr_ratio)+":")
	print(classification_report(y_test, y_prf_RF_tp, target_names=target_names))
	print("Random Forest:topic model Accuracy:"+str(accuracy_score(y_test, y_prf_RF_tp)))
	print("--- %s Topic seconds ---" % (time.time() - start_time_tp))
	
	start_time_URL = time.time()
	X_train, X_test, y_train, y_test = train_test_split(X_tweet_ratioURL_feature, Y_tweet_target, train_size=tr_ratio, random_state=0)
	clf_svm_ratioURL=svm.SVC(kernel='linear', C=1)
	clf_svm_ratioURL.fit(X_train,y_train)
	y_prf_svm_ratioURL=clf_svm_ratioURL.predict(X_test)
	clf_AdaBoost_ratioURL=AdaBoostClassifier(n_estimators=100)
	clf_AdaBoost_ratioURL.fit(X_train,y_train)
	y_prf_AdaBoost_ratioURL=clf_AdaBoost_ratioURL.predict(X_test)
	clf_RF_ratioURL = RandomForestClassifier(n_estimators=10)
	clf_RF_ratioURL.fit(X_train,y_train)
	y_prf_RF_ratioURL=clf_RF_ratioURL.predict(X_test)
	target_names = ['human', 'bot']
	print("SVM:ratioURL with " + str(number_user*tr_ratio)+":")
	print(classification_report(y_test, y_prf_svm_ratioURL, target_names=target_names))
	print("SVM:ratioURL Accuracy:"+str(accuracy_score(y_test, y_prf_svm_ratioURL)))
	print("AdaBoost:ratioURL with " + str(number_user*tr_ratio)+":")
	print(classification_report(y_test, y_prf_AdaBoost_ratioURL, target_names=target_names))
	print("AdaBoost:ratioURL Accuracy:"+str(accuracy_score(y_test, y_prf_AdaBoost_ratioURL)))
	print("Random Forest:ratioURL with " + str(number_user*tr_ratio)+":")
	print(classification_report(y_test, y_prf_RF_ratioURL, target_names=target_names))
	print("Random Forest:ratioURL Accuracy:"+str(accuracy_score(y_test, y_prf_RF_ratioURL)))
	print("--- %s URL seconds ---" % (time.time() - start_time_URL))
	
	start_time_avl = time.time()		
	X_train, X_test, y_train, y_test = train_test_split(X_tweet_avgLength_feature, Y_tweet_target, train_size=tr_ratio, random_state=0)
	clf_svm_avgLength=svm.SVC(kernel='linear', C=1)
	clf_svm_avgLength.fit(X_train,y_train)
	y_prf_svm_avgLength=clf_svm_avgLength.predict(X_test)
	clf_AdaBoost_avgLength=AdaBoostClassifier(n_estimators=100)
	clf_AdaBoost_avgLength.fit(X_train,y_train)
	y_prf_AdaBoost_avgLength=clf_AdaBoost_avgLength.predict(X_test)
	clf_RF_avgLength = RandomForestClassifier(n_estimators=10)
	clf_RF_avgLength.fit(X_train,y_train)
	y_prf_RF_avgLength=clf_RF_avgLength.predict(X_test)
	target_names = ['human', 'bot']
	print("SVM:avgLength with" + str(number_user*tr_ratio)+":")
	print(classification_report(y_test, y_prf_svm_avgLength, target_names=target_names))
	print("SVM:avgLength with Accuracy:"+str(accuracy_score(y_test, y_prf_svm_avgLength)))
	print("AdaBoost:avgLength with" + str(number_user*tr_ratio)+":")
	print(classification_report(y_test, y_prf_AdaBoost_avgLength, target_names=target_names))
	print("AdaBoost:avgLength Accuracy:"+str(accuracy_score(y_test, y_prf_AdaBoost_avgLength)))
	print("Random Forest:avgLength with" + str(number_user*tr_ratio)+":")
	print(classification_report(y_test, y_prf_RF_avgLength, target_names=target_names))
	print("Random Forest:avgLength Accuracy:"+str(accuracy_score(y_test, y_prf_RF_avgLength)))
	print("--- %s AvgLength seconds ---" % (time.time() - start_time_avl))
	
	start_time_he = time.time()	
	X_train, X_test, y_train, y_test = train_test_split(X_tweet_Heuristic_feature, Y_tweet_target, train_size=tr_ratio, random_state=0)
	clf_svm_Heuristic=svm.SVC(kernel='linear', C=1)
	clf_svm_Heuristic.fit(X_train,y_train)
	y_prf_svm_Heuristic=clf_svm_Heuristic.predict(X_test)
	clf_AdaBoost_Heuristic=AdaBoostClassifier(n_estimators=100)
	clf_AdaBoost_Heuristic.fit(X_train,y_train)
	y_prf_AdaBoost_Heuristic=clf_AdaBoost_Heuristic.predict(X_test)
	clf_RF_Heuristic = RandomForestClassifier(n_estimators=10)
	clf_RF_Heuristic.fit(X_train,y_train)
	y_prf_RF_Heuristic=clf_RF_Heuristic.predict(X_test)
	target_names = ['human', 'bot']
	print("SVM:Heuristic with" + str(number_user*tr_ratio)+":")
	print(classification_report(y_test, y_prf_svm_Heuristic, target_names=target_names))
	print("SVM:Heuristic with Accuracy:"+str(accuracy_score(y_test, y_prf_svm_Heuristic)))
	print("AdaBoost:Heuristic with" + str(number_user*tr_ratio)+":")
	print(classification_report(y_test, y_prf_AdaBoost_Heuristic, target_names=target_names))
	print("AdaBoost:Heuristic Accuracy:"+str(accuracy_score(y_test, y_prf_AdaBoost_Heuristic)))
	print("Random Forest:Heuristic with" + str(number_user*tr_ratio)+":")
	print(classification_report(y_test, y_prf_RF_Heuristic, target_names=target_names))
	print("Random Forest:Heuristic Accuracy:"+str(accuracy_score(y_test, y_prf_RF_Heuristic)))
	print("--- %s Heuristic seconds ---" % (time.time() - start_time_he))

print("\n---------------------------------Scale up--------------------------------------------------\n")

for tr_ratio in Test_tr_ratio:
	start_time_hy = time.time()
	trNum=number_user*tr_ratio
	print("train data is "+ str(tr_ratio*100)+"% of the total users and the total number is "+str(trNum))
	X_tweet_hybrid_feature=np.concatenate((X_tweet_topic_feature*wt/trNum,X_tweet_ratioURL_feature,X_tweet_avgLength_feature),axis=1)
	#X_tweet_hybrid_feature=np.concatenate((X_tweet_topic_feature,X_tweet_ratioURL_feature,X_tweet_avgLength_feature),axis=1)
	X_train, X_test, y_train, y_test = train_test_split(X_tweet_hybrid_feature, Y_tweet_target, train_size=tr_ratio, random_state=0)
	clf_svm_hy=svm.SVC(kernel='linear', C=1)
	clf_svm_hy.fit(X_train,y_train)
	y_prf_svm_hy=clf_svm_hy.predict(X_test)
	clf_AdaBoost_hy=AdaBoostClassifier(n_estimators=100)
	clf_AdaBoost_hy.fit(X_train,y_train)
	y_prf_AdaBoost_hy=clf_AdaBoost_hy.predict(X_test)
	clf_RF_hy = RandomForestClassifier(n_estimators=10)
	clf_RF_hy.fit(X_train,y_train)
	y_prf_RF_hy=clf_RF_hy.predict(X_test)
	target_names = ['human', 'bot']
	print("SVM:hybrid model with " + str(number_user*tr_ratio)+":")
	print(classification_report(y_test, y_prf_svm_hy, target_names=target_names))
	print("SVM:hybrid model Accuracy:"+str(accuracy_score(y_test, y_prf_svm_hy)))
	print("AdaBoost:hybrid model with " + str(number_user*tr_ratio)+":")
	print(classification_report(y_test, y_prf_AdaBoost_hy, target_names=target_names))
	print("AdaBoost:hybrid model Accuracy:"+str(accuracy_score(y_test, y_prf_AdaBoost_hy)))
	print("Random Forest:hybrid model with " + str(number_user*tr_ratio)+":")
	print(classification_report(y_test, y_prf_RF_hy, target_names=target_names))
	print("Random Forest:hybrid model Accuracy:"+str(accuracy_score(y_test, y_prf_RF_hy)))
	print("--- %s hybrid seconds ---" % (time.time() - start_time_hy))

	start_time_tp = time.time()
	X_train, X_test, y_train, y_test = train_test_split(X_tweet_topic_feature, Y_tweet_target, train_size=tr_ratio, random_state=0)
	clf_svm_tp=svm.SVC(kernel='linear', C=1)
	clf_svm_tp.fit(X_train,y_train)
	y_prf_svm_tp=clf_svm_tp.predict(X_test)
	clf_AdaBoost_tp=AdaBoostClassifier(n_estimators=100)
	clf_AdaBoost_tp.fit(X_train,y_train)
	y_prf_AdaBoost_tp=clf_AdaBoost_tp.predict(X_test)
	clf_RF_tp = RandomForestClassifier(n_estimators=10)
	clf_RF_tp.fit(X_train,y_train)
	y_prf_RF_tp=clf_RF_tp.predict(X_test)
	target_names = ['human', 'bot']
	print("SVM:topic model only with " + str(number_user*tr_ratio)+":")
	print(classification_report(y_test, y_prf_svm_tp, target_names=target_names))
	print("SVM:topic model only Accuracy:"+str(accuracy_score(y_test, y_prf_svm_tp)))
	print("AdaBoost:topic model only with " + str(number_user*tr_ratio)+":")
	print(classification_report(y_test, y_prf_AdaBoost_tp, target_names=target_names))
	print("AdaBoost:topic model only Accuracy:"+str(accuracy_score(y_test, y_prf_AdaBoost_tp)))
	print("Random Forest:topic model only with " + str(number_user*tr_ratio)+":")
	print(classification_report(y_test, y_prf_RF_tp, target_names=target_names))
	print("Random Forest:topic model Accuracy:"+str(accuracy_score(y_test, y_prf_RF_tp)))
	print("--- %s Topic seconds ---" % (time.time() - start_time_tp))
	
	start_time_URL = time.time()
	X_train, X_test, y_train, y_test = train_test_split(X_tweet_ratioURL_feature, Y_tweet_target, train_size=tr_ratio, random_state=0)
	clf_svm_ratioURL=svm.SVC(kernel='linear', C=1)
	clf_svm_ratioURL.fit(X_train,y_train)
	y_prf_svm_ratioURL=clf_svm_ratioURL.predict(X_test)
	clf_AdaBoost_ratioURL=AdaBoostClassifier(n_estimators=100)
	clf_AdaBoost_ratioURL.fit(X_train,y_train)
	y_prf_AdaBoost_ratioURL=clf_AdaBoost_ratioURL.predict(X_test)
	clf_RF_ratioURL = RandomForestClassifier(n_estimators=10)
	clf_RF_ratioURL.fit(X_train,y_train)
	y_prf_RF_ratioURL=clf_RF_ratioURL.predict(X_test)
	target_names = ['human', 'bot']
	print("SVM:ratioURL with " + str(number_user*tr_ratio)+":")
	print(classification_report(y_test, y_prf_svm_ratioURL, target_names=target_names))
	print("SVM:ratioURL Accuracy:"+str(accuracy_score(y_test, y_prf_svm_ratioURL)))
	print("AdaBoost:ratioURL with " + str(number_user*tr_ratio)+":")
	print(classification_report(y_test, y_prf_AdaBoost_ratioURL, target_names=target_names))
	print("AdaBoost:ratioURL Accuracy:"+str(accuracy_score(y_test, y_prf_AdaBoost_ratioURL)))
	print("Random Forest:ratioURL with " + str(number_user*tr_ratio)+":")
	print(classification_report(y_test, y_prf_RF_ratioURL, target_names=target_names))
	print("Random Forest:ratioURL Accuracy:"+str(accuracy_score(y_test, y_prf_RF_ratioURL)))
	print("--- %s URL seconds ---" % (time.time() - start_time_URL))
	
	start_time_avl = time.time()		
	X_train, X_test, y_train, y_test = train_test_split(X_tweet_avgLength_feature, Y_tweet_target, train_size=tr_ratio, random_state=0)
	clf_svm_avgLength=svm.SVC(kernel='linear', C=1)
	clf_svm_avgLength.fit(X_train,y_train)
	y_prf_svm_avgLength=clf_svm_avgLength.predict(X_test)
	clf_AdaBoost_avgLength=AdaBoostClassifier(n_estimators=100)
	clf_AdaBoost_avgLength.fit(X_train,y_train)
	y_prf_AdaBoost_avgLength=clf_AdaBoost_avgLength.predict(X_test)
	clf_RF_avgLength = RandomForestClassifier(n_estimators=10)
	clf_RF_avgLength.fit(X_train,y_train)
	y_prf_RF_avgLength=clf_RF_avgLength.predict(X_test)
	target_names = ['human', 'bot']
	print("SVM:avgLength with" + str(number_user*tr_ratio)+":")
	print(classification_report(y_test, y_prf_svm_avgLength, target_names=target_names))
	print("SVM:avgLength with Accuracy:"+str(accuracy_score(y_test, y_prf_svm_avgLength)))
	print("AdaBoost:avgLength with" + str(number_user*tr_ratio)+":")
	print(classification_report(y_test, y_prf_AdaBoost_avgLength, target_names=target_names))
	print("AdaBoost:avgLength Accuracy:"+str(accuracy_score(y_test, y_prf_AdaBoost_avgLength)))
	print("Random Forest:avgLength with" + str(number_user*tr_ratio)+":")
	print(classification_report(y_test, y_prf_RF_avgLength, target_names=target_names))
	print("Random Forest:avgLength Accuracy:"+str(accuracy_score(y_test, y_prf_RF_avgLength)))
	print("--- %s AvgLength seconds ---" % (time.time() - start_time_avl))
	
	start_time_he = time.time()	
	X_train, X_test, y_train, y_test = train_test_split(X_tweet_Heuristic_feature, Y_tweet_target, train_size=tr_ratio, random_state=0)
	clf_svm_Heuristic=svm.SVC(kernel='linear', C=1)
	clf_svm_Heuristic.fit(X_train,y_train)
	y_prf_svm_Heuristic=clf_svm_Heuristic.predict(X_test)
	clf_AdaBoost_Heuristic=AdaBoostClassifier(n_estimators=100)
	clf_AdaBoost_Heuristic.fit(X_train,y_train)
	y_prf_AdaBoost_Heuristic=clf_AdaBoost_Heuristic.predict(X_test)
	clf_RF_Heuristic = RandomForestClassifier(n_estimators=10)
	clf_RF_Heuristic.fit(X_train,y_train)
	y_prf_RF_Heuristic=clf_RF_Heuristic.predict(X_test)
	target_names = ['human', 'bot']
	print("SVM:Heuristic with" + str(number_user*tr_ratio)+":")
	print(classification_report(y_test, y_prf_svm_Heuristic, target_names=target_names))
	print("SVM:Heuristic with Accuracy:"+str(accuracy_score(y_test, y_prf_svm_Heuristic)))
	print("AdaBoost:Heuristic with" + str(number_user*tr_ratio)+":")
	print(classification_report(y_test, y_prf_AdaBoost_Heuristic, target_names=target_names))
	print("AdaBoost:Heuristic Accuracy:"+str(accuracy_score(y_test, y_prf_AdaBoost_Heuristic)))
	print("Random Forest:Heuristic with" + str(number_user*tr_ratio)+":")
	print(classification_report(y_test, y_prf_RF_Heuristic, target_names=target_names))
	print("Random Forest:Heuristic Accuracy:"+str(accuracy_score(y_test, y_prf_RF_Heuristic)))
	print("--- %s Heuristic seconds ---" % (time.time() - start_time_he))


print("--- %s total seconds ---" % (time.time() - start_time))

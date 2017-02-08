from nltk.tokenize import RegexpTokenizer
from stop_words import get_stop_words
from nltk.stem.porter import PorterStemmer
from gensim import corpora, models
from datetime import datetime
import time
import gensim
import numpy as np
import matplotlib.pyplot as plt


def TweetExtract(filename,LDA_topic_number=5,passNum=20,target=1):
  tokenizer = RegexpTokenizer(r'\w+')
  
  # create English stop words list
  en_stop = get_stop_words('en')
  
  # Create p_stemmer of class PorterStemmer
  p_stemmer = PorterStemmer()
 
  num_lines = sum(1 for line in open(filename))
  LDA_topic_number=5
  count=0
  refID=''
  userTweet=''
  tweetTotalCount=0
  tweetURLCount=0
  reftweetTime=0.0
  tweetTotalTime=0
  doc_set=[]
  h_URL=[]
  h_Length=[]
  h_TweetIn=[]
  d_topic=[]
  d_target=[]
  
  
  with open(filename, "r") as f:
    for line in f:
        line=line.rstrip('\n\r')
        linetoken=line.split("\t")
        count=count+1
        tempID=linetoken[0]
        tempTweet=linetoken[2]
        tempTweetCreateTime=linetoken[3]
        d = datetime.strptime(tempTweetCreateTime, "%Y-%m-%d %H:%M:%S")
        t=time.mktime(d.timetuple())
        tempTweet=tempTweet.decode('UTF-8')
        if (count==1):
          refID=tempID
          reftweetTime=t
          userTweet=tempTweet
          tweetTotalTime=0
          tweetTotalCount=1
          if (tempTweet.find("http://")>-1):
                 tweetURLCount=tweetURLCount+1
          
        else:
          if (tempID==refID):
              tweetTotalTime=tweetTotalTime+abs(t-reftweetTime)
              reftweetTime=t
              userTweet=userTweet+tempTweet
              tweetTotalCount=tweetTotalCount+1
              if (tempTweet.find("http://")>-1):
                 tweetURLCount=tweetURLCount+1
                  
          else:
              doc_set.append(userTweet)
              ratioURL=tweetURLCount/float(tweetTotalCount)
              avgLength=sum(c != ' ' for c in userTweet)/float(tweetTotalCount)
              if (tweetTotalCount-1)>0 :
               avgTweetIn=tweetTotalTime/float(tweetTotalCount-1)
              else:
               avgTweetIn=0
              h_URL.append(ratioURL)
              h_Length.append(avgLength)
              h_TweetIn.append(avgTweetIn)
              reftweetTime=t
              tweetTotalTime=0
              tweetTotalCount=1
              tweetURLCount=0
              if (tempTweet.find("http://")>-1):
                 tweetURLCount=tweetURLCount+1
              userTweet=tempTweet
              refID=tempID
              
  doc_set.append(userTweet)
  h_URL.append(tweetURLCount/float(tweetTotalCount))
  h_Length.append(sum(c != ' ' for c in userTweet)/float(tweetTotalCount))
  if (tweetTotalCount-1)>0 :
   h_TweetIn.append(tweetTotalTime/float(tweetTotalCount-1))
  else:
   h_TweetIn.append(0)
  
  
  # list for tokenized documents in loop
  texts = []
  
  # loop through document list
  for i in doc_set:
      
      # clean and tokenize document string
      raw = i.lower()
      tokens = tokenizer.tokenize(raw)
  
      # remove stop words from tokens
      stopped_tokens = [i for i in tokens if not i in en_stop]
      
      # stem tokens
      stemmed_tokens = [p_stemmer.stem(i) for i in stopped_tokens]
      
      # add tokens to list
      texts.append(stemmed_tokens)
  
  # turn our tokenized documents into a id <-> term dictionary
  dictionary = corpora.Dictionary(texts)
      
  # convert tokenized documents into a document-term matrix
  corpus = [dictionary.doc2bow(text) for text in texts]
  
  # generate LDA model
  ldamodel = gensim.models.ldamodel.LdaModel(corpus, num_topics=LDA_topic_number, id2word = dictionary, passes=passNum)
  #ldamodel.print_topics(LDA_topic_number)
  
  for d in corpus:
   topic_feature=[]
   tempD_topic=ldamodel.get_document_topics(d,0.00001)
   for topic_id, topic_p in tempD_topic:
       topic_feature.append(topic_p)
   d_topic.append(topic_feature)
   d_target.append(target)
  # print('the tweet topic feature for each user count:\n')
  # print(d_topic)
  return TweetUser(h_URL,h_Length,h_TweetIn,d_topic,d_target)
 


class TweetUser(object):
  def __init__(self, ratioURL, avgLength, avgTweetin,docTopic,doctarget):
     self.ratioURL = ratioURL
     self.avgLength = avgLength
     self.avgTweetin = avgTweetin
     self.docTopic=docTopic
     self.docTarget=doctarget


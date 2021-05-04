import pandas as pd
from ast import literal_eval
import nltk
from nltk.corpus import stopwords
import re
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.svm import SVC
import requests
from bs4 import BeautifulSoup
import pickle

def de_repeat(text):    
    pattern = re.compile(r"(.)\1{2,}")
    return pattern.sub(r"\1\1", text)

class MLModel():
    
    clf=pickle.load(open('./modules/model.pkl', 'rb'))
    stop=pickle.load(open('./modules/stopwords.pkl', 'rb'))
    Tv=None
    binarizer=None
    
    @classmethod
    def train(cls):
        df= pd.read_csv("Organised_Data_gt_7.csv")
        
        df['Content_Tags']=df['Content_Tags'].apply(literal_eval)
        
        Train_X = pickle.load(open('./modules/train.pkl', 'rb'))
        cls.binarizer = MultiLabelBinarizer()
        labels = cls.binarizer.fit_transform(df['Content_Tags'])
        
        Y=labels
        cls.Tv = TfidfVectorizer(max_df=0.5,max_features=10000 ,stop_words="english", sublinear_tf=True, norm='l2', ngram_range=(1, 2))
        
        xtrain_tfidf = cls.Tv.fit_transform(Train_X)
        
    @classmethod  
    def test_para(cls,text):
        text = " ".join(x.lower() for x in text.split())
        text=text.replace('[^\w\s]',' ')
        text= " ".join(x for x in text.split() if x not in cls.stop)
        pattern = re.compile(r"(.)\1{2,}")
        text = pattern.sub(r"\1\1", text)
        k=cls.clf.predict(cls.Tv.transform([text]))
        return cls.binarizer.inverse_transform(k)
    
    @classmethod
    def test_link(cls,url):
        page=requests.get(url)
        soup = BeautifulSoup(page.content,'html.parser')
        text = soup.find_all("p")
        each_para=[]
        for j in range(len(text)):
          each_para.append(text[j].get_text())
        text =str(text)
        return cls.test_para(text)





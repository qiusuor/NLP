import json
from gensim.models import TfidfModel
from gensim.corpora import Dictionary
from itertools import islice

def extract_keywords(dct,tfidf,threshold=0.2,topk=5):
    '''
    dct :: Dictionary
    tfidf :: model[doc], [(int, number)]
    threshold :: 提取tftdf值超过它的词
    topk :: 最多提取个数
    '''
    tfidf = sorted(tfidf,key=lambda x:x[1],reverse=True)
    return list(islice([dct[w] for w,score in tfidf if score > threshold],topk))

if __name__ == '__main__':
    with open('finance_news_test.json',encoding='utf-8') as f:
        data = json.load(f)
        data = [doc.split() for doc in data]

    dct = Dictionary.load('news.dict')
    corpus = [dct.doc2bow(doc) for doc in data]
    model = TfidfModel.load('news_tfidf.model')

    for text,doc in zip(data,corpus):
        print (' '.join(text).replace(' ',''))
        print (extract_keywords(dct,model[doc]))
        print ()


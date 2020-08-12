import json
from gensim.models import TfidfModel
from gensim.corpora import Dictionary
from gensim import matutils

def vectorize(docs,vocab_size):
    '''
    docs :: iterable of iterable of (int, number)
    vocab_size :: 词表大小
    '''
    return matutils.corpus2dense(docs,vocab_size)


if __name__ == '__main__':
    with open('finance_news_test.json', encoding='utf-8') as f:
        data = json.load(f)
        data = [doc.split() for doc in data]

    dct = Dictionary.load('news.dict')
    corpus = [dct.doc2bow(doc) for doc in data]
    model = TfidfModel.load('news_tfidf.model')
    vocab_size = len(dct.token2id)

    for doc in corpus:
        # print(model[doc],len(model[doc]),len(vectorize([model[doc]],vocab_size)),len(vectorize([model[doc]],vocab_size)[0]))
        # break
        print (vectorize([model[doc]],vocab_size))

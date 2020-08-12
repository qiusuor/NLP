from glob import glob
import jieba
import json
from gensim.models import TfidfModel
from gensim.corpora import Dictionary


data = []
for filename in glob(r'C:\Users\75043\PycharmProjects\NLP\TF-IDF\corups\THUCNews\体育\*.txt')[:1100]:
    with open(filename, encoding='utf-8') as f:
        # print(filename)
        text = ' '.join(jieba.cut(f.read().replace('\n','')))
        data.append(text)

with open('finance_news_train.json','w', encoding='utf-8') as f:
    json.dump(data[:1000] ,f,indent=2,ensure_ascii=False)

with open('finance_news_test.json','w', encoding='utf-8') as f:
    json.dump(data[1000:] ,f,indent=2,ensure_ascii=False)


with open('finance_news_train.json', encoding='utf-8') as f:
    data = json.load(f)
    data = [doc.split() for doc in data] # the parameter of Dictionaryis iterable of iterable of str

dct = Dictionary(data)
corpus = [dct.doc2bow(doc) for doc in data] # convert corpus to BoW format
# print(corpus[0])
model = TfidfModel(corpus)  # fit model
dct.save('news.dict')
# print(dct[0],dct[1],len(dct),dct)
model.save('news_tfidf.model')



## TF-IDF



TF(term frequency): 某个词在某篇文章中出现的频率

IDF(inverse document frequency):逆文档频率，衡量词的重要性

$TF(t)=\frac{词t在某篇文章中出现的总次数}{某篇文章的总词数}$

$IDF(t)=log \frac{总文档数}{出现了该词的文档数+1}$

$TF-IDF=TF*IDF$


主要用途：

1. 生成文本向量
2. 提取关键词

基于jieba和Gensim实现了TF-IDF，并通过TF-IDF进行关键词提取和doc2vec转换。

下载[数据集](http://thuctc.thunlp.org/#%E8%8E%B7%E5%8F%96%E9%93%BE%E6%8E%A5 )并解压至corups目录

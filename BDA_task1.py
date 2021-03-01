from pyspark.sql import SQLContext, Row
from pyspark.sql import SparkSession
from pyspark import SparkContext
import re
import pyspark
import numpy as np
from __future__ import division
conf = pyspark.SparkConf().setAppName('Untitled5').setMaster('local')
sc = pyspark.SparkContext(conf=conf)
spark = SparkSession(sc)
text = sc.wholeTextFiles("../books/",8)
text=text.map(lambda a,b:Row(title =a.replace('../books/',''), text=b))
text=text.toDF(["doc","text"])
N = text.count()
path_sw="../stopwords_en.txt"
with io.open(path_sw, 'r', encoding='utf8') as f:
     stopwords=(f.read()).split()
        
def tokenize(s):
    return re.split("\\W+", s.lower())
def rm_stop_words(s):
    return   s.map(lambda x : x not in stopwords) 
def total_number_dw(s):
    return len(Counter(s)) 
#split text into words
tokenized_text = text.map(lambda (title,content): (title, tokenize(content)) )
#removing stopwords
tokenized_text=tokenized_text.map(lambda (title,content): (title, rm_stop_words(content)) )
#For each book  the total number of distinct words 
total_distinct_words=tokenized_text.map(lambda (title,content): (title, total_number_dw(content)) )\
                            .filter(lambda title,count: count)
#count word frequency in each document.
term_frequency = tokenized_text.flatMapValues(lambda x: x).countByValue()

#count how many documents a word appears in.
document_frequency = tokenized_text.flatMapValues(lambda x: x).distinct()\
                        .filter(lambda x: x[1] != '')\
                        .map(lambda (title,word): (word,title)).countByKey()

def tf_idf(N, tf, df):
    result = []
    for key, value in tf.items():
        doc = key[0]
        term = key[1]
        df = document_frequency[term]
        if (df>0):
            tf_idf = float(value)*np.log(number_of_docs/df)
        
        result.append({"doc":doc, "term":term, "score":tf_idf})
    return result

tf_idf_output = tf_idf(N, term_frequency, document_frequency)

tfidf_RDD = sc.parallelize(tf_idf_output).map(lambda x: (x['term'],(x['doc'],x['score']) )) # the corpus with tfidf scores

def search(w):
    word= sc.parallelize(w).map(lambda x: (x,1) ).collectAsMap()
    bcTokens = sc.broadcast(word)
    
    joined_tfidf = tfidf_RDD.map(lambda k,v: (k,bcTokens.value.get(k,'-'),v) ).filter(lambda a,b,c: b != '-' )
    
  
     return joined_tfidf
input_word = input("Enter  word to search in a booklist: ")
result= search(str(input_word))
sorted_result = result.sortBy(lambda x: x.score)
df = sorted_result.toDF()
df.select("doc").show()

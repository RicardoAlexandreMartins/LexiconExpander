import gensim
import logging
import os
import csv
import numpy as np
import pymysql.cursors
import pandas as pd
import re, os
import copy
import numpy as np
from sklearn.decomposition import PCA
import argparse
import chardet


from nltk.corpus import stopwords 
from nltk.tokenize import word_tokenize


import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.style.use("ggplot")

from sklearn.manifold import TSNE


username = "database username"
passw= "database password"
server = "database server ip"

class WordLexicon:

    def __init__(self, word, positive, negative, anger, anticipation, disgust, fear, joy, sadness, surprise, trust):
        self.word = word
        self.positive = positive
        self.negative = negative
        self.anger = anger
        self.anticipation = anticipation
        self.disgust = disgust
        self.fear = fear
        self.joy = joy
        self.sadness = sadness
        self.surprise = surprise
        self.trust = trust
		
    def copy(self):
        return WordLexicon(self.word, self.positive, self.negative, self.anger, self.anticipation, self.disgust, self.fear, self.joy, self.sadness, self.surprise, self.trust)

logging.basicConfig(
    format='%(asctime)s : %(levelname)s : %(message)s',
    level=logging.INFO)


def select(schema, tablename, condicaoWhere, username, passw, server):
    try:

        cnn = pymysql.connect(user=username, password=passwd, host=server, db=schema, cursorclass=pymysql.cursors.DictCursor)
   	
        cursor = cnn.cursor()
        sql = "SELECT texto from {} {}".format(tablename, condicaoWhere)
        print("Carregando informações do Banco de Dados")
        cursor.execute(sql)
        data=cursor.fetchall()
        cursor.close()
        cnn.commit()
        cnn.close()
		
        return pd.DataFrame(data)
		
    except UnicodeEncodeError as err:
        pass
    except pymysql.err.IntegrityError:
        exit(1)
    except Exception as err:
        print('Function select - Got error {!r}, errno is {}'.format(err, err.args[0]))


def read_input(schema, tabela, condicao, username, passw, server):
    #logging.info("reading file {0}...this may take a while".format(input_file))
    data = select(schema, tabela, condicao, username, passw, server)
	
    for row in data['texto']:
        yield gensim.utils.simple_preprocess(clean_str(row))

def loadLexicon(path):
    
    enc = find_encoding(path)
    df = pd.read_csv(path, sep=',', encoding = enc)
	
    listLexicon = []
  
    for index, row in df.iterrows():
        print('Reading lexicon word ' + str(index) + ": " + row['Word'])
        wordLexicon = WordLexicon(row['Word'],row['Positive'],row['Negative'],row['Anger'],row['Anticipation'],row['Disgust'],row['Fear'],row['Joy'],row['Sadness'],row['Surprise'],row['Trust'])
        listLexicon.append(wordLexicon)
 
    
    return listLexicon		

	
def loadThesaurus(path):
    listThesaurus = []
    
    print ("Loading thesaurus")
    #enc = find_encoding(path)
    #data = pd.read_csv(path, encoding=enc)
    #print(data)
	
    with open(path, 'rb') as f: 
        file = f.readlines()
        for line in file:
            l = line.decode('utf-8')		
            l = re.sub(r'\r\n', '', l)
            print(line) 
	
def clean_str(string):
    stop_words = set(stopwords.words('spanish')) 
    word_tokens = word_tokenize(string) 
    filtered_sentence = [w for w in word_tokens if not w in stop_words] 
    filtered_sentence = [] 
  
    for w in word_tokens: 
        if w not in stop_words: 
            filtered_sentence.append(" " + w) 
    
    string = ''.join(filtered_sentence)
	
    # Remove os emojis
    string = re.sub('(\<.*?\>)', ' ', string, flags=re.UNICODE)
    # Remove os imagens
    string = re.sub('^pic.twitter.*$', ' ', string, flags=re.UNICODE)


    #Remove links
    string = re.sub(r'\w+:\/{2}[\d\w-]+(\.[\d\w-]+)*(?:(?:\/[^\s/]*))*', ' ', string, flags=re.MULTILINE)
    #Remove mentions e hashtags
    string = re.sub(r'(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)', ' ', string, flags=re.MULTILINE)

	
    string = re.sub(r'(\<.*?\>)', ' ', string, flags=re.UNICODE)

    return string.strip().lower()
	
def options():
    ap = argparse.ArgumentParser(prog="LexiconExpander.py",
            usage="python3 %(prog)s [options]",
            description="Ferramenta para gerar lexicos expandidos")
    ap.add_argument("-s", "--schema", help="Schema da tabela")
    ap.add_argument("-t", "--tablename", help="Nome da tabela")
    ap.add_argument("-m", "--modelo", help="Nome do modelo")
    ap.add_argument("-c", "--condicao", help="Clausula where do banco de dados")
    ap.add_argument("-e", "--epochs", help="Epochs a serem utilizadas para treinamento")
    ap.add_argument("-l", "--lexicon", help="Caminho do lexico a ser expandido")
    ap.add_argument("-th", "--thesaurus", help="Caminho do thesaurus a ser expandido")
    ap.add_argument("-f", "--file", help="Arquivo com exemplos a serem carregados")
	
    args = ap.parse_args()
    return args

def recursiveSearch(model, newLexicon, original, word, level):
    # Percorre o lexico
    try:
        print("Level " + str(level) + " - Original word: " + original.word)
        words = [w.word for w in newLexicon]
			
        similarities = model.wv.most_similar(positive=word) 

        for k, v in similarities:
            if (k not in words) and v > 0.7 and k.strip() != l.word.strip():
                newWord = original.copy()
                newWord.word = k
                newLexicon.append(newWord)
                    
                return recursiveSearch(model, newLexicon, original, k, level+1)
    except KeyError:
        pass
			
    return newLexicon

def find_encoding(fname):
    r_file = open(fname, 'rb').read()
    result = chardet.detect(r_file)
    charenc = result['encoding']
    print("Encoding is:" + charenc)
    return charenc
	
def loadFile(path):
    enc = find_encoding(path)
    df = pd.read_csv(path, names=["Texto"], sep='\n', encoding = enc)
    
    for row in df['Texto']:
        yield gensim.utils.simple_preprocess(clean_str(row))


def similaritySearch(thesaurus, newLexicon):
    pass    
	
if __name__ == '__main__':

    args = options() 
    
    #loadThesaurus(args.thesaurus)
  

    
    if args.epochs is None:
        args.epochs = 100

#    if args.file is None or args.schema is None or args.tablename is None or args.modelo is None or args.lexicon is None:
#        print("Arguments missing")
#        exit(1)
    
    lexicon = loadLexicon(args.lexicon)	
    #synonyms = loadSynonyms(args.thesaurus)

    if args.file :
        documents = list(loadFile(args.file))
    else:
        documents = list(read_input(args.schema, args.tablename, args.condicao, username, passw, server))

    logging.info("Done reading data file")


    # build vocabulary and train model
    model = gensim.models.Word2Vec(documents, size=50, window=5, min_count=10, workers=12)
    model.train(documents, total_examples=len(documents), epochs=int(args.epochs))

 
    newLexicon = lexicon.copy()
    
	# Percorre o lexico
    for l in lexicon: 
        newLexicon = recursiveSearch(model, newLexicon, l, l.word, 1)
		
    # Analisa as similaridades
    #newLexicon = similaritySearch(args.thesaurus, newLexicon)
	
    # Salva o novo lexico
    variables = ['Word','Anger','Anticipation','Disgust','Fear','Joy','Sadness','Surprise','Trust', 'Positive', 'Negative']
    df = pd.DataFrame([[getattr(i,j) for j in variables] for i in newLexicon], columns = variables)
    
    df.to_csv(args.modelo, sep=',', encoding='utf-8', index=False)
import os
import sys
import time
import numpy as np
from text import *
from utils import open_data
from tqdm import tqdm
from nltk.corpus import stopwords
from flask import Flask, render_template, request

app = Flask(__name__)

class BayesClassifier:

    def __init__(self, file):
        self.pos = 0
        self.neg = 0
        self.set = set()
        self.dict = dict()
        self.lines = self.open_file(file)
        self.stopwords = stopwords.words('english')
    
    def open_file(self, file):
        if os.path.exists(file):
            print("Load data from " + file)
            data = open_data(file)
            return data.readlines()
        else:
            print(file + " is not exist!")
            sys.exit();
            
    def parse_file(self):
        print("File parsing started!")
        start = time.time()
        self.preprocess()
        self.dict = {key: {'+': 0, '-': 0} for key in self.set}
        self.load_words()
        self.pos = self.count('+')
        self.neg = self.count('-')
        end = time.time()
        print("Preprocess: ", end - start)
            
    def preprocess(self):
        for line in self.lines:
            self.set.update(self.get_data(line))

    def load_words(self):
        for line in tqdm(self.lines):
            list_words = self.get_data(line)
            appearance = np.array([self.dict[word][line[0]] for word in list_words if word in self.dict])
            appearance = appearance + 1
            for i in range(len(appearance)):
                self.dict[list_words[i]][line[0]] = appearance[i]
                
    def count_unknown(self, words):
        return len([word for word in words if word not in self.dict])

    def count_distinct(self, sign):
        return np.sum([(lambda x: 1 if x > 0 else 0)(count) for count in self.list_all(sign)])

    def count(self, sign):
        return np.sum([count for count in self.list_all(sign)])

    def list_all(self, sign):
        return [self.dict[word][sign] for word in self.dict]

    def process(self, sign, words):
        unknown = self.count_unknown(words)
        result = [(self.pos if sign == '+' else self.neg) / (self.pos + self.neg)]
        result = result + [((self.dict[word][sign] if word in self.dict else 0) + 1) / ((self.pos if sign == '+' else self.neg) + self.count_distinct(sign) + unknown) for word in words]
        return np.prod(result)
        
    def get_data(self, data):
        test_data = [x for x in words(data) if x not in self.stopwords]
        return test_data
       
bs = BayesClassifier('/opt/bayes_classifier_numpy/data_base')

@app.route('/')
def index():
    bs.parse_file()
    return render_template('index.html')

@app.route('/classify', methods=['POST'])
def classify():
    data = request.form['sentence']
    start = time.time()
    test_data = bs.get_data(data)
    Ppos = bs.process('+', test_data)
    Pneg = bs.process('-', test_data)
    result = '+ : ' + data if Ppos > Pneg else '- : ' + data
    end = time.time()
    print("Process: ", end - start)
    return render_template('result.html', result=result)

if __name__ == '__main__':
   app.run(host="0.0.0.0", port="5001")

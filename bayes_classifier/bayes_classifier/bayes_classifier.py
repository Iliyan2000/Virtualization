import os
import sys
import time
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
        self.dict = {}
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
        for line in tqdm(self.lines):
            words_line = [x for x in words(line) if x not in self.stopwords]
            self.add_word(line[0], words_line)
        end = time.time()
        print("Preprocess: ", end - start)
            
    def add_word(self, sign, words):
        for word in words:
            if sign == '+':
                self.pos += 1
                if word in self.dict:
                    self.dict[word]['+'] += 1
                else:
                    self.dict[word] = {'+': 1, '-': 0 }
            elif sign == '-':
                self.neg += 1
                if word in self.dict:
                    self.dict[word]['-'] += 1
                else:
                    self.dict[word] = {'+': 0, '-': 1 }
            else:
                print("Unrecognized symbol " + sign)
                print("Program will terminate!")
                sys.exit(1)
                
    def count_unknown(self, words):
        result = 0
        for word in words:
            if word not in self.dict:
                result += 1
        return result

    def count_distinct(self, sign):
        result = 0
        for word in self.dict:
            if self.dict[word][sign] != 0:
                result += 1
        return result
        
    def process(self, sign, words):
        result = (self.pos if sign == '+' else self.neg) / (self.pos + self.neg)
        unknown = self.count_unknown(words)
        for word in words:
            result *= ((self.dict[word][sign] if word in self.dict else 0) + 1) / ((self.pos if sign == '+' else self.neg) + self.count_distinct(sign) + unknown)
        return result
        
    def get_data(self, data):
        test_data = [x for x in words(data) if x not in self.stopwords]
        return test_data
       
bs = BayesClassifier('/opt/bayes_classifier/data_base')

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
    app.run(host="0.0.0.0", port="5000")

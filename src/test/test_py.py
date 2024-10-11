

def test_reversed():
    for i in range(3):
        print(i, end=", ")
    print()
    for i in reversed(range(3)):
        print(i, end=", ")
    print()

def test_tokenizing():
    text = "Tokenizing is a core task of NLP"
    tokenized_text = list(text)
    print(tokenized_text)
    s = sorted(set(tokenized_text))
    print(s)
    for idx, ch in  enumerate(s):
        print(idx, ':', ch, end=", ")
    print()

import copy
def test_deep_copy():
    original_list = [[1, 2, 3], [4, 5, 6]]
    copied_list = copy.deepcopy(original_list)
    original_list[0][0] = 100
    print(original_list)
    print(copied_list)

def test_zip():
    x, y = zip(['a', 1], ['b', 2], ['c', 3])
    print('x: ', x)
    print('y: ', y)

    s = [['a', 1], ['b', 2], ['c', 3]]
    x, y = zip(*s)
    print('x: ', x)
    print('y: ', y)

    ll = ('a', 'b', 'c', 'd')
    xl = ('1', '2', '3')

    for l, x in zip(ll, xl):
        print('l: ', l, ', x: ', x)


import nltk
from nltk.tokenize import word_tokenize

def test_data_preprocessing():
    d0 = ['나는 사과를 좋아해', '나는 바나나를 좋아해', '나는 사과를 싫어해', '나는 바나나를 싫어해']
    print(type(d0), d0)

    d1 = []
    for s in d0:
        tokens = word_tokenize(s)
        d1.append(tokens)
    print(d1)

    dic = {
        '나는': [0.1, 0.2, 0.9],
        '사과를': [0.3, 0.5, 0.1],
        '바나나를': [0.3, 0.5, 0.2],
        '좋아해': [0.7, 0.6, 0.5],
        '싫어해': [0.5, 0.6, 0.7]
    }

    for i in range(len(d1)):
        for j in range(len(d1[i])):
            d1[i][j] = dic[d1[i][j]]
    print(d1)

result1 = 0
result2 = 0

def add1(num):
    global result1
    result1 += num
    return result1

def add2(num):
    global result2
    result2 += num
    return result2

def test_global():
    print(add1(3))
    print(add1(4))
    print(add2(3))
    print(add2(7))

class Calculator:
    def __init__(self):
        self.result = 0
    
    def add(self, num):
        self.result += num
        return self.result

def test_class():
    cal1 = Calculator()
    cal2 = Calculator()

    print(cal1.add(3))
    print(cal1.add(4))
    print(cal2.add(3))
    print(cal2.add(7))


def test_map():
    data = [
        ['1000', 'Steve', 90.72], 
        ['1001', 'James', 78.09], 
        ['1002', 'Doyeon', 98.43], 
        ['1003', 'Jane', 64.19], 
        ['1004', 'Pilwoong', 81.30],
        ['1005', 'Tony', 99.14],
    ]
    print(data)
    
    def append_hello(x):
        x.append('Hello')
        return x

    #data = list(map(append_hello, data))
    data2 = list(map(lambda x: x + ['Hello'], data))
    print(data)
    print(data2)

import os
def test_fs():
    print('test_fs')
    for path, dirs, files in os.walk('./src/transformer'):
        # All the directory path under ./      
        print('path: ', path)
        # Sub directories under the 'path'
        print('dirs: ', dirs)
        # Files under the 'path'
        print('files: ', files)
        # path + file concatenation
        for file in files:
            print(os.path.join(path, file))        

#test_reversed()
#test_tokenizing()
#test_deep_copy()
test_zip()
#test_data_preprocessing()
#test_global()
#test_class()
#test_map()
#test_fs()
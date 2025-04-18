import token
import nltk
import numpy as np
#nltk.download('punkt')
from nltk.stem.porter import PorterStemmer
stemmer = PorterStemmer()

def tokenize(sentence):
    return nltk.word_tokenize(sentence)

def Stemming(word):
    return stemmer.stem(word.lower())

def bag_of_words(tokenized_sentence, all_words):
    ''' 
    sentence = ["hello", "how", "are", "you"]
    words = ["hello", "hey", "hi", "good day", "greetings", "what's up?", "howdy", "yo", "hi there", "hello there", "hey bot"]
    bag = [     1   ,   0  ,  0  ,     0     ,      0     ,      0      ,    0   ,  0  ,     0     ,       0      ,     0    ]
    '''
    tokenized_sentence = [Stemming(w) for w in tokenized_sentence]
    bag = np.zeros(len(all_words), dtype=np.float32)
    for idx, w, in enumerate(all_words):
        if w in tokenized_sentence:
            bag[idx] = 1.0            
    return bag

sentence = ["hello", "hey", "good day"]
words = ["hello", "hey", "hi", "good day", "greetings", "what's up?", "howdy", "yo", "hi there", "hello there", "hey bot"]
bag = bag_of_words(sentence, words)
print(bag)


'''
a = "Of course! What do you need help with?"
print(a)
a = tokenize(a)
print(a)

words = ["organize", "organizes", "organizing"]
stemmer_words = [Stemming(w) for w in words]
print(stemmer_words)
'''
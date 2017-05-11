from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import pickle
import os,csv
from keras.preprocessing import sequence
import nltk.translate.bleu_score
import nltk.tokenize as tk

#some parameters
MAXLEN = 16
VOCABSIZE = 4000



def reader(csvDir = 'dataset/coco_train_all.csv',imageDir = 'dataset/coco_train_all',
               maxObjects = 20, train_split_rate = 0.8):
    csvItr = csv.reader(open(csvDir,'r'))
    next(csvItr)
    imgList = []
    questionList = []
    cnt = 0
    for line in csvItr:
        try:
            if cnt == maxObjects: break
            im = Image.open(os.path.join(imageDir,'{}.jpg').format(line[0]))
            im = im.resize((224,224))
            im = im.convert('RGB')
            im = np.array(im)
            plt.imshow(im)
            plt.show()
            im = im.transpose((2,0,1))
            imgList.append(np.array(im))
            questions = line[2].split('---')
            questionList.append(questions)
            #print(np.array(im).shape)
            cnt = cnt + 1
        except FileNotFoundError:
            print('File {} open error'.format(line[0]))

    print(np.array(imgList).shape)
    return np.array(imgList),np.array(questionList),train_split_rate

def getPartialAndNext(imgarr,questionarr):
    assert(len(imgarr) == len(questionarr))

    #each img repeats 5 times, for there a 5 questions for each img
    #each quetions begins with "<BGN>" and "<EOL>"

    partial_x = []
    vocab_set = set()
    word2idx = {}
    idx2word = []
    next_word = []
    question_words_arr = []

    for questionlist in questionarr:
        for question in questionlist:
            question_words = [i.lower() for i in str.split(question,' ')]
            if question_words[-1].endswith('?'):
                s = question_words[-1]
                question_words.pop()
                question_words.append(s[:-1])
                question_words.append('?')
            list.insert(question_words,0,'<BGN>')
            list.append(question_words,'<EOL>')
            question_words_arr.append(question_words)
            for word in question_words:
                vocab_set.add(word)

    assert(len(vocab_set) < VOCABSIZE-2)

    idx2word.append('<PAD>') #<PAD> has the index 0
    idx2word.append('<UNK>') #<UNK> has the index 1
    for word in vocab_set:
        idx2word.append(word)
    for i,word in enumerate(idx2word):
        word2idx[word] = i


    #generate partial_question_x
    img_repeat = []
    img_no = -1
    for i,question_words in enumerate(question_words_arr):
        i = i % 5
        if i == 0:img_no += 1
        partial_caption_len = 0
        l = []
        while len(question_words) < MAXLEN:
            question_words.insert(0,'<PAD>')
        while partial_caption_len < MAXLEN:
            if question_words[partial_caption_len] != '<PAD>':
                partial_x.append([i for i in l])
                l.append(word2idx[question_words[partial_caption_len]])
                next_word_l = [0] * VOCABSIZE
                next_word_l[word2idx[question_words[partial_caption_len]]] = 1
                next_word.append(next_word_l)
                img_repeat.append(imgarr[img_no])
            partial_caption_len += 1

    partial_x = sequence.pad_sequences(partial_x, maxlen=MAXLEN)

    with open('idx2word.pickle','wb') as f:
        pickle.dump(idx2word,f)
    with open('word2idx.pickle','wb') as f:
        pickle.dump(word2idx,f)

    return np.array(img_repeat), np.array(partial_x),np.array(next_word),idx2word,word2idx




























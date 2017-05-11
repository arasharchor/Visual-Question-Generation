from keras.models import *
import h5py
import pickle
import FileReader
from keras.layers import *
from Model import pop
from nltk.translate.bleu_score import sentence_bleu

TESTCASES = 100
vocab_size = 4000
maxlen = 16

def max_prob_word(l):
    tup_l = [i for i in enumerate(l)]
    return max(tup_l,key = lambda x:x[1])[0]

def load_model(json_path = 'model_0129.json',weight_path = 'model_0129.h5'):
    with open(json_path,'r') as f:
        model = model_from_json(f.read())
    model.load_weights(weight_path)
    return model



def main():
    model = load_model()
    with open('word2idx.pickle','rb') as f:
        word2idx = pickle.load(f)
    with open('idx2word.pickle','rb') as f:
        idx2word = pickle.load(f)
    model.compile(loss='categorical_crossentropy', optimizer='rmsprop')

    imgarr_x,quesarr = FileReader.reader('dataset/coco_subset_of_train.csv','dataset/coco_train_all')
    maxlen = FileReader.MAXLEN

    question_words_arr_x = []
    for question in quesarr:
        question_words = [i.lower() for i in str.split(question, ' ')]
        if question_words[-1].endswith('?'):
            s = question_words[-1]
            question_words.pop()
            question_words.append(s[:-1])
            question_words.append('?')
        question_words_arr_x.append(question_words)

    logf = open('eval0218.csv','w')
    avg_bleu = 0
    for i in range(len(imgarr_x)):
        gen_seq = [word2idx['<PAD>']] * maxlen
        gen_seq[-1] = word2idx['<BGN>']
        for j in range(maxlen):
            prob_arr = model.predict([np.array([imgarr_x[i]]),np.array([gen_seq])])
            gen_word_idx = max_prob_word(prob_arr[0])
            gen_seq[:-1] = gen_seq[1:]
            gen_seq[-1] = gen_word_idx
            if gen_word_idx == word2idx['<EOL>']:break
        gen_ques = [idx2word[i] for i in gen_seq if idx2word[i] not in ['<PAD>','<BGN>','<EOL>']]
        gen_ques.pop(0)
        gen_ques.pop()
        #bleu
        bleu_score = sentence_bleu(references = question_words_arr_x[5*i:5*i+5],hypothesis = gen_ques)
        avg_bleu += bleu_score
        #log
        logf.write('{},{}\n'.format(gen_ques.join(' '),bleu_score))

    avg_bleu /= len(imgarr_x)
    logf.write('avg+bleu: '+str(avg_bleu))
    logf.close()

if __name__ == '__main__':
    main()

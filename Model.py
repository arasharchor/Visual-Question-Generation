import FileReader
import pickle
from keras.models import *
from keras.layers import *
from keras import backend as K
import h5py

K.set_image_dim_ordering('th')

MAXLEN = FileReader.MAXLEN
VOCABSIZE = FileReader.VOCABSIZE


def pop(model):
    '''Removes a layer instance on top of the layer stack.
    This code is thanks to @joelthchao https://github.com/fchollet/keras/issues/2371#issuecomment-211734276
    '''
    if not model.outputs:
        raise Exception('Sequential model cannot be popped: model is empty.')
    else:
        model.layers.pop()
        if not model.layers:
            model.outputs = []
            model.inbound_nodes = []
            model.outbound_nodes = []
        else:
            model.layers[-1].outbound_nodes = []
            model.outputs = [model.layers[-1].output]
            print(1)
        model.built = False

    return model


def model_train(images, partial_captions, next_words, vocab_size,
                max_question_len, path_vgg16_weight=None, do_fitting = True):
    img_model = Sequential()

    img_model.add(ZeroPadding2D((1, 1), input_shape=(3, 224, 224), trainable=False))
    img_model.add(Convolution2D(64, 3, 3, activation='relu', trainable=False))
    img_model.add(ZeroPadding2D((1, 1), trainable=False))
    img_model.add(Convolution2D(64, 3, 3, activation='relu', trainable=False))
    img_model.add(MaxPooling2D((2, 2), strides=(2, 2), trainable=False))

    img_model.add(ZeroPadding2D((1, 1), trainable=False))
    img_model.add(Convolution2D(128, 3, 3, activation='relu', trainable=False))
    img_model.add(ZeroPadding2D((1, 1), trainable=False))
    img_model.add(Convolution2D(128, 3, 3, activation='relu', trainable=False))
    img_model.add(MaxPooling2D((2, 2), strides=(2, 2), trainable=False))

    img_model.add(ZeroPadding2D((1, 1), trainable=False))
    img_model.add(Convolution2D(256, 3, 3, activation='relu', trainable=False))
    img_model.add(ZeroPadding2D((1, 1), trainable=False))
    img_model.add(Convolution2D(256, 3, 3, activation='relu', trainable=False))
    img_model.add(ZeroPadding2D((1, 1), trainable=False))
    img_model.add(Convolution2D(256, 3, 3, activation='relu', trainable=False))
    img_model.add(MaxPooling2D((2, 2), strides=(2, 2), trainable=False))

    img_model.add(ZeroPadding2D((1, 1), trainable=False))
    img_model.add(Convolution2D(512, 3, 3, activation='relu', trainable=False))
    img_model.add(ZeroPadding2D((1, 1), trainable=False))
    img_model.add(Convolution2D(512, 3, 3, activation='relu', trainable=False))
    img_model.add(ZeroPadding2D((1, 1), trainable=False))
    img_model.add(Convolution2D(512, 3, 3, activation='relu', trainable=False))
    img_model.add(MaxPooling2D((2, 2), strides=(2, 2), trainable=False))

    img_model.add(ZeroPadding2D((1, 1), trainable=False))
    img_model.add(Convolution2D(512, 3, 3, activation='relu', trainable=False))
    img_model.add(ZeroPadding2D((1, 1), trainable=False))
    img_model.add(Convolution2D(512, 3, 3, activation='relu', trainable=False))
    img_model.add(ZeroPadding2D((1, 1), trainable=False))
    img_model.add(Convolution2D(512, 3, 3, activation='relu', trainable=False))
    img_model.add(MaxPooling2D((2, 2), strides=(2, 2), trainable=False))

    img_model.add(Flatten(trainable=False))
    img_model.add(Dense(4096, activation='relu',trainable=False))
    img_model.add(Dropout(0.5,trainable=False))
    img_model.add(Dense(4096, activation='relu',trainable=False))

    f = h5py.File(path_vgg16_weight)
    for k in range(f.attrs['nb_layers']):
        if k >= len(img_model.layers):
            # we don't look at the last (fully-connected) layers in the savefile
            break
        g = f['layer_{}'.format(k)]
        weights = [g['param_{}'.format(p)] for p in range(g.attrs['nb_params'])]
        img_model.layers[k].set_weights(weights)
    f.close()

    language_model = Sequential()
    language_model.add(Embedding(vocab_size, 128, input_length=max_question_len,mask_zero=True))
    language_model.add(LSTM(output_dim=256,return_sequences=False))
    language_model.add(Dropout(0.5))
    img_model.add(Dense(output_dim=256,activation='relu'))
    model = Sequential()

    print('Processing Merge')
    model.add(Merge([img_model, language_model], mode='sum'))
    model.add(Dense(vocab_size))
    model.add(Activation('softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam')

    for layers in img_model.layers:
        print(layers.name, ': ' ,layers.input_shape)
    for layers in language_model.layers:
        print(layers.name ,': ',layers.input_shape)
    for layers in model.layers:
        print(layers.name, ': ', layers.input_shape)

    def model_fitting():
        print('Processing Fitting')
        print(images.shape)
        print(partial_captions.shape)

        model.fit([images, partial_captions], next_words,
                  validation_split=0.1, batch_size=64, nb_epoch=5)
        print('Model training complete')

    if do_fitting == True:
        model_fitting()

    with open('model_0129.json','w') as f:
        f.write(model.to_json())
    model.save_weights('model_0129.h5',overwrite=True)
    return model

def test_bleu(generated_questions,references):
    from nltk.translate.bleu_score import sentence_bleu
    bleu_sum = 0
    for i in range(len(generated_questions)):
        hypo = generated_questions[i]
        ref = references[5*i:5*i+5]
        bleu_sum += sentence_bleu(ref,hypo)
    bleu_avg = bleu_sum / len(generated_questions)
    return bleu_avg

def main():
    imgarr, seqarr, split_rate = FileReader.reader()
    img_repeat, partial_x, next_word, idx2word, word2idx = FileReader.getPartialAndNext(imgarr, seqarr)

    print("shape of img,seq,nxt:", img_repeat.shape, partial_x.shape, next_word.shape)

    model = model_train(img_repeat, partial_x, next_word,
                        FileReader.VOCABSIZE, FileReader.MAXLEN, 'vgg16_weights.h5',do_fitting=True)

    result = model.predict([img_repeat[-100:], partial_x[-100:]])
    with open('log.txt', 'w') as f:
        cnt = 0
        for line in result:
            for item in line:
                f.write(str(item) + ' ')
            f.write('\n')
            cnt += 1
        for line in result:
            idx_word = [i for i in enumerate(line)]
            idx_word.sort(key=lambda x: x[1], reverse=True)
            f.write('{}:{},{}:{},{}:{}\n'.format(idx2word[idx_word[0][0]], idx_word[0][1],
                                                 idx2word[idx_word[1][0]], idx_word[1][1],
                                                 idx2word[idx_word[2][0]], idx_word[2][1]))

    f.close()

if __name__ == '__main__':
    main()

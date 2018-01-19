import numpy as np
import keras.backend as K
import gensim
import keras
from keras.models import Model, load_model
from keras.layers import Input, LSTM, Dense, GRU, Embedding, Bidirectional, BatchNormalization, TimeDistributed
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import one_hot
from keras.utils import to_categorical
from keras.callbacks import History ,ModelCheckpoint, EarlyStopping
from keras.layers.merge import add, dot, concatenate
import numpy as np
import random
import pandas as pd
import os, sys
import pickle

os.environ['CUDA_VISIBLE_DEVICES'] = str(sys.argv[1])
print("use GPU ",str(sys.argv[1]))
train_data = np.load("data/train.data")
test_data = np.load("data/test.data")
print(len(train_data),"個 training 音檔")
print(len(test_data),"個 testing 音檔")
max_frame_length = np.max([len(sample) for sample in train_data])
print("max langth of wav:",max_frame_length)

aug_train_data = []
for data in train_data:
    aug_train_data.append(data[::2])
train_data = train_data + aug_train_data



# load caption
with open("data/train.caption","r") as f:
    train_caption = f.readlines()
    train_caption = [sent.strip() for sent in train_caption]
    train_sentences = [sent.split(" ") for sent in train_caption]
with open("data/test.csv","r") as f:
    test_choice = f.readlines()
    test_choice = [sent.strip() for sent in test_choice]
    test_corpus = ",".join(test_choice)
    test_sentences = [sent.split(" ") for sent in test_corpus.split(",")]
    test_corpus = test_corpus.replace(",", " ")


# In[30]:


# chinese character level tokenizer
tokenizer = Tokenizer(num_words=None,filters='\n', lower=True, split=" ", char_level=False)
tokenizer.fit_on_texts(train_caption + [test_corpus])
print("number of token in caption:", len(tokenizer.word_index))
inv_map = {v: k for k, v in tokenizer.word_index.items()}

train_caption_sequences = tokenizer.texts_to_sequences(train_caption)
max_length = np.max([len(i) for i in train_caption_sequences])
print("max length:", max_length)



# pad sequence
train_caption_pad = pad_sequences(train_caption_sequences + train_caption_sequences, maxlen=max_length)
train_data_pad = pad_sequences(train_data, maxlen=max_frame_length,dtype='float32')
# revert
print([inv_map[i] for i in  train_caption_pad[1] if i != 0])
print(train_caption_pad.shape)
print(train_data_pad.shape)


# build training tensor (truth and fake for binary calssification)
sent_word_count = np.count_nonzero(train_caption_pad,axis=1)
false_caption = []
false_mfcc = train_data_pad
true_caption = train_caption_pad
true_mfcc = train_data_pad


## rolling way
false_caption = np.concatenate((np.roll(train_caption_pad,1,axis=0),
                                np.roll(train_caption_pad,2,axis=0),
                               np.roll(train_caption_pad,3,axis=0),
                               np.roll(train_caption_pad,4,axis=0),
                               np.roll(train_caption_pad,5,axis=0)))
false_mfcc = np.concatenate((train_data_pad,
                             train_data_pad,
                             train_data_pad,
                             train_data_pad,
                             train_data_pad))
true_caption = train_caption_pad
true_mfcc = train_data_pad



ground_truth = [ 1 for _ in range(len(true_caption))] + [0 for _ in range(len(false_caption))]
train_mfcc = np.concatenate((true_mfcc, np.array(false_mfcc)))
train_caption = np.concatenate((true_caption, np.array(false_caption)))



print(train_mfcc.shape)
print(train_caption.shape)


print(train_caption_pad[4])
print("where",np.where(np.count_nonzero(train_caption_pad,axis=1)==5)[0])
np.random.choice(np.where(np.count_nonzero(train_caption_pad,axis=1)==5)[0],1)




# model
emb_size = 100
batch_size = 512
epochs = 6
for i in range(0,40):
    mfcc_input = Input(shape=(train_mfcc.shape[1],train_mfcc.shape[2]))
    mfcc_lstm1 = Bidirectional(LSTM(128,dropout=0.2, return_sequences=True))(mfcc_input)
    mfcc_lstm2 = Bidirectional(LSTM(64,dropout=0.2))(mfcc_lstm1)
#     mfcc_lstm3 = Bidirectional(LSTM(32,dropout=0.2))(mfcc_lstm2)
    
    caption_input = Input(shape=(13,))
    emb = Embedding(len(tokenizer.word_index)+1 ,output_dim= emb_size, 
                    input_length=max_length,trainable=True)(caption_input)
    caption_lstm1 = Bidirectional(LSTM(128,dropout=0.2, return_sequences = True))(emb)
    caption_lstm2 = Bidirectional(LSTM(64,dropout=0.2))(caption_lstm1)
#     caption_lstm3 = Bidirectional(LSTM(32,dropout=0.2))(caption_lstm2)
    
    merge = keras.layers.dot([mfcc_lstm2, caption_lstm2],1)
    output_dense = Dense(1,activation="sigmoid")(merge)
    model = Model(inputs=[mfcc_input, caption_input], outputs=output_dense)
    model.compile(optimizer='adam', loss='binary_crossentropy',metrics=['acc'])
    print(model.summary())
    # training
    total_sample_size = len(ground_truth)
    random_index = np.random.choice(total_sample_size,total_sample_size, replace=False)
    
    input_mfcc = train_mfcc[random_index]
    input_caption = train_caption[random_index]
    input_ground_truth = np.array(ground_truth)[random_index]

    hist = History()
#     check_save  = ModelCheckpoint("models/model_1v3-{epoch:05d}-{val_acc:.5f}.h5",monitor='val_acc',save_best_only=True)
    check_save  = ModelCheckpoint("models/model4_aug1_1V5_2layers_"+str(batch_size)+"_"+str(i)+".h5",monitor='val_acc',save_best_only=True)
    early_stop = EarlyStopping(monitor="val_loss", patience=4)
    model.fit([input_mfcc, input_caption], input_ground_truth,
              batch_size=batch_size,
              epochs=epochs,
              validation_split=0.05,
             callbacks=[check_save, hist, early_stop])
import codecs
import re
import os
import gc 
import json
import keras
import codecs
import numpy as np
import pandas as pd
from random import choice
import keras.backend as K
import tensorflow as tf
from keras.layers import *
from keras.callbacks import *
from keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold
from keras_bert import load_trained_model_from_checkpoint, Tokenizer
# from bert4keras.bert import load_pretrained_model
# from bert4keras.utils import SimpleTokenizer, load_vocab
from keras.models import Model
from keras.optimizers import Adam
from keras.backend.tensorflow_backend import set_session

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
set_session(tf.Session(config=config))

# 数据读取
print("数据读取->->->->->->->->->->->")
train_data = []
with codecs.open("./datasets/train.csv", 'r', 'utf-8') as f:
    for line in f.readlines():
        label, comment = line.strip().split('\t')
        train_data.append([label, comment])
train_data = pd.DataFrame(train_data[1:], columns=train_data[0])
train_data = train_data[['comment', 'label']]

test_data = pd.read_csv('./datasets/test_new.csv')

# train_data['comment'] = train_data['comment'].str.replace('[^\w\s]','')
# test_data['comment'] = test_data['comment'].str.replace('[^\w\s]','')

# 数据基础统计，设定样本截断长度，这里我们选取覆盖99%数据的长度170。
print("数据基础统计->->->->->->->->->->->")
train_len_comment=[]
for i in range(0, len(train_data)):
    train_len_comment.append(len(train_data.comment[i]))
train_data['train_len_comment'] = train_len_comment
print(train_data.train_len_comment.describe(percentiles=[.5, .6,.7, .8, .85,.90,.99]))
print("************************************")

test_len_comment=[]
for i in range(0, len(test_data)):
    test_len_comment.append(len(test_data.comment[i]))
test_data['test_len_comment'] = test_len_comment
print(test_data.test_len_comment.describe(percentiles=[.5, .6,.7, .8, .85,.90,.99]))
print("************************************")

# 预训练模型
# config_path = 'G:/pretrain/chinese_wwm_ext_L-12_H-768_A-12/bert_config.json'
# checkpoint_path = 'G:/pretrain/chinese_wwm_ext_L-12_H-768_A-12/bert_model.ckpt'
# dict_path = 'G:/pretrain/chinese_wwm_ext_L-12_H-768_A-12/vocab.txt'

# config_path = 'G:/pretrain/roeberta_zh_L-24_H-1024_A-16/bert_config_large.json'
# checkpoint_path = 'G:/pretrain/roeberta_zh_L-24_H-1024_A-16/roberta_l24_large_model'
# dict_path = 'G:/pretrain/roeberta_zh_L-24_H-1024_A-16/vocab.txt'


# config_path = 'G:/pretrain/chinese_roberta_wwm_ext_L-12_H-768_A-12/bert_config.json'
# checkpoint_path = 'G:/pretrain/chinese_roberta_wwm_ext_L-12_H-768_A-12/bert_model.ckpt'
# dict_path = 'G:/pretrain/chinese_roberta_wwm_ext_L-12_H-768_A-12/vocab.txt'

config_path = 'G:/pretrain/chinese_L-12_H-768_A-12/bert_config.json'
checkpoint_path = 'G:/pretrain/chinese_L-12_H-768_A-12/bert_model.ckpt'
dict_path = 'G:/pretrain/chinese_L-12_H-768_A-12/vocab.txt'

# config_path = 'G:/pretrain/baidu_ernie/bert_config.json'
# checkpoint_path = 'G:/pretrain/baidu_ernie/bert_model.ckpt'
# dict_path = 'G:/pretrain/baidu_ernie/vocab.txt'

maxlen = 170
batch_size = 8
num_epochs = 6
learning_rate = 2e-5
nfold = 5
state = 13145

token_dict = {}
with codecs.open(dict_path, 'r', 'utf8') as reader:
    for line in reader:
        token = line.strip()
        token_dict[token] = len(token_dict)

# Tokenizer过程
class OurTokenizer(Tokenizer):
    def _tokenize(self, text):
        R = []
        for c in text:
            if c in self._token_dict:
                R.append(c)
            elif self._is_space(c):
                R.append('[unused1]')  # space类用未经训练的[unused1]表示
            else:
                R.append('[UNK]')  # 剩余的字符是[UNK]
        return R
'''
默认情况下，分词后句子首位会分别加上[CLS]和[SEP]标记，
其中[CLS]位置对应的输出向量是能代表整句的句向量（反正Bert是这样设计的），
而[SEP]则是句间的分隔符，其余部分则是单字输出（对于中文来说）。
'''
tokenizer = OurTokenizer(token_dict)

def seq_padding(X, padding=0):
    L = [len(x) for x in X]
    ML = max(L)
    return np.array([
        np.concatenate([x, [padding] * (ML - len(x))]) if len(x) < ML else x for x in X
    ])


class data_generator:
    def __init__(self, data, batch_size=batch_size, shuffle=True):
        self.data = data
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.steps = len(self.data) // self.batch_size
        if len(self.data) % self.batch_size != 0:
            self.steps += 1

    def __len__(self):
        return self.steps

    def __iter__(self):
        while True:
            idxs = list(range(len(self.data)))

            if self.shuffle:
                np.random.shuffle(idxs)

            X1, X2, Y = [], [], []
            for i in idxs:
                d = self.data[i]
                text = d[0][:maxlen]
                x1, x2 = tokenizer.encode(first=text)
                y = d[1]
                X1.append(x1)
                X2.append(x2)
                Y.append([y])
                if len(X1) == self.batch_size or i == idxs[-1]:
                    X1 = seq_padding(X1)
                    X2 = seq_padding(X2)
                    Y = np.array(Y)
                    yield [X1, X2], Y
                    [X1, X2, Y] = [], [], []

#bert模型
def build_bert(nclass):
    bert_model = load_trained_model_from_checkpoint(config_path, checkpoint_path, seq_len=None)
	# 注意，尽管可以设置seq_len=None，但是仍要保证序列长度不超过512
    for l in bert_model.layers:
        l.trainable = True

    x1_in = Input(shape=(None,))
    x2_in = Input(shape=(None,))

    x = bert_model([x1_in, x2_in])
    x = Lambda(lambda x: x[:, 0])(x)
    p = Dense(nclass, activation='sigmoid')(x)

    model = Model([x1_in, x2_in], p)
    model.compile(loss='binary_crossentropy',
                  optimizer=Adam(learning_rate),# 用足够小的学习率
                  metrics=['accuracy'])
    print(model.summary())
    return model

def run_cv(nfold, data, data_test):
	# StratifiedKFold 分层采样交叉切分，确保训练集，测试集中各类别样本的比例与原始数据集中相同。
    skf = StratifiedKFold(n_splits=nfold, shuffle=False, random_state=state).split(data[:, 0], data[:, 1])
    train_model_pred = np.zeros((len(data), 1))
    test_model_pred = np.zeros((len(data_test), 1))

    for i, (train_fold, valid_fold) in enumerate(skf):
        print('第%s折开始训练' % (i + 1))
        X_train, X_valid, = data[train_fold, :], data[valid_fold, :]

        model = build_bert(1)

        train_D = data_generator(X_train, shuffle=True)
        valid_D = data_generator(X_valid, shuffle=True)
        test_D = data_generator(data_test, shuffle=False)
        
        early_stopping = EarlyStopping(monitor='val_loss', patience=2)
        plateau = ReduceLROnPlateau(monitor="val_loss", verbose=1, mode='min', factor=0.2, patience=2)
        checkpoint = ModelCheckpoint('./model_save/11034/' + str(i) + '.hdf5', monitor='val_loss',
                                     verbose=2, save_best_only=True, mode='min', save_weights_only=True)

        model.fit_generator(
            train_D.__iter__(),
            steps_per_epoch=len(train_D),
            epochs = num_epochs,
            validation_data=valid_D.__iter__(),
            validation_steps=len(valid_D),
            callbacks=[early_stopping, plateau, checkpoint],
        )

        train_model_pred[valid_fold, :] = model.predict_generator(valid_D.__iter__(), steps=len(valid_D), verbose=1)
        test_model_pred += model.predict_generator(test_D.__iter__(), steps=len(test_D), verbose=1) / nfold

        del model;
        gc.collect()
        K.clear_session()

    return train_model_pred, test_model_pred

DATA_LIST = []
for data_row in train_data.iloc[:].itertuples():
    DATA_LIST.append((data_row.comment, data_row.label))
DATA_LIST = np.array(DATA_LIST)

DATA_LIST_TEST = []
for data_row in test_data.iloc[:].itertuples():
    DATA_LIST_TEST.append((data_row.comment, 0))
DATA_LIST_TEST = np.array(DATA_LIST_TEST)

train_model_pred, test_model_pred = run_cv(nfold, DATA_LIST, DATA_LIST_TEST)

# 预测概率文件和结果文件保存
pd.DataFrame(test_model_pred).to_csv("./test_pred/test_model_pred11034.csv", index=None)

test_pred = pd.read_csv("./test_pred/test_model_pred11034.csv")
for i in range(0,len(test_pred)):
    if test_pred['0'][i] > 0.5:
        test_pred['0'][i] = 1
    else:
        test_pred['0'][i] = 0
test_data['label'] = np.int64(test_pred['0'])
test_data[['id', 'label']].to_csv('./submit/sub11034.csv', index=None)
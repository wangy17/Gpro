import sys

from keras.layers import Input, Dense, Conv1D, MaxPooling1D, BatchNormalization, LSTM
from keras.layers.core import Flatten, Dropout,Lambda
from keras.models import Model
from keras.layers.merge import Concatenate
from keras import optimizers
from keras.callbacks import EarlyStopping,ModelCheckpoint
import numpy as np
from keras import regularizers
from keras import backend as K
import os
from ..ProcessData import seq2oh,GetCharMap,load_fun_data
def prvalue(y_true, y_pred):
    x = y_true
    y = y_pred
    mx = K.mean(x)
    my = K.mean(y)
    xm, ym = x-mx, y-my
    r_num = K.sum(xm * ym)
    r_den = K.sqrt(K.sum(xm*xm) * K.sum(ym*ym))
    r = r_num / r_den
    return r

class RNN():

    def BuildModel(self,
                   train_data,
                   val_data=None,
                   DIM = 128,
                   kernel_size = 5
                   ):
        self.x,self.y = load_fun_data(train_data)
        self.charmap, self.invcharmap = GetCharMap(self.x)
        self.x = seq2oh(self.x,self.charmap)
        self.seq_len = self.x.shape[1]
        if val_data != None:
            self.val_x, self.val_y = load_fun_data(val_data)
            self.val_x = seq2oh(self.val_x,self.charmap)
        else:
            d = self.x.shape[0]//10 *9
            self.val_x, self.val_y = self.x[d:,:,:], self.y[d:]
            self.x, self.y = self.x[:d,:,:], self.y[:d]
        self.DIM = DIM
        self.kernel_size = 5
        regW = regularizers.l2(0.00001)
        seqInput = Input(shape=(self.seq_len,4),name='seqInput')
        lstm1 = LSTM(self.DIM)(seqInput)
        
        x = Conv1D(self.DIM, self.kernel_size, activation='relu', activity_regularizer = None)(seqInput)
        x = MaxPooling1D(pool_size=2)(x)
        x = BatchNormalization()(x)
        lstm2 = LSTM(self.DIM)(x)
        
        x = Conv1D(self.DIM*2, self.kernel_size, activation='relu', activity_regularizer = None)(x)
        x = MaxPooling1D(pool_size=2)(x)
        x = BatchNormalization()(x)
        lstm3 = LSTM(self.DIM)(x)

        x = Concatenate()([lstm1,lstm2,lstm3])
        x = Dropout(0.2)(x)
        x = BatchNormalization()(x)
        y = Dense(1,activation='relu', kernel_regularizer= regW)(x)
        self.model = Model(inputs=[seqInput],outputs=[y])
        return

    def Train(self,
              lr=1e-4,
              beta1=0.5,
              beta2=0.9,
              batch_size=32,
              epoch=1000,
              earlystop=20,
              weight_dir='./predict_model',
              model_name='predictor',
              ):
        self.batch_size = batch_size
        self.epoch = epoch
        self.earlystop = earlystop
        self.weight_dir = weight_dir
        if os.path.exists(self.weight_dir) == False:
            os.makedirs(self.weight_dir)
        self.model_name = model_name
        
        self.opt = optimizers.Adam(lr=lr,beta_1=beta1,beta_2=beta2)
        self.model.compile(optimizer=self.opt,loss='mse',metrics = [prvalue])
        json_string = self.model.to_json()
        open(weight_dir+model_name+'.json','w').write(json_string)
        self.model.fit([self.x],[self.y],
                       batch_size=self.batch_size,
                       epochs=self.epoch,
                       verbose=1,
                       validation_data=(self.val_x,self.val_y),
                       callbacks=[EarlyStopping(patience=self.earlystop),
                                  ModelCheckpoint(filepath=self.weight_dir+'/weight.h5',save_best_only=True)])
#        self.model.load_weights(self.weight_dir+'/weight.h5')
#        pred = self.model.predict(self.val_x)
#        plot(self.val_y,pred,'expression')
        return

    def load(self,weight_dir='./predict_model'):
        self.model.load_weights(weight_dir+'/weight.h5')
        return
    
    def Predictor(self,seq,datatype='str'):
        if datatype=='str':
            seq = seq2oh(seq,self.charmap)
        y = self.model.predict(seq)
        y = np.reshape(y,[y.shape[0]])
        return y
    
def plot(real,pred,name):
    import matplotlib.pyplot as plt
    plt.clf()
    plt.scatter(real,pred)
    plt.xlabel('True value')
    plt.ylabel('Predict value')
    plt.savefig(name+'.jpg')
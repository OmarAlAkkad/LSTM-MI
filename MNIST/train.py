'''
ReNet-Keras
Code written by: Xiaoqing Liu
If you use significant portions of this code or the ideas from our paper, please cite it :)

'''
import argparse
from model import build_model
from load_2D_data import train_total
from tensorflow.keras.optimizers import Adam,SGD

def train():
    opt = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon = 0.1, decay = 1e-6)

    model = build_model( nClasses=2)
    model.compile(loss='binary_crossentropy',optimizer= opt ,metrics=['accuracy'])
    x,y=train_total('./Parasitized')

    model.fit(x,y,batch_size=1,epochs=5)

if __name__ == '__main__':
    opt = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon = 0.1, decay = 1e-6)

    model = build_model( nClasses=2)
    model.compile(loss='binary_crossentropy',optimizer= opt ,metrics=['accuracy'])
    x,y=train_total('./Parasitized')

    model.fit(x,y,batch_size=1,epochs=5)
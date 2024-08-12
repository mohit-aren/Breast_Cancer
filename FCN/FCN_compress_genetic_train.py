import numpy as np
import pandas as pd

import json
import sys
from PIL import Image, ImageOps

#from skimage.io import imread
#from matplotlib import pyplot as plt
import random

import os
#os.environ['KERAS_BACKEND'] = 'theano'
#os.environ['THEANO_FLAGS'] ='mode=FAST_RUN,device=cpu'
#os.environ['THEANO_FLAGS'] = 'mode=FAST_RUN, device=gpu0, floatX=float32, optimizer=fast_compile'

#from tensorflow.keras import models
from keras.optimizers import SGD
#from tensorflow.keras.layers import Input, ZeroPadding2D
#from tensorflow.keras.layers import Activation, Flatten, Reshape
#from tensorflow.keras.layers import Convolution2D, MaxPooling2D, UpSampling2D
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.models import Model
#from tensorflow.keras import utils as  np_utils
#from keras.applications import imagenet_utils

path = 'results/'
img_w = 320
img_h = 320
n_labels = 2

Lung = [255,255,255]

Sky = [128,128,128]
Building = [128,0,0]
Pole = [192,192,128]
Road_marking = [255,69,0]
Road = [128,64,128]
Pavement = [60,40,222]
Tree = [128,128,0]
SignSymbol = [192,128,128]
Fence = [64,64,128]
Car = [64,0,128]
Pedestrian = [64,64,0]
Bicyclist = [0,128,192]
Unlabelled = [0,0,0]


n_train = 89
n_test = 6
n_val = 5

def label_map(labels):
    label_map = np.zeros([img_h, img_w, n_labels])    
    for r in range(img_h):
        for c in range(img_w):
            label_map[r, c, labels[r][c]] = 1
    return label_map

def label_map1(labels):
    label_map = np.zeros([img_h, img_w, n_labels])    
    for r in range(img_h):
        for c in range(img_w):
            #print(labels[r][c])
            if(labels[r][c][0] >180 and labels[r][c][1] > 180 and labels[r][c][2] > 180):
                label_map[r, c, 1] = 1
            elif(labels[r][c][0] < 50 and labels[r][c][1] < 50 and labels[r][c][2] < 50):
                label_map[r, c, 0] = 1
    return label_map


import os
import imageio

def prep_data1(mode):
    data = []
    label = []
    
    folder_path = 'benign' # path + mode

    #images_path = os.listdir(folder_path)
    images_path = os.listdir(folder_path)

    if(mode == 'train'):
        n = 50
    elif(mode == 'val'):
        n = 45
    else:
        n = 10
    for index, image in enumerate(images_path):

        index += 1
        filename = os.path.join(folder_path, folder_path + ' ('+ str(index) + ').png')
    
        print(index, filename)
        if(index > 50 and mode == 'train'):
            continue
        elif((index < 51 or index > 95) and mode =='val'):
            continue
        elif((index < 96 or index > 105) and mode == 'test'):
            continue
        
        #truth_file = filename.split('.png')
        truth_file = filename.split('.png')
        
        tfile = truth_file[0] + '_mask.png'
    
        
        print(tfile)
        if(filename == ""):
            break
        #img1 = Image.open(filename)
        img1 = Image.open(filename)
        w, h = img1.size
        start = (w-320)//2
        s_h = (h-320)//2
        
        new_im = img1.crop((start, s_h, start+320, s_h+320))
        #new_size = tuple([544, 512])
        
        # create a new image and paste the resized on it
        
        #new_im = img1.resize((320, 320))
        #new_im = Image.new("RGB", (544, 512))
        #new_im.paste(img1, ((544-new_size[0])//2,
                            #(512-new_size[1])//2))



        #img2 = Image.open(tfile)
        img2 = Image.open(tfile)
        #new_im1 = img2.resize((320, 320))
        new_im1 = img2.convert('RGB')
        new_im1 = new_im1.crop((start, s_h, start+320, s_h+320))
        #new_size = tuple([544, 512])
        
        # create a new image and paste the resized on it
        
        #new_im1 = Image.new("RGB", (544, 512))
        #new_im1.paste(img2, ((544-new_size[0])//2,
                            #(512-new_size[1])//2))


        #index += 1
        # create a new image and paste the resized on it
        

        #img, gt = [imread(path + mode + '/' + filename + '.png')], imread(path + mode + '-colormap/' + filename + '.png')
        
        img, gt = [np.array(new_im,dtype=np.uint8)], np.array(new_im1,dtype=np.uint8)
        data.append(np.reshape(img,(320, 320,3)))
        label.append(label_map1(gt))
        sys.stdout.write('\r')
        sys.stdout.flush()
    sys.stdout.write('\r')
    sys.stdout.flush()
    data, label = np.array(data), np.array(label).reshape((n, img_h * img_w, n_labels))

    print( mode + ': OK')
    print( '\tshapes: {}, {}'.format(data.shape, label.shape))
    print( '\ttypes:  {}, {}'.format(data.dtype, label.dtype))
    #print( '\tmemory: {}, {} MB'.format(data.nbytes / 1048544, label.nbytes / 1048544))

    return data, label
    


def prep_data(mode):
    assert mode in {'test', 'train'}, \
        'mode should be either \'test\' or \'train\''
    data = []
    label = []
    df = pd.read_csv(path + mode + '.csv')
    n = n_train if mode == 'train' else n_test
    for i, item in df.iterrows():
        if i >= n:
            break
        img, gt = [imread(path + item[0])], np.clip(imread(path + item[1]), 0, 1)
        data.append(np.reshape(img,(256,256,1)))
        label.append(label_map(gt))
        sys.stdout.write('\r')
        sys.stdout.write(mode + ": [%-20s] %d%%" % ('=' * int(20. * (i + 1) / n - 1) + '>',
                                                    int(100. * (i + 1) / n)))
        sys.stdout.flush()
    sys.stdout.write('\r')
    sys.stdout.flush()
    data, label = np.array(data), np.array(label).reshape((n, img_h * img_w, n_labels))

    print( mode + ': OK')
    print( '\tshapes: {}, {}'.format(data.shape, label.shape))
    print( '\ttypes:  {}, {}'.format(data.dtype, label.dtype))
    print( '\tmemory: {}, {} MB'.format(data.nbytes / 1048576, label.nbytes / 1048576))

    return data, label

"""
def plot_results(output):
    gt = []
    df = pd.read_csv(path + 'test.csv')
    for i, item in df.iterrows():
        gt.append(np.clip(imread(path + item[1]), 0, 1))

    plt.figure(figsize=(15, 2 * n_test))
    for i, item in df.iterrows():
        plt.subplot(n_test, 4, 4 * i + 1)
        plt.title('Ground Truth')
        plt.axis('off')
        gt = imread(path + item[1])
        plt.imshow(np.clip(gt, 0, 1))

        plt.subplot(n_test, 4, 4 * i + 2)
        plt.title('Prediction')
        plt.axis('off')
        labeled = np.argmax(output[i], axis=-1)
        plt.imshow(labeled)

        plt.subplot(n_test, 4, 4 * i + 3)
        plt.title('Heat map')
        plt.axis('off')
        plt.imshow(output[i][:, :, 1])

        plt.subplot(n_test, 4, 4 * i + 4)
        plt.title('Comparison')
        plt.axis('off')
        rgb = np.empty((img_h, img_w, 3))
        rgb[:, :, 0] = labeled
        rgb[:, :, 1] = imread(path + item[0])
        rgb[:, :, 2] = gt
        plt.imshow(rgb)

    plt.savefig('result.png')
    plt.show()
"""

#########################################################################################################

import numpy as np
#import matplotlib.pyplot as plt
#from pylab import *
import os
import sys
#from tensorflow.keras_contrib.applications import densenet
#from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l2
from tensorflow.keras.layers import *
#from tensorflow.keras.engine import Layer
#from tensorflow.keras.applications.vgg16 import *
from tensorflow.keras.models import *
#from keras_applications.imagenet_utils import _obtain_input_shape
#import keras.backend as K
import tensorflow as tf

from utils.get_weights_path import *
from utils.basics import *
from utils.resnet_helpers import *
from utils.BilinearUpSampling import *

def FCN_Vgg16_32s(input_shape=(320, 320, 3), weight_decay=0., batch_momentum=0.9, batch_shape=None, classes=2):
    if batch_shape:
        img_input = Input(batch_shape=batch_shape)
        image_size = batch_shape[1:3]
    else:
        img_input = Input(shape=input_shape)
        image_size = input_shape[0:2]
    # Block 1
    x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv1', kernel_regularizer=l2(weight_decay))(img_input)
    x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv2', kernel_regularizer=l2(weight_decay))(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(x)

    # Block 2
    x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv1', kernel_regularizer=l2(weight_decay))(x)
    x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv2', kernel_regularizer=l2(weight_decay))(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(x)

    # Block 3
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv1', kernel_regularizer=l2(weight_decay))(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv2', kernel_regularizer=l2(weight_decay))(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv3', kernel_regularizer=l2(weight_decay))(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')(x)

    # Block 4
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv1', kernel_regularizer=l2(weight_decay))(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv2', kernel_regularizer=l2(weight_decay))(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv3', kernel_regularizer=l2(weight_decay))(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool')(x)

    # Block 5
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv1', kernel_regularizer=l2(weight_decay))(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv2', kernel_regularizer=l2(weight_decay))(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv3', kernel_regularizer=l2(weight_decay))(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool')(x)

    # Convolutional layers transfered from fully-connected layers
    x = Conv2D(4096, (7, 7), activation='relu', padding='same', name='fc1', kernel_regularizer=l2(weight_decay))(x)
    x = Dropout(0.5)(x)
    x = Conv2D(4096, (1, 1), activation='relu', padding='same', name='fc2', kernel_regularizer=l2(weight_decay))(x)
    x = Dropout(0.5)(x)
    #classifying layer
    x = Conv2D(classes, (1, 1), kernel_initializer='he_normal', activation='linear', padding='valid', strides=(1, 1), kernel_regularizer=l2(weight_decay))(x)

    x = BilinearUpSampling2D(size=(32, 32))(x)
    x = Reshape((320*320, 2))(x)
    x = Activation("softmax")(x)

    model = Model(img_input, x)

    #weights_path = os.path.expanduser(os.path.join('~', '.keras/models/fcn_vgg16_weights_tf_dim_ordering_tf_kernels.h5'))
    #model.load_weights(weights_path, by_name=True)
    return model

#########################################################################################################
    


def FCN_Vgg16_32s_compress(layer1_filters, layer2_filters, layer3_filters, layer4_filters, layer5_filters, layer6_filters, layer7_filters, layer8_filters,layer9_filters, layer10_filters, layer11_filters, layer12_filters, layer13_filters, layer14_filters, layer15_filters, input_shape=(320, 320, 3), weight_decay=0., batch_momentum=0.9, batch_shape=None, classes=2):
    if batch_shape:
        img_input = Input(batch_shape=batch_shape)
        image_size = batch_shape[1:3]
    else:
        img_input = Input(shape=input_shape)
        image_size = input_shape[0:2]
    # Block 1
    x = Conv2D(layer1_filters, (3, 3), activation='relu', padding='same', name='block1_conv1', kernel_regularizer=l2(weight_decay))(img_input)
    x = Conv2D(layer2_filters, (3, 3), activation='relu', padding='same', name='block1_conv2', kernel_regularizer=l2(weight_decay))(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(x)

    # Block 2
    x = Conv2D(layer3_filters, (3, 3), activation='relu', padding='same', name='block2_conv1', kernel_regularizer=l2(weight_decay))(x)
    x = Conv2D(layer4_filters, (3, 3), activation='relu', padding='same', name='block2_conv2', kernel_regularizer=l2(weight_decay))(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(x)

    # Block 3
    x = Conv2D(layer5_filters, (3, 3), activation='relu', padding='same', name='block3_conv1', kernel_regularizer=l2(weight_decay))(x)
    x = Conv2D(layer6_filters, (3, 3), activation='relu', padding='same', name='block3_conv2', kernel_regularizer=l2(weight_decay))(x)
    x = Conv2D(layer7_filters, (3, 3), activation='relu', padding='same', name='block3_conv3', kernel_regularizer=l2(weight_decay))(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')(x)

    # Block 4
    x = Conv2D(layer8_filters, (3, 3), activation='relu', padding='same', name='block4_conv1', kernel_regularizer=l2(weight_decay))(x)
    x = Conv2D(layer9_filters, (3, 3), activation='relu', padding='same', name='block4_conv2', kernel_regularizer=l2(weight_decay))(x)
    x = Conv2D(layer10_filters, (3, 3), activation='relu', padding='same', name='block4_conv3', kernel_regularizer=l2(weight_decay))(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool')(x)

    # Block 5
    x = Conv2D(layer11_filters, (3, 3), activation='relu', padding='same', name='block5_conv1', kernel_regularizer=l2(weight_decay))(x)
    x = Conv2D(layer12_filters, (3, 3), activation='relu', padding='same', name='block5_conv2', kernel_regularizer=l2(weight_decay))(x)
    x = Conv2D(layer13_filters, (3, 3), activation='relu', padding='same', name='block5_conv3', kernel_regularizer=l2(weight_decay))(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool')(x)

    # Convolutional layers transfered from fully-connected layers
    x = Conv2D(layer14_filters, (7, 7), activation='relu', padding='same', name='fc1', kernel_regularizer=l2(weight_decay))(x)
    x = Dropout(0.5)(x)
    x = Conv2D(layer15_filters, (1, 1), activation='relu', padding='same', name='fc2', kernel_regularizer=l2(weight_decay))(x)
    x = Dropout(0.5)(x)
    #classifying layer
    x = Conv2D(classes, (1, 1), kernel_initializer='he_normal', activation='linear', padding='valid', strides=(1, 1), kernel_regularizer=l2(weight_decay))(x)

    x = BilinearUpSampling2D(size=(32, 32))(x)
    x = Reshape((320*320, 2))(x)
    x = Activation("softmax")(x)

    model = Model(img_input, x)

    #weights_path = os.path.expanduser(os.path.join('~', '.keras/models/fcn_vgg16_weights_tf_dim_ordering_tf_kernels.h5'))
    #model.load_weights(weights_path, by_name=True)
    return model



"""
with open('model_5l.json') as model_file:
    autoencoder = models.model_from_json(model_file.read())
"""

autoencoder = FCN_Vgg16_32s()

print('Start')
optimizer = SGD(lr=0.001, momentum=0.9, decay=0.0005, nesterov=False)
autoencoder.compile(loss="categorical_crossentropy", optimizer=optimizer, metrics=['accuracy'])
print( 'Compiled: OK')
autoencoder.summary()

# Train model or load weights

train_data, train_label = prep_data1('train')
val_data, val_label = prep_data1('val')
nb_epoch = 100
batch_size = 2
#history = autoencoder.fit(train_data, train_label, batch_size=batch_size, nb_epoch=nb_epoch, verbose=1, validation_data=(val_data, val_label))

autoencoder.load_weights('model_5l_weight_fcn_lungs.1.hdf5')

test_data, test_label = prep_data1('test')
score = autoencoder.evaluate(test_data, test_label, verbose=1)
print( 'Test score:', score[0])
print( 'Test accuracy:', score[1])

olayer1_filters = 64
olayer2_filters = 64
olayer3_filters = 128
olayer4_filters = 128
olayer5_filters = 256
olayer6_filters = 256
olayer7_filters = 256
olayer8_filters = 512
olayer9_filters = 512
olayer10_filters = 512
olayer11_filters = 512
olayer12_filters = 512
olayer13_filters = 512
olayer14_filters = 4096
olayer15_filters = 4096


for withtrain in range(0,11):
    model_final = FCN_Vgg16_32s_compress(olayer1_filters, olayer2_filters, olayer3_filters, olayer4_filters, olayer5_filters, olayer6_filters, olayer7_filters, olayer8_filters,olayer9_filters, olayer10_filters, olayer11_filters, olayer12_filters, olayer13_filters, olayer14_filters, olayer15_filters)

    print('Start')
    optimizer = SGD(lr=0.001, momentum=0.9, decay=0.0005, nesterov=False)
    model_final.compile(loss="categorical_crossentropy", optimizer=optimizer, metrics=['accuracy'])
    print( 'Compiled: OK')
    model_final.summary()
    
    wt = "FCN_pruned_weights_" + str(withtrain) + "ga.h5"
    model_final.load_weights(wt)

    nlayer1_filters = olayer1_filters
    nlayer2_filters = olayer2_filters
    nlayer3_filters = olayer3_filters
    nlayer4_filters = olayer4_filters
    nlayer5_filters = olayer5_filters
    nlayer6_filters = olayer6_filters
    nlayer7_filters = olayer7_filters
    nlayer8_filters = olayer8_filters
    nlayer9_filters = olayer9_filters
    nlayer10_filters = olayer10_filters
    nlayer11_filters = olayer11_filters
    nlayer12_filters = olayer12_filters
    nlayer13_filters = olayer13_filters
    nlayer14_filters = olayer14_filters
    nlayer15_filters = olayer15_filters

    ####################### 1st convolution layer with olayer1_filters filters
    print('1st convolution layer with olayer1_filters filters')
    A = []
    Acc = []
    
    arr = model_final.evaluate(test_data, test_label, verbose=1)
    print(arr)
    
    filters, biases = model_final.layers[1].get_weights()
    filters1 = np.copy(filters)
    biases1 = np.copy(biases)
    
    for index in range(0,10):
        # normalize filter values to 0-1 so we can visualize them
        # plot first few filters
        n_filters, ix = len(filters), 1
        B = []
        for j in range(0,olayer1_filters):
            x = random.randint(0,1)
            #print(x)
            B.append(x)
        
        for i in range(n_filters):
            f = filters[:, i]
            if(B[i] == 0):
                filters1[:,:,:, i] = 0
                biases1[i] = 0
    
    
        model_final.layers[1].set_weights([filters1, biases1])
        
        filters1 = np.copy(filters)
        biases1 = np.copy(biases)
    
        arr = model_final.evaluate(test_data, test_label, verbose=1)
        print (arr)
        if(arr[1] > 0.000010):
            A.append(B) 
            Acc.append(arr[1])
            print(index, 'added')
        
    
    max1 = 0
    max_index1 = 0
    index = 0
    while index < len(Acc):
        if(Acc[index]> max1):
            max1 = Acc[index]
            max_index1 = index
        index += 1
            
    Acc[max_index1] = -1
    max2 = 0
    max_index2 = 0
    index = 0
    while index < len(Acc):
        if(Acc[index]> max2):
            max2 = Acc[index]
            max_index2 = index
        index += 1
            
    
    par1 = np.copy(A[max_index1])
    par2 = np.copy(A[max_index2])
    print(max1, max2)
    temp_sat = 0
    for index_ga in range(0, 20):
        new_max = 0
        child = []
        temp_index = 0
        while (new_max < max1 and new_max < max2 and temp_index < 10):
            k = random.randint(10,100)
            
            child = np.copy(par1)
            #Crossover
            for index in range(k, olayer1_filters):
                child[index] = par2[index]
                
            #Mutation
            
            for l in range(0,5):
                temp_mut = random.randint(0,olayer1_filters-1)
                child[temp_mut] = 1-child[temp_mut]
                
                
            filters1 = np.copy(filters)
            biases1 = np.copy(biases)
            
            for i in range(0,olayer1_filters):
                f = filters[:,:,:, i]
                #for j in range(3):
                    #if(child[i] == 0):
                        #filters1[:, :, :, i][:,:,j] = 0
                if(child[i] == 0)        :
                    biases1[i] = 0
                    filters1[:,:,:, i] = 0
            
            model_final.layers[1].set_weights([filters1, biases1])
            arr = model_final.evaluate(test_data, test_label, verbose=1)
            new_max = 0.99*arr[1]+0.01*len(child)/np.sum(child)
            print (arr)
            temp_index += 1
        if(new_max >= max1 and new_max-max1 < 0.000001):
            temp_sat += 1
        if(temp_sat >0):
            filters1 = np.copy(filters)
            biases1 = np.copy(biases)
            
            for i in range(0,olayer1_filters):
                f = filters[:,:,:, i]
                #for j in range(3):
                    #if(par1[i] == 0):
                        #filters1[:, :, :, i][:,:,j] = 0
                if(par1[i] == 0)        :
                    biases1[i] = 0
                    filters1[:,:,:, i] = 0
                    
            model_final.layers[1].set_weights([filters1, biases1])
            arr = model_final.evaluate(test_data, test_label, verbose=1)
            print(par1)
            print('Filters left', np.sum(par1))
            print (arr)
            break
        print('new max', new_max)
        if(new_max > max1):
            max2 = max1
            par2 = np.copy(par1)
            par1 = np.copy(child)
            max1 = new_max
            print('max1', max1)
            print(par1)
        elif(new_max > max2):
            par2 = np.copy(child)
            max2 = new_max
            print('max2', max2)
    
           
    new_num = np.sum(par1)
           
    print(new_num)


    if(new_num > 28):
        olayer1_filters = new_num
    else:
        par1 = []
        for k in range(0, olayer1_filters):
            par1.append(1)
            
    A1 = np.copy(par1)
    ####################### 2nd convolution layer with olayer2_filters filters
    print('2nd convolution layer with olayer2_filters filters')
    A = []
    Acc = []
    
    arr = model_final.evaluate(test_data, test_label, verbose=1)
    print(arr)
    
    filters, biases = model_final.layers[2].get_weights()
    filters1 = np.copy(filters)
    biases1 = np.copy(biases)
    
    for index in range(0,10):
        # normalize filter values to 0-1 so we can visualize them
        # plot first few filters
        n_filters, ix = len(filters), 1
        B = []
        for j in range(0,olayer2_filters):
            x = random.randint(0,1)
            #print(x)
            B.append(x)
        
        for i in range(n_filters):
            f = filters[:, i]
            if(B[i] == 0):
                filters1[:,:,:, i] = 0
                biases1[i] = 0
    
    
        model_final.layers[2].set_weights([filters1, biases1])
        
        filters1 = np.copy(filters)
        biases1 = np.copy(biases)
    
        arr = model_final.evaluate(test_data, test_label, verbose=1)
        print (arr)
        if(arr[1] > 0.000010):
            A.append(B) 
            Acc.append(arr[1])
            print(index, 'added')
        
    
    max1 = 0
    max_index1 = 0
    index = 0
    while index < len(Acc):
        if(Acc[index]> max1):
            max1 = Acc[index]
            max_index1 = index
        index += 1
            
    Acc[max_index1] = -1
    max2 = 0
    max_index2 = 0
    index = 0
    while index < len(Acc):
        if(Acc[index]> max2):
            max2 = Acc[index]
            max_index2 = index
        index += 1
            
    
    par1 = np.copy(A[max_index1])
    par2 = np.copy(A[max_index2])
    print(max1, max2)
    temp_sat = 0
    for index_ga in range(0, 20):
        new_max = 0
        child = []
        temp_index = 0
        while (new_max < max1 and new_max < max2 and temp_index < 10):
            k = random.randint(10,100)
            
            child = np.copy(par1)
            #Crossover
            for index in range(k, olayer2_filters):
                child[index] = par2[index]
                
            #Mutation
            
            for l in range(0,5):
                temp_mut = random.randint(0,olayer2_filters-1)
                child[temp_mut] = 1-child[temp_mut]
                
                
            filters1 = np.copy(filters)
            biases1 = np.copy(biases)
            
            for i in range(0,olayer2_filters):
                f = filters[:,:,:, i]
                #for j in range(3):
                    #if(child[i] == 0):
                        #filters1[:, :, :, i][:,:,j] = 0
                if(child[i] == 0)        :
                    biases1[i] = 0
                    filters1[:,:,:, i] = 0
            
            model_final.layers[2].set_weights([filters1, biases1])
            arr = model_final.evaluate(test_data, test_label, verbose=1)
            new_max = 0.99*arr[1]+0.01*len(child)/np.sum(child)
            print (arr)
            temp_index += 1
        if(new_max >= max1 and new_max-max1 < 0.000001):
            temp_sat += 1
        if(temp_sat >0):
            filters1 = np.copy(filters)
            biases1 = np.copy(biases)
            
            for i in range(0,olayer2_filters):
                f = filters[:,:,:, i]
                #for j in range(3):
                    #if(par1[i] == 0):
                        #filters1[:, :, :, i][:,:,j] = 0
                if(par1[i] == 0)        :
                    biases1[i] = 0
                    filters1[:,:,:, i] = 0
                    
            model_final.layers[2].set_weights([filters1, biases1])
            arr = model_final.evaluate(test_data, test_label, verbose=1)
            print(par1)
            print('Filters left', np.sum(par1))
            print (arr)
            break
        print('new max', new_max)
        if(new_max > max1):
            max2 = max1
            par2 = np.copy(par1)
            par1 = np.copy(child)
            max1 = new_max
            print('max1', max1)
            print(par1)
        elif(new_max > max2):
            par2 = np.copy(child)
            max2 = new_max
            print('max2', max2)
    
    new_num = np.sum(par1)
           
    print(new_num)


    if(new_num > 28):
        olayer2_filters = new_num
    else:
        par1 = []
        for k in range(0, olayer2_filters):
            par1.append(1)
            
    A2 = np.copy(par1)
    
    ####################### 3rd convolution layer with olayer3_filters filters
    print('3rd convolution layer with olayer3_filters filters')
    A = []
    Acc = []
    
    arr = model_final.evaluate(test_data, test_label, verbose=1)
    print(arr)
    
    filters, biases = model_final.layers[4].get_weights()
    filters1 = np.copy(filters)
    biases1 = np.copy(biases)
    
    for index in range(0,10):
        # normalize filter values to 0-1 so we can visualize them
        # plot first few filters
        n_filters, ix = len(filters), 1
        B = []
        for j in range(0,olayer3_filters):
            x = random.randint(0,1)
            #print(x)
            B.append(x)
        
        for i in range(n_filters):
            f = filters[:, i]
            if(B[i] == 0):
                filters1[:,:,:, i] = 0
                biases1[i] = 0
    
    
        model_final.layers[4].set_weights([filters1, biases1])
        
        filters1 = np.copy(filters)
        biases1 = np.copy(biases)
    
        arr = model_final.evaluate(test_data, test_label, verbose=1)
        print (arr)
        if(arr[1] > 0.000010):
            A.append(B) 
            Acc.append(arr[1])
            print(index, 'added')
        
    
    max1 = 0
    max_index1 = 0
    index = 0
    while index < len(Acc):
        if(Acc[index]> max1):
            max1 = Acc[index]
            max_index1 = index
        index += 1
            
    Acc[max_index1] = -1
    max2 = 0
    max_index2 = 0
    index = 0
    while index < len(Acc):
        if(Acc[index]> max2):
            max2 = Acc[index]
            max_index2 = index
        index += 1
            
    
    par1 = np.copy(A[max_index1])
    par2 = np.copy(A[max_index2])
    print(max1, max2)
    temp_sat = 0
    for index_ga in range(0, 20):
        new_max = 0
        child = []
        temp_index = 0
        while (new_max < max1 and new_max < max2 and temp_index < 10):
            k = random.randint(10,100)
            
            child = np.copy(par1)
            #Crossover
            for index in range(k, olayer3_filters):
                child[index] = par2[index]
                
            #Mutation
            
            for l in range(0,5):
                temp_mut = random.randint(0,olayer3_filters-1)
                child[temp_mut] = 1-child[temp_mut]
                
                
            filters1 = np.copy(filters)
            biases1 = np.copy(biases)
            
            for i in range(0,olayer3_filters):
                f = filters[:,:,:, i]
                #for j in range(3):
                    #if(child[i] == 0):
                        #filters1[:, :, :, i][:,:,j] = 0
                if(child[i] == 0)        :
                    biases1[i] = 0
                    filters1[:,:,:, i] = 0
            
            model_final.layers[4].set_weights([filters1, biases1])
            arr = model_final.evaluate(test_data, test_label, verbose=1)
            new_max = 0.99*arr[1]+0.01*len(child)/np.sum(child)
            print (arr)
            temp_index += 1
        if(new_max >= max1 and new_max-max1 < 0.000001):
            temp_sat += 1
        if(temp_sat >0):
            filters1 = np.copy(filters)
            biases1 = np.copy(biases)
            
            for i in range(0,olayer3_filters):
                f = filters[:,:,:, i]
                #for j in range(3):
                    #if(par1[i] == 0):
                        #filters1[:, :, :, i][:,:,j] = 0
                if(par1[i] == 0)        :
                    biases1[i] = 0
                    filters1[:,:,:, i] = 0
                    
            model_final.layers[4].set_weights([filters1, biases1])
            arr = model_final.evaluate(test_data, test_label, verbose=1)
            print(par1)
            print('Filters left', np.sum(par1))
            print (arr)
            break
        print('new max', new_max)
        if(new_max > max1):
            max2 = max1
            par2 = np.copy(par1)
            par1 = np.copy(child)
            max1 = new_max
            print('max1', max1)
            print(par1)
        elif(new_max > max2):
            par2 = np.copy(child)
            max2 = new_max
            print('max2', max2)
    
    new_num = np.sum(par1)
           
    print(new_num)


    if(new_num > 28):
        olayer3_filters = new_num
    else:
        par1 = []
        for k in range(0, olayer3_filters):
            par1.append(1)
            
    A3 = np.copy(par1)

    ####################### 4th convolution layer with olayer4_filters filters
    print('4th convolution layer with olayer4_filters filters')
    A = []
    Acc = []
    
    arr = model_final.evaluate(test_data, test_label, verbose=1)
    print(arr)
    
    filters, biases = model_final.layers[5].get_weights()
    filters1 = np.copy(filters)
    biases1 = np.copy(biases)
    
    for index in range(0,10):
        # normalize filter values to 0-1 so we can visualize them
        # plot first few filters
        n_filters, ix = len(filters), 1
        B = []
        for j in range(0,olayer4_filters):
            x = random.randint(0,1)
            #print(x)
            B.append(x)
        
        for i in range(n_filters):
            f = filters[:, i]
            if(B[i] == 0):
                filters1[:,:,:, i] = 0
                biases1[i] = 0
    
    
        model_final.layers[5].set_weights([filters1, biases1])
        
        filters1 = np.copy(filters)
        biases1 = np.copy(biases)
    
        arr = model_final.evaluate(test_data, test_label, verbose=1)
        print (arr)
        if(arr[1] > 0.000010):
            A.append(B) 
            Acc.append(arr[1])
            print(index, 'added')
        
    
    max1 = 0
    max_index1 = 0
    index = 0
    while index < len(Acc):
        if(Acc[index]> max1):
            max1 = Acc[index]
            max_index1 = index
        index += 1
            
    Acc[max_index1] = -1
    max2 = 0
    max_index2 = 0
    index = 0
    while index < len(Acc):
        if(Acc[index]> max2):
            max2 = Acc[index]
            max_index2 = index
        index += 1
            
    
    par1 = np.copy(A[max_index1])
    par2 = np.copy(A[max_index2])
    print(max1, max2)
    temp_sat = 0
    for index_ga in range(0, 20):
        new_max = 0
        child = []
        temp_index = 0
        while (new_max < max1 and new_max < max2 and temp_index < 10):
            k = random.randint(10,100)
            
            child = np.copy(par1)
            #Crossover
            for index in range(k, olayer4_filters):
                child[index] = par2[index]
                
            #Mutation
            
            for l in range(0,5):
                temp_mut = random.randint(0,olayer4_filters-1)
                child[temp_mut] = 1-child[temp_mut]
                
                
            filters1 = np.copy(filters)
            biases1 = np.copy(biases)
            
            for i in range(0,olayer4_filters):
                f = filters[:,:,:, i]
                #for j in range(3):
                    #if(child[i] == 0):
                        #filters1[:, :, :, i][:,:,j] = 0
                if(child[i] == 0)        :
                    biases1[i] = 0
                    filters1[:,:,:, i] = 0
            
            model_final.layers[5].set_weights([filters1, biases1])
            arr = model_final.evaluate(test_data, test_label, verbose=1)
            new_max = 0.99*arr[1]+0.01*len(child)/np.sum(child)
            print (arr)
            temp_index += 1
        if(new_max >= max1 and new_max-max1 < 0.000001):
            temp_sat += 1
        if(temp_sat >0):
            filters1 = np.copy(filters)
            biases1 = np.copy(biases)
            
            for i in range(0,olayer4_filters):
                f = filters[:,:,:, i]
                #for j in range(3):
                    #if(par1[i] == 0):
                        #filters1[:, :, :, i][:,:,j] = 0
                if(par1[i] == 0)        :
                    biases1[i] = 0
                    filters1[:,:,:, i] = 0
                    
            model_final.layers[5].set_weights([filters1, biases1])
            arr = model_final.evaluate(test_data, test_label, verbose=1)
            print(par1)
            print('Filters left', np.sum(par1))
            print (arr)
            break
        print('new max', new_max)
        if(new_max > max1):
            max2 = max1
            par2 = np.copy(par1)
            par1 = np.copy(child)
            max1 = new_max
            print('max1', max1)
            print(par1)
        elif(new_max > max2):
            par2 = np.copy(child)
            max2 = new_max
            print('max2', max2)
    
    new_num = np.sum(par1)
           
    print(new_num)


    if(new_num > 28):
        olayer4_filters = new_num
    else:
        par1 = []
        for k in range(0, olayer4_filters):
            par1.append(1)
            
    A4 = np.copy(par1)
    
    ####################### 5th convolution layer with olayer5_filters filters
    print('5th convolution layer with olayer5_filters filters')
    A = []
    Acc = []
    
    arr = model_final.evaluate(test_data, test_label, verbose=1)
    print(arr)
    
    filters, biases = model_final.layers[7].get_weights()
    filters1 = np.copy(filters)
    biases1 = np.copy(biases)
    
    for index in range(0,10):
        # normalize filter values to 0-1 so we can visualize them
        # plot first few filters
        n_filters, ix = len(filters), 1
        B = []
        for j in range(0,olayer5_filters):
            x = random.randint(0,1)
            #print(x)
            B.append(x)
        
        for i in range(n_filters):
            f = filters[:, i]
            if(B[i] == 0):
                filters1[:,:,:, i] = 0
                biases1[i] = 0
    
    
        model_final.layers[7].set_weights([filters1, biases1])
        
        filters1 = np.copy(filters)
        biases1 = np.copy(biases)
    
        arr = model_final.evaluate(test_data, test_label, verbose=1)
        print (arr)
        if(arr[1] > 0.000010):
            A.append(B) 
            Acc.append(arr[1])
            print(index, 'added')
        
    
    max1 = 0
    max_index1 = 0
    index = 0
    while index < len(Acc):
        if(Acc[index]> max1):
            max1 = Acc[index]
            max_index1 = index
        index += 1
            
    Acc[max_index1] = -1
    max2 = 0
    max_index2 = 0
    index = 0
    while index < len(Acc):
        if(Acc[index]> max2):
            max2 = Acc[index]
            max_index2 = index
        index += 1
            
    
    par1 = np.copy(A[max_index1])
    par2 = np.copy(A[max_index2])
    print(max1, max2)
    temp_sat = 0
    for index_ga in range(0, 20):
        new_max = 0
        child = []
        temp_index = 0
        while (new_max < max1 and new_max < max2 and temp_index < 10):
            k = random.randint(10,100)
            
            child = np.copy(par1)
            #Crossover
            for index in range(k, olayer5_filters):
                child[index] = par2[index]
                
            #Mutation
            
            for l in range(0,5):
                temp_mut = random.randint(0,olayer5_filters-1)
                child[temp_mut] = 1-child[temp_mut]
                
                
            filters1 = np.copy(filters)
            biases1 = np.copy(biases)
            
            for i in range(0,olayer5_filters):
                f = filters[:,:,:, i]
                #for j in range(3):
                    #if(child[i] == 0):
                        #filters1[:, :, :, i][:,:,j] = 0
                if(child[i] == 0)        :
                    biases1[i] = 0
                    filters1[:,:,:, i] = 0
            
            model_final.layers[7].set_weights([filters1, biases1])
            arr = model_final.evaluate(test_data, test_label, verbose=1)
            new_max = 0.99*arr[1]+0.01*len(child)/np.sum(child)
            print (arr)
            temp_index += 1
        if(new_max >= max1 and new_max-max1 < 0.000001):
            temp_sat += 1
        if(temp_sat >0):
            filters1 = np.copy(filters)
            biases1 = np.copy(biases)
            
            for i in range(0,olayer5_filters):
                f = filters[:,:,:, i]
                #for j in range(3):
                    #if(par1[i] == 0):
                        #filters1[:, :, :, i][:,:,j] = 0
                if(par1[i] == 0)        :
                    biases1[i] = 0
                    filters1[:,:,:, i] = 0
                    
            model_final.layers[7].set_weights([filters1, biases1])
            arr = model_final.evaluate(test_data, test_label, verbose=1)
            print(par1)
            print('Filters left', np.sum(par1))
            print (arr)
            break
        print('new max', new_max)
        if(new_max > max1):
            max2 = max1
            par2 = np.copy(par1)
            par1 = np.copy(child)
            max1 = new_max
            print('max1', max1)
            print(par1)
        elif(new_max > max2):
            par2 = np.copy(child)
            max2 = new_max
            print('max2', max2)
    
    new_num = np.sum(par1)
           
    print(new_num)


    if(new_num > 28):
        olayer5_filters = new_num
    else:
        par1 = []
        for k in range(0, olayer5_filters):
            par1.append(1)
            
    A5 = np.copy(par1)
    
    ####################### 6th convolution layer with olayer6_filters filters
    print('6th convolution layer with olayer6_filters filters')
    A = []
    Acc = []
    
    arr = model_final.evaluate(test_data, test_label, verbose=1)
    print(arr)
    
    filters, biases = model_final.layers[8].get_weights()
    filters1 = np.copy(filters)
    biases1 = np.copy(biases)
    
    for index in range(0,10):
        # normalize filter values to 0-1 so we can visualize them
        # plot first few filters
        n_filters, ix = len(filters), 1
        B = []
        for j in range(0,olayer6_filters):
            x = random.randint(0,1)
            #print(x)
            B.append(x)
        
        for i in range(n_filters):
            f = filters[:, i]
            if(B[i] == 0):
                filters1[:,:,:, i] = 0
                biases1[i] = 0
    
    
        model_final.layers[8].set_weights([filters1, biases1])
        
        filters1 = np.copy(filters)
        biases1 = np.copy(biases)
    
        arr = model_final.evaluate(test_data, test_label, verbose=1)
        print (arr)
        if(arr[1] > 0.000010):
            A.append(B) 
            Acc.append(arr[1])
            print(index, 'added')
        
    
    max1 = 0
    max_index1 = 0
    index = 0
    while index < len(Acc):
        if(Acc[index]> max1):
            max1 = Acc[index]
            max_index1 = index
        index += 1
            
    Acc[max_index1] = -1
    max2 = 0
    max_index2 = 0
    index = 0
    while index < len(Acc):
        if(Acc[index]> max2):
            max2 = Acc[index]
            max_index2 = index
        index += 1
            
    
    par1 = np.copy(A[max_index1])
    par2 = np.copy(A[max_index2])
    print(max1, max2)
    temp_sat = 0
    for index_ga in range(0, 20):
        new_max = 0
        child = []
        temp_index = 0
        while (new_max < max1 and new_max < max2 and temp_index < 10):
            k = random.randint(10,100)
            
            child = np.copy(par1)
            #Crossover
            for index in range(k, olayer6_filters):
                child[index] = par2[index]
                
            #Mutation
            
            for l in range(0,5):
                temp_mut = random.randint(0,olayer6_filters-1)
                child[temp_mut] = 1-child[temp_mut]
                
                
            filters1 = np.copy(filters)
            biases1 = np.copy(biases)
            
            for i in range(0,olayer6_filters):
                f = filters[:,:,:, i]
                #for j in range(3):
                    #if(child[i] == 0):
                        #filters1[:, :, :, i][:,:,j] = 0
                if(child[i] == 0)        :
                    biases1[i] = 0
                    filters1[:,:,:, i] = 0
            
            model_final.layers[8].set_weights([filters1, biases1])
            arr = model_final.evaluate(test_data, test_label, verbose=1)
            new_max = 0.99*arr[1]+0.01*len(child)/np.sum(child)
            print (arr)
            temp_index += 1
        if(new_max >= max1 and new_max-max1 < 0.000001):
            temp_sat += 1
        if(temp_sat >0):
            filters1 = np.copy(filters)
            biases1 = np.copy(biases)
            
            for i in range(0,olayer6_filters):
                f = filters[:,:,:, i]
                #for j in range(3):
                    #if(par1[i] == 0):
                        #filters1[:, :, :, i][:,:,j] = 0
                if(par1[i] == 0)        :
                    biases1[i] = 0
                    filters1[:,:,:, i] = 0
                    
            model_final.layers[8].set_weights([filters1, biases1])
            arr = model_final.evaluate(test_data, test_label, verbose=1)
            print(par1)
            print('Filters left', np.sum(par1))
            print (arr)
            break
        print('new max', new_max)
        if(new_max > max1):
            max2 = max1
            par2 = np.copy(par1)
            par1 = np.copy(child)
            max1 = new_max
            print('max1', max1)
            print(par1)
        elif(new_max > max2):
            par2 = np.copy(child)
            max2 = new_max
            print('max2', max2)
    
    new_num = np.sum(par1)
           
    print(new_num)


    if(new_num > 28):
        olayer6_filters = new_num
    else:
        par1 = []
        for k in range(0, olayer6_filters):
            par1.append(1)
            
    A6 = np.copy(par1)

    ####################### 7th convolution layer with olayer7_filters filters
    print('7th convolution layer with olayer7_filters filters')
    A = []
    Acc = []
    
    arr = model_final.evaluate(test_data, test_label, verbose=1)
    print(arr)
    
    filters, biases = model_final.layers[9].get_weights()
    filters1 = np.copy(filters)
    biases1 = np.copy(biases)
    
    for index in range(0,10):
        # normalize filter values to 0-1 so we can visualize them
        # plot first few filters
        n_filters, ix = len(filters), 1
        B = []
        for j in range(0,olayer7_filters):
            x = random.randint(0,1)
            #print(x)
            B.append(x)
        
        for i in range(n_filters):
            f = filters[:, i]
            if(B[i] == 0):
                filters1[:,:,:, i] = 0
                biases1[i] = 0
    
    
        model_final.layers[9].set_weights([filters1, biases1])
        
        filters1 = np.copy(filters)
        biases1 = np.copy(biases)
    
        arr = model_final.evaluate(test_data, test_label, verbose=1)
        print (arr)
        if(arr[1] > 0.000010):
            A.append(B) 
            Acc.append(arr[1])
            print(index, 'added')
        
    
    max1 = 0
    max_index1 = 0
    index = 0
    while index < len(Acc):
        if(Acc[index]> max1):
            max1 = Acc[index]
            max_index1 = index
        index += 1
            
    Acc[max_index1] = -1
    max2 = 0
    max_index2 = 0
    index = 0
    while index < len(Acc):
        if(Acc[index]> max2):
            max2 = Acc[index]
            max_index2 = index
        index += 1
            
    
    par1 = np.copy(A[max_index1])
    par2 = np.copy(A[max_index2])
    print(max1, max2)
    temp_sat = 0
    for index_ga in range(0, 20):
        new_max = 0
        child = []
        temp_index = 0
        while (new_max < max1 and new_max < max2 and temp_index < 10):
            k = random.randint(10,100)
            
            child = np.copy(par1)
            #Crossover
            for index in range(k, olayer7_filters):
                child[index] = par2[index]
                
            #Mutation
            
            for l in range(0,5):
                temp_mut = random.randint(0,olayer7_filters-1)
                child[temp_mut] = 1-child[temp_mut]
                
                
            filters1 = np.copy(filters)
            biases1 = np.copy(biases)
            
            for i in range(0,olayer7_filters):
                f = filters[:,:,:, i]
                #for j in range(3):
                    #if(child[i] == 0):
                        #filters1[:, :, :, i][:,:,j] = 0
                if(child[i] == 0)        :
                    biases1[i] = 0
                    filters1[:,:,:, i] = 0
            
            model_final.layers[9].set_weights([filters1, biases1])
            arr = model_final.evaluate(test_data, test_label, verbose=1)
            new_max = 0.99*arr[1]+0.01*len(child)/np.sum(child)
            print (arr)
            temp_index += 1
        if(new_max >= max1 and new_max-max1 < 0.000001):
            temp_sat += 1
        if(temp_sat >0):
            filters1 = np.copy(filters)
            biases1 = np.copy(biases)
            
            for i in range(0,olayer7_filters):
                f = filters[:,:,:, i]
                #for j in range(3):
                    #if(par1[i] == 0):
                        #filters1[:, :, :, i][:,:,j] = 0
                if(par1[i] == 0)        :
                    biases1[i] = 0
                    filters1[:,:,:, i] = 0
                    
            model_final.layers[9].set_weights([filters1, biases1])
            arr = model_final.evaluate(test_data, test_label, verbose=1)
            print(par1)
            print('Filters left', np.sum(par1))
            print (arr)
            break
        print('new max', new_max)
        if(new_max > max1):
            max2 = max1
            par2 = np.copy(par1)
            par1 = np.copy(child)
            max1 = new_max
            print('max1', max1)
            print(par1)
        elif(new_max > max2):
            par2 = np.copy(child)
            max2 = new_max
            print('max2', max2)
    
    new_num = np.sum(par1)
           
    print(new_num)


    if(new_num > 28):
        olayer7_filters = new_num
    else:
        par1 = []
        for k in range(0, olayer7_filters):
            par1.append(1)
            
    A7 = np.copy(par1)
    
    ####################### 8th convolution layer with olayer8_filters filters
    print('8th convolution layer with olayer8_filters filters')
    A = []
    Acc = []
    
    arr = model_final.evaluate(test_data, test_label, verbose=1)
    print(arr)
    
    filters, biases = model_final.layers[11].get_weights()
    filters1 = np.copy(filters)
    biases1 = np.copy(biases)
    
    for index in range(0,10):
        # normalize filter values to 0-1 so we can visualize them
        # plot first few filters
        n_filters, ix = len(filters), 1
        B = []
        for j in range(0,olayer8_filters):
            x = random.randint(0,1)
            #print(x)
            B.append(x)
        
        for i in range(n_filters):
            f = filters[:, i]
            if(B[i] == 0):
                filters1[:,:,:, i] = 0
                biases1[i] = 0
    
    
        model_final.layers[11].set_weights([filters1, biases1])
        
        filters1 = np.copy(filters)
        biases1 = np.copy(biases)
    
        arr = model_final.evaluate(test_data, test_label, verbose=1)
        print (arr)
        if(arr[1] > 0.000010):
            A.append(B) 
            Acc.append(arr[1])
            print(index, 'added')
        
    
    max1 = 0
    max_index1 = 0
    index = 0
    while index < len(Acc):
        if(Acc[index]> max1):
            max1 = Acc[index]
            max_index1 = index
        index += 1
            
    Acc[max_index1] = -1
    max2 = 0
    max_index2 = 0
    index = 0
    while index < len(Acc):
        if(Acc[index]> max2):
            max2 = Acc[index]
            max_index2 = index
        index += 1
            
    
    par1 = np.copy(A[max_index1])
    par2 = np.copy(A[max_index2])
    print(max1, max2)
    temp_sat = 0
    for index_ga in range(0, 20):
        new_max = 0
        child = []
        temp_index = 0
        while (new_max < max1 and new_max < max2 and temp_index < 10):
            k = random.randint(10,100)
            
            child = np.copy(par1)
            #Crossover
            for index in range(k, olayer8_filters):
                child[index] = par2[index]
                
            #Mutation
            
            for l in range(0,5):
                temp_mut = random.randint(0,olayer8_filters-1)
                child[temp_mut] = 1-child[temp_mut]
                
                
            filters1 = np.copy(filters)
            biases1 = np.copy(biases)
            
            for i in range(0,olayer8_filters):
                f = filters[:,:,:, i]
                #for j in range(3):
                    #if(child[i] == 0):
                        #filters1[:, :, :, i][:,:,j] = 0
                if(child[i] == 0)        :
                    biases1[i] = 0
                    filters1[:,:,:, i] = 0
            
            model_final.layers[11].set_weights([filters1, biases1])
            arr = model_final.evaluate(test_data, test_label, verbose=1)
            new_max = 0.99*arr[1]+0.01*len(child)/np.sum(child)
            print (arr)
            temp_index += 1
        if(new_max >= max1 and new_max-max1 < 0.000001):
            temp_sat += 1
        if(temp_sat >0):
            filters1 = np.copy(filters)
            biases1 = np.copy(biases)
            
            for i in range(0,olayer8_filters):
                f = filters[:,:,:, i]
                #for j in range(3):
                    #if(par1[i] == 0):
                        #filters1[:, :, :, i][:,:,j] = 0
                if(par1[i] == 0)        :
                    biases1[i] = 0
                    filters1[:,:,:, i] = 0
                    
            model_final.layers[11].set_weights([filters1, biases1])
            arr = model_final.evaluate(test_data, test_label, verbose=1)
            print(par1)
            print('Filters left', np.sum(par1))
            print (arr)
            break
        print('new max', new_max)
        if(new_max > max1):
            max2 = max1
            par2 = np.copy(par1)
            par1 = np.copy(child)
            max1 = new_max
            print('max1', max1)
            print(par1)
        elif(new_max > max2):
            par2 = np.copy(child)
            max2 = new_max
            print('max2', max2)
    
    new_num = np.sum(par1)
           
    print(new_num)


    if(new_num > 28):
        olayer8_filters = new_num
    else:
        par1 = []
        for k in range(0, olayer8_filters):
            par1.append(1)
            
    A8 = np.copy(par1)
    
    ####################### 9th convolution layer with olayer9_filters filters
    print('9th convolution layer with olayer9_filters filters')
    A = []
    Acc = []
    
    arr = model_final.evaluate(test_data, test_label, verbose=1)
    print(arr)
    
    filters, biases = model_final.layers[12].get_weights()
    filters1 = np.copy(filters)
    biases1 = np.copy(biases)
    
    for index in range(0,10):
        # normalize filter values to 0-1 so we can visualize them
        # plot first few filters
        n_filters, ix = len(filters), 1
        B = []
        for j in range(0,olayer9_filters):
            x = random.randint(0,1)
            #print(x)
            B.append(x)
        
        for i in range(n_filters):
            f = filters[:, i]
            if(B[i] == 0):
                filters1[:,:,:, i] = 0
                biases1[i] = 0
    
    
        model_final.layers[12].set_weights([filters1, biases1])
        
        filters1 = np.copy(filters)
        biases1 = np.copy(biases)
    
        arr = model_final.evaluate(test_data, test_label, verbose=1)
        print (arr)
        if(arr[1] > 0.000010):
            A.append(B) 
            Acc.append(arr[1])
            print(index, 'added')
        
    
    max1 = 0
    max_index1 = 0
    index = 0
    while index < len(Acc):
        if(Acc[index]> max1):
            max1 = Acc[index]
            max_index1 = index
        index += 1
            
    Acc[max_index1] = -1
    max2 = 0
    max_index2 = 0
    index = 0
    while index < len(Acc):
        if(Acc[index]> max2):
            max2 = Acc[index]
            max_index2 = index
        index += 1
            
    
    par1 = np.copy(A[max_index1])
    par2 = np.copy(A[max_index2])
    print(max1, max2)
    temp_sat = 0
    for index_ga in range(0, 20):
        new_max = 0
        child = []
        temp_index = 0
        while (new_max < max1 and new_max < max2 and temp_index < 10):
            k = random.randint(10,100)
            
            child = np.copy(par1)
            #Crossover
            for index in range(k, olayer9_filters):
                child[index] = par2[index]
                
            #Mutation
            
            for l in range(0,5):
                temp_mut = random.randint(0,olayer9_filters-1)
                child[temp_mut] = 1-child[temp_mut]
                
                
            filters1 = np.copy(filters)
            biases1 = np.copy(biases)
            
            for i in range(0,olayer9_filters):
                f = filters[:,:,:, i]
                #for j in range(3):
                    #if(child[i] == 0):
                        #filters1[:, :, :, i][:,:,j] = 0
                if(child[i] == 0)        :
                    biases1[i] = 0
                    filters1[:,:,:, i] = 0
            
            model_final.layers[12].set_weights([filters1, biases1])
            arr = model_final.evaluate(test_data, test_label, verbose=1)
            new_max = 0.99*arr[1]+0.01*len(child)/np.sum(child)
            print (arr)
            temp_index += 1
        if(new_max >= max1 and new_max-max1 < 0.000001):
            temp_sat += 1
        if(temp_sat >0):
            filters1 = np.copy(filters)
            biases1 = np.copy(biases)
            
            for i in range(0,olayer9_filters):
                f = filters[:,:,:, i]
                #for j in range(3):
                    #if(par1[i] == 0):
                        #filters1[:, :, :, i][:,:,j] = 0
                if(par1[i] == 0)        :
                    biases1[i] = 0
                    filters1[:,:,:, i] = 0
                    
            model_final.layers[12].set_weights([filters1, biases1])
            arr = model_final.evaluate(test_data, test_label, verbose=1)
            print(par1)
            print('Filters left', np.sum(par1))
            print (arr)
            break
        print('new max', new_max)
        if(new_max > max1):
            max2 = max1
            par2 = np.copy(par1)
            par1 = np.copy(child)
            max1 = new_max
            print('max1', max1)
            print(par1)
        elif(new_max > max2):
            par2 = np.copy(child)
            max2 = new_max
            print('max2', max2)
    
    new_num = np.sum(par1)
           
    print(new_num)


    if(new_num > 28):
        olayer9_filters = new_num
    else:
        par1 = []
        for k in range(0, olayer9_filters):
            par1.append(1)
            
    A9 = np.copy(par1)
    
    ####################### 10th convolution layer with olayer10_filters filters
    print('10th convolution layer with olayer10_filters filters')
    A = []
    Acc = []
    
    arr = model_final.evaluate(test_data, test_label, verbose=1)
    print(arr)
    
    filters, biases = model_final.layers[13].get_weights()
    filters1 = np.copy(filters)
    biases1 = np.copy(biases)
    
    for index in range(0,10):
        # normalize filter values to 0-1 so we can visualize them
        # plot first few filters
        n_filters, ix = len(filters), 1
        B = []
        for j in range(0,olayer10_filters):
            x = random.randint(0,1)
            #print(x)
            B.append(x)
        
        for i in range(n_filters):
            f = filters[:, i]
            if(B[i] == 0):
                filters1[:,:,:, i] = 0
                biases1[i] = 0
    
    
        model_final.layers[13].set_weights([filters1, biases1])
        
        filters1 = np.copy(filters)
        biases1 = np.copy(biases)
    
        arr = model_final.evaluate(test_data, test_label, verbose=1)
        print (arr)
        if(arr[1] > 0.000010):
            A.append(B) 
            Acc.append(arr[1])
            print(index, 'added')
        
    
    max1 = 0
    max_index1 = 0
    index = 0
    while index < len(Acc):
        if(Acc[index]> max1):
            max1 = Acc[index]
            max_index1 = index
        index += 1
            
    Acc[max_index1] = -1
    max2 = 0
    max_index2 = 0
    index = 0
    while index < len(Acc):
        if(Acc[index]> max2):
            max2 = Acc[index]
            max_index2 = index
        index += 1
            
    
    par1 = np.copy(A[max_index1])
    par2 = np.copy(A[max_index2])
    print(max1, max2)
    temp_sat = 0
    for index_ga in range(0, 20):
        new_max = 0
        child = []
        temp_index = 0
        while (new_max < max1 and new_max < max2 and temp_index < 10):
            k = random.randint(10,100)
            
            child = np.copy(par1)
            #Crossover
            for index in range(k, olayer10_filters):
                child[index] = par2[index]
                
            #Mutation
            
            for l in range(0,5):
                temp_mut = random.randint(0,olayer10_filters-1)
                child[temp_mut] = 1-child[temp_mut]
                
                
            filters1 = np.copy(filters)
            biases1 = np.copy(biases)
            
            for i in range(0,olayer10_filters):
                f = filters[:,:,:, i]
                #for j in range(3):
                    #if(child[i] == 0):
                        #filters1[:, :, :, i][:,:,j] = 0
                if(child[i] == 0)        :
                    biases1[i] = 0
                    filters1[:,:,:, i] = 0
            
            model_final.layers[13].set_weights([filters1, biases1])
            arr = model_final.evaluate(test_data, test_label, verbose=1)
            new_max = 0.99*arr[1]+0.01*len(child)/np.sum(child)
            print (arr)
            temp_index += 1
        if(new_max >= max1 and new_max-max1 < 0.000001):
            temp_sat += 1
        if(temp_sat >0):
            filters1 = np.copy(filters)
            biases1 = np.copy(biases)
            
            for i in range(0,olayer10_filters):
                f = filters[:,:,:, i]
                #for j in range(3):
                    #if(par1[i] == 0):
                        #filters1[:, :, :, i][:,:,j] = 0
                if(par1[i] == 0)        :
                    biases1[i] = 0
                    filters1[:,:,:, i] = 0
                    
            model_final.layers[13].set_weights([filters1, biases1])
            arr = model_final.evaluate(test_data, test_label, verbose=1)
            print(par1)
            print('Filters left', np.sum(par1))
            print (arr)
            break
        print('new max', new_max)
        if(new_max > max1):
            max2 = max1
            par2 = np.copy(par1)
            par1 = np.copy(child)
            max1 = new_max
            print('max1', max1)
            print(par1)
        elif(new_max > max2):
            par2 = np.copy(child)
            max2 = new_max
            print('max2', max2)
    
    new_num = np.sum(par1)
           
    print(new_num)


    if(new_num > 28):
        olayer10_filters = new_num
    else:
        par1 = []
        for k in range(0, olayer10_filters):
            par1.append(1)
            
    A10 = np.copy(par1)
    
    ####################### 11th convolution layer with olayer11_filters filters
    print('11th convolution layer with olayer11_filters filters')
    A = []
    Acc = []
    
    arr = model_final.evaluate(test_data, test_label, verbose=1)
    print(arr)
    
    filters, biases = model_final.layers[15].get_weights()
    filters1 = np.copy(filters)
    biases1 = np.copy(biases)
    
    for index in range(0,10):
        # normalize filter values to 0-1 so we can visualize them
        # plot first few filters
        n_filters, ix = len(filters), 1
        B = []
        for j in range(0,olayer11_filters):
            x = random.randint(0,1)
            #print(x)
            B.append(x)
        
        for i in range(n_filters):
            f = filters[:, i]
            if(B[i] == 0):
                filters1[:,:,:, i] = 0
                biases1[i] = 0
    
    
        model_final.layers[15].set_weights([filters1, biases1])
        
        filters1 = np.copy(filters)
        biases1 = np.copy(biases)
    
        arr = model_final.evaluate(test_data, test_label, verbose=1)
        print (arr)
        if(arr[1] > 0.000010):
            A.append(B) 
            Acc.append(arr[1])
            print(index, 'added')
        
    
    max1 = 0
    max_index1 = 0
    index = 0
    while index < len(Acc):
        if(Acc[index]> max1):
            max1 = Acc[index]
            max_index1 = index
        index += 1
            
    Acc[max_index1] = -1
    max2 = 0
    max_index2 = 0
    index = 0
    while index < len(Acc):
        if(Acc[index]> max2):
            max2 = Acc[index]
            max_index2 = index
        index += 1
            
    
    par1 = np.copy(A[max_index1])
    par2 = np.copy(A[max_index2])
    print(max1, max2)
    temp_sat = 0
    for index_ga in range(0, 20):
        new_max = 0
        child = []
        temp_index = 0
        while (new_max < max1 and new_max < max2 and temp_index < 10):
            k = random.randint(10,100)
            
            child = np.copy(par1)
            #Crossover
            for index in range(k, olayer11_filters):
                child[index] = par2[index]
                
            #Mutation
            
            for l in range(0,5):
                temp_mut = random.randint(0,olayer11_filters-1)
                child[temp_mut] = 1-child[temp_mut]
                
                
            filters1 = np.copy(filters)
            biases1 = np.copy(biases)
            
            for i in range(0,olayer11_filters):
                f = filters[:,:,:, i]
                #for j in range(3):
                    #if(child[i] == 0):
                        #filters1[:, :, :, i][:,:,j] = 0
                if(child[i] == 0)        :
                    biases1[i] = 0
                    filters1[:,:,:, i] = 0
            
            model_final.layers[15].set_weights([filters1, biases1])
            arr = model_final.evaluate(test_data, test_label, verbose=1)
            new_max = 0.99*arr[1]+0.01*len(child)/np.sum(child)
            print (arr)
            temp_index += 1
        if(new_max >= max1 and new_max-max1 < 0.000001):
            temp_sat += 1
        if(temp_sat >0):
            filters1 = np.copy(filters)
            biases1 = np.copy(biases)
            
            for i in range(0,olayer11_filters):
                f = filters[:,:,:, i]
                #for j in range(3):
                    #if(par1[i] == 0):
                        #filters1[:, :, :, i][:,:,j] = 0
                if(par1[i] == 0)        :
                    biases1[i] = 0
                    filters1[:,:,:, i] = 0
                    
            model_final.layers[15].set_weights([filters1, biases1])
            arr = model_final.evaluate(test_data, test_label, verbose=1)
            print(par1)
            print('Filters left', np.sum(par1))
            print (arr)
            break
        print('new max', new_max)
        if(new_max > max1):
            max2 = max1
            par2 = np.copy(par1)
            par1 = np.copy(child)
            max1 = new_max
            print('max1', max1)
            print(par1)
        elif(new_max > max2):
            par2 = np.copy(child)
            max2 = new_max
            print('max2', max2)
    
    new_num = np.sum(par1)
           
    print(new_num)


    if(new_num > 28):
        olayer11_filters = new_num
    else:
        par1 = []
        for k in range(0, olayer11_filters):
            par1.append(1)
            
    A11 = np.copy(par1)
    
    ####################### 12th convolution layer with olayer12_filters filters
    print('12th convolution layer with olayer12_filters filters')
    A = []
    Acc = []
    
    arr = model_final.evaluate(test_data, test_label, verbose=1)
    print(arr)
    
    filters, biases = model_final.layers[16].get_weights()
    filters1 = np.copy(filters)
    biases1 = np.copy(biases)
    
    for index in range(0,10):
        # normalize filter values to 0-1 so we can visualize them
        # plot first few filters
        n_filters, ix = len(filters), 1
        B = []
        for j in range(0,olayer12_filters):
            x = random.randint(0,1)
            #print(x)
            B.append(x)
        
        for i in range(n_filters):
            f = filters[:, i]
            if(B[i] == 0):
                filters1[:,:,:, i] = 0
                biases1[i] = 0
    
    
        model_final.layers[16].set_weights([filters1, biases1])
        
        filters1 = np.copy(filters)
        biases1 = np.copy(biases)
    
        arr = model_final.evaluate(test_data, test_label, verbose=1)
        print (arr)
        if(arr[1] > 0.000010):
            A.append(B) 
            Acc.append(arr[1])
            print(index, 'added')
        
    
    max1 = 0
    max_index1 = 0
    index = 0
    while index < len(Acc):
        if(Acc[index]> max1):
            max1 = Acc[index]
            max_index1 = index
        index += 1
            
    Acc[max_index1] = -1
    max2 = 0
    max_index2 = 0
    index = 0
    while index < len(Acc):
        if(Acc[index]> max2):
            max2 = Acc[index]
            max_index2 = index
        index += 1
            
    
    par1 = np.copy(A[max_index1])
    par2 = np.copy(A[max_index2])
    print(max1, max2)
    temp_sat = 0
    for index_ga in range(0, 20):
        new_max = 0
        child = []
        temp_index = 0
        while (new_max < max1 and new_max < max2 and temp_index < 10):
            k = random.randint(10,100)
            
            child = np.copy(par1)
            #Crossover
            for index in range(k, olayer12_filters):
                child[index] = par2[index]
                
            #Mutation
            
            for l in range(0,5):
                temp_mut = random.randint(0,olayer12_filters-1)
                child[temp_mut] = 1-child[temp_mut]
                
                
            filters1 = np.copy(filters)
            biases1 = np.copy(biases)
            
            for i in range(0,olayer12_filters):
                f = filters[:,:,:, i]
                #for j in range(3):
                    #if(child[i] == 0):
                        #filters1[:, :, :, i][:,:,j] = 0
                if(child[i] == 0)        :
                    biases1[i] = 0
                    filters1[:,:,:, i] = 0
            
            model_final.layers[16].set_weights([filters1, biases1])
            arr = model_final.evaluate(test_data, test_label, verbose=1)
            new_max = 0.99*arr[1]+0.01*len(child)/np.sum(child)
            print (arr)
            temp_index += 1
        if(new_max >= max1 and new_max-max1 < 0.000001):
            temp_sat += 1
        if(temp_sat >0):
            filters1 = np.copy(filters)
            biases1 = np.copy(biases)
            
            for i in range(0,olayer12_filters):
                f = filters[:,:,:, i]
                #for j in range(3):
                    #if(par1[i] == 0):
                        #filters1[:, :, :, i][:,:,j] = 0
                if(par1[i] == 0)        :
                    biases1[i] = 0
                    filters1[:,:,:, i] = 0
                    
            model_final.layers[16].set_weights([filters1, biases1])
            arr = model_final.evaluate(test_data, test_label, verbose=1)
            print(par1)
            print('Filters left', np.sum(par1))
            print (arr)
            break
        print('new max', new_max)
        if(new_max > max1):
            max2 = max1
            par2 = np.copy(par1)
            par1 = np.copy(child)
            max1 = new_max
            print('max1', max1)
            print(par1)
        elif(new_max > max2):
            par2 = np.copy(child)
            max2 = new_max
            print('max2', max2)
    
    new_num = np.sum(par1)
           
    print(new_num)


    if(new_num > 28):
        olayer12_filters = new_num
    else:
        par1 = []
        for k in range(0, olayer12_filters):
            par1.append(1)
            
    A12 = np.copy(par1)
  
    ####################### 13th convolution layer with olayer13_filters filters
    print('13th convolution layer with olayer13_filters filters')
    A = []
    Acc = []
    
    arr = model_final.evaluate(test_data, test_label, verbose=1)
    print(arr)
    
    filters, biases = model_final.layers[17].get_weights()
    filters1 = np.copy(filters)
    biases1 = np.copy(biases)
    
    for index in range(0,10):
        # normalize filter values to 0-1 so we can visualize them
        # plot first few filters
        n_filters, ix = len(filters), 1
        B = []
        for j in range(0,olayer13_filters):
            x = random.randint(0,1)
            #print(x)
            B.append(x)
        
        for i in range(n_filters):
            f = filters[:, i]
            if(B[i] == 0):
                filters1[:,:,:, i] = 0
                biases1[i] = 0
    
    
        model_final.layers[17].set_weights([filters1, biases1])
        
        filters1 = np.copy(filters)
        biases1 = np.copy(biases)
    
        arr = model_final.evaluate(test_data, test_label, verbose=1)
        print (arr)
        if(arr[1] > 0.000010):
            A.append(B) 
            Acc.append(arr[1])
            print(index, 'added')
        
    
    max1 = 0
    max_index1 = 0
    index = 0
    while index < len(Acc):
        if(Acc[index]> max1):
            max1 = Acc[index]
            max_index1 = index
        index += 1
            
    Acc[max_index1] = -1
    max2 = 0
    max_index2 = 0
    index = 0
    while index < len(Acc):
        if(Acc[index]> max2):
            max2 = Acc[index]
            max_index2 = index
        index += 1
            
    
    par1 = np.copy(A[max_index1])
    par2 = np.copy(A[max_index2])
    print(max1, max2)
    temp_sat = 0
    for index_ga in range(0, 20):
        new_max = 0
        child = []
        temp_index = 0
        while (new_max < max1 and new_max < max2 and temp_index < 10):
            k = random.randint(10,100)
            
            child = np.copy(par1)
            #Crossover
            for index in range(k, olayer13_filters):
                child[index] = par2[index]
                
            #Mutation
            
            for l in range(0,5):
                temp_mut = random.randint(0,olayer13_filters-1)
                child[temp_mut] = 1-child[temp_mut]
                
                
            filters1 = np.copy(filters)
            biases1 = np.copy(biases)
            
            for i in range(0,olayer13_filters):
                f = filters[:,:,:, i]
                #for j in range(3):
                    #if(child[i] == 0):
                        #filters1[:, :, :, i][:,:,j] = 0
                if(child[i] == 0)        :
                    biases1[i] = 0
                    filters1[:,:,:, i] = 0
            
            model_final.layers[17].set_weights([filters1, biases1])
            arr = model_final.evaluate(test_data, test_label, verbose=1)
            new_max = 0.99*arr[1]+0.01*len(child)/np.sum(child)
            print (arr)
            temp_index += 1
        if(new_max >= max1 and new_max-max1 < 0.000001):
            temp_sat += 1
        if(temp_sat >0):
            filters1 = np.copy(filters)
            biases1 = np.copy(biases)
            
            for i in range(0,olayer13_filters):
                f = filters[:,:,:, i]
                #for j in range(3):
                    #if(par1[i] == 0):
                        #filters1[:, :, :, i][:,:,j] = 0
                if(par1[i] == 0)        :
                    biases1[i] = 0
                    filters1[:,:,:, i] = 0
                    
            model_final.layers[17].set_weights([filters1, biases1])
            arr = model_final.evaluate(test_data, test_label, verbose=1)
            print(par1)
            print('Filters left', np.sum(par1))
            print (arr)
            break
        print('new max', new_max)
        if(new_max > max1):
            max2 = max1
            par2 = np.copy(par1)
            par1 = np.copy(child)
            max1 = new_max
            print('max1', max1)
            print(par1)
        elif(new_max > max2):
            par2 = np.copy(child)
            max2 = new_max
            print('max2', max2)
    
    new_num = np.sum(par1)
           
    print(new_num)


    if(new_num > 28):
        olayer13_filters = new_num
    else:
        par1 = []
        for k in range(0, olayer13_filters):
            par1.append(1)
            
    A13 = np.copy(par1)
    
    ####################### 14th convolution layer with olayer14_filters filters
    print('14th convolution layer with olayer14_filters filters')
    A = []
    Acc = []
    
    arr = model_final.evaluate(test_data, test_label, verbose=1)
    print(arr)
    
    filters, biases = model_final.layers[19].get_weights()
    filters1 = np.copy(filters)
    biases1 = np.copy(biases)
    
    for index in range(0,10):
        # normalize filter values to 0-1 so we can visualize them
        # plot first few filters
        n_filters, ix = len(filters), 1
        B = []
        for j in range(0,olayer14_filters):
            x = random.randint(0,1)
            #print(x)
            B.append(x)
        
        for i in range(n_filters):
            f = filters[:, i]
            if(B[i] == 0):
                filters1[:,:,:, i] = 0
                biases1[i] = 0
    
    
        model_final.layers[19].set_weights([filters1, biases1])
        
        filters1 = np.copy(filters)
        biases1 = np.copy(biases)
    
        arr = model_final.evaluate(test_data, test_label, verbose=1)
        print (arr)
        if(arr[1] > 0.000010):
            A.append(B) 
            Acc.append(arr[1])
            print(index, 'added')
        
    
    max1 = 0
    max_index1 = 0
    index = 0
    while index < len(Acc):
        if(Acc[index]> max1):
            max1 = Acc[index]
            max_index1 = index
        index += 1
            
    Acc[max_index1] = -1
    max2 = 0
    max_index2 = 0
    index = 0
    while index < len(Acc):
        if(Acc[index]> max2):
            max2 = Acc[index]
            max_index2 = index
        index += 1
            
    
    par1 = np.copy(A[max_index1])
    par2 = np.copy(A[max_index2])
    print(max1, max2)
    temp_sat = 0
    for index_ga in range(0, 20):
        new_max = 0
        child = []
        temp_index = 0
        while (new_max < max1 and new_max < max2 and temp_index < 10):
            k = random.randint(10,100)
            
            child = np.copy(par1)
            #Crossover
            for index in range(k, olayer14_filters):
                child[index] = par2[index]
                
            #Mutation
            
            for l in range(0,5):
                temp_mut = random.randint(0,olayer14_filters-1)
                child[temp_mut] = 1-child[temp_mut]
                
                
            filters1 = np.copy(filters)
            biases1 = np.copy(biases)
            
            for i in range(0,olayer14_filters):
                f = filters[:,:,:, i]
                #for j in range(3):
                    #if(child[i] == 0):
                        #filters1[:, :, :, i][:,:,j] = 0
                if(child[i] == 0)        :
                    biases1[i] = 0
                    filters1[:,:,:, i] = 0
            
            model_final.layers[19].set_weights([filters1, biases1])
            arr = model_final.evaluate(test_data, test_label, verbose=1)
            new_max = 0.99*arr[1]+0.01*len(child)/np.sum(child)
            print (arr)
            temp_index += 1
        if(new_max >= max1 and new_max-max1 < 0.000001):
            temp_sat += 1
        if(temp_sat >0):
            filters1 = np.copy(filters)
            biases1 = np.copy(biases)
            
            for i in range(0,olayer14_filters):
                f = filters[:,:,:, i]
                #for j in range(3):
                    #if(par1[i] == 0):
                        #filters1[:, :, :, i][:,:,j] = 0
                if(par1[i] == 0)        :
                    biases1[i] = 0
                    filters1[:,:,:, i] = 0
                    
            model_final.layers[19].set_weights([filters1, biases1])
            arr = model_final.evaluate(test_data, test_label, verbose=1)
            print(par1)
            print('Filters left', np.sum(par1))
            print (arr)
            break
        print('new max', new_max)
        if(new_max > max1):
            max2 = max1
            par2 = np.copy(par1)
            par1 = np.copy(child)
            max1 = new_max
            print('max1', max1)
            print(par1)
        elif(new_max > max2):
            par2 = np.copy(child)
            max2 = new_max
            print('max2', max2)
    
    new_num = np.sum(par1)
           
    print(new_num)


    if(new_num > 28):
        olayer14_filters = new_num
    else:
        par1 = []
        for k in range(0, olayer14_filters):
            par1.append(1)
            
    A14 = np.copy(par1)
    ####################### 15th convolution layer with olayer15_filters filters
    print('15th convolution layer with olayer15_filters filters')
    A = []
    Acc = []
    
    arr = model_final.evaluate(test_data, test_label, verbose=1)
    print(arr)
    
    filters, biases = model_final.layers[21].get_weights()
    filters1 = np.copy(filters)
    biases1 = np.copy(biases)
    
    for index in range(0,10):
        # normalize filter values to 0-1 so we can visualize them
        # plot first few filters
        n_filters, ix = len(filters), 1
        B = []
        for j in range(0,olayer15_filters):
            x = random.randint(0,1)
            #print(x)
            B.append(x)
        
        for i in range(n_filters):
            f = filters[:, i]
            if(B[i] == 0):
                filters1[:,:,:, i] = 0
                biases1[i] = 0
    
    
        model_final.layers[21].set_weights([filters1, biases1])
        
        filters1 = np.copy(filters)
        biases1 = np.copy(biases)
    
        arr = model_final.evaluate(test_data, test_label, verbose=1)
        print (arr)
        if(arr[1] > 0.000010):
            A.append(B) 
            Acc.append(arr[1])
            print(index, 'added')
        
    
    max1 = 0
    max_index1 = 0
    index = 0
    while index < len(Acc):
        if(Acc[index]> max1):
            max1 = Acc[index]
            max_index1 = index
        index += 1
            
    Acc[max_index1] = -1
    max2 = 0
    max_index2 = 0
    index = 0
    while index < len(Acc):
        if(Acc[index]> max2):
            max2 = Acc[index]
            max_index2 = index
        index += 1
            
    
    par1 = np.copy(A[max_index1])
    par2 = np.copy(A[max_index2])
    print(max1, max2)
    temp_sat = 0
    for index_ga in range(0, 20):
        new_max = 0
        child = []
        temp_index = 0
        while (new_max < max1 and new_max < max2 and temp_index < 10):
            k = random.randint(10,100)
            
            child = np.copy(par1)
            #Crossover
            for index in range(k, olayer15_filters):
                child[index] = par2[index]
                
            #Mutation
            
            for l in range(0,5):
                temp_mut = random.randint(0,olayer15_filters-1)
                child[temp_mut] = 1-child[temp_mut]
                
                
            filters1 = np.copy(filters)
            biases1 = np.copy(biases)
            
            for i in range(0,olayer15_filters):
                f = filters[:,:,:, i]
                #for j in range(3):
                    #if(child[i] == 0):
                        #filters1[:, :, :, i][:,:,j] = 0
                if(child[i] == 0)        :
                    biases1[i] = 0
                    filters1[:,:,:, i] = 0
            
            model_final.layers[21].set_weights([filters1, biases1])
            arr = model_final.evaluate(test_data, test_label, verbose=1)
            new_max = 0.99*arr[1]+0.01*len(child)/np.sum(child)
            print (arr)
            temp_index += 1
        if(new_max >= max1 and new_max-max1 < 0.000001):
            temp_sat += 1
        if(temp_sat >0):
            filters1 = np.copy(filters)
            biases1 = np.copy(biases)
            
            for i in range(0,olayer15_filters):
                f = filters[:,:,:, i]
                #for j in range(3):
                    #if(par1[i] == 0):
                        #filters1[:, :, :, i][:,:,j] = 0
                if(par1[i] == 0)        :
                    biases1[i] = 0
                    filters1[:,:,:, i] = 0
                    
            model_final.layers[21].set_weights([filters1, biases1])
            arr = model_final.evaluate(test_data, test_label, verbose=1)
            print(par1)
            print('Filters left', np.sum(par1))
            print (arr)
            break
        print('new max', new_max)
        if(new_max > max1):
            max2 = max1
            par2 = np.copy(par1)
            par1 = np.copy(child)
            max1 = new_max
            print('max1', max1)
            print(par1)
        elif(new_max > max2):
            par2 = np.copy(child)
            max2 = new_max
            print('max2', max2)
    
    new_num = np.sum(par1)
           
    print(new_num)


    if(new_num > 28):
        olayer15_filters = new_num
    else:
        par1 = []
        for k in range(0, olayer15_filters):
            par1.append(1)
            
    A15 = np.copy(par1)

    
    model1 = FCN_Vgg16_32s_compress(olayer1_filters, olayer2_filters, olayer3_filters, olayer4_filters, olayer5_filters, olayer6_filters, olayer7_filters, olayer8_filters,olayer9_filters, olayer10_filters, olayer11_filters, olayer12_filters, olayer13_filters, olayer14_filters, olayer15_filters)

    print('Start')
    optimizer = SGD(lr=0.001, momentum=0.9, decay=0.0005, nesterov=False)
    model1.compile(loss="categorical_crossentropy", optimizer=optimizer, metrics=['accuracy'])
    print( 'Compiled: OK')
    model1.summary()
    
    layerr = model_final.layers[0].get_weights()
    model1.layers[0].set_weights(layerr)
    
    model = model_final
    ######################## 1st convolution layer with 128 filters
    filters, biases = model.layers[1].get_weights()
    filters1, biases1 = model1.layers[1].get_weights()
    print('Shape', filters.shape)
    print('Shape', filters1.shape)
    # normalize filter values to 0-1 so we can visualize them
    # plot first few filters
    n_filters, ix = nlayer1_filters, 1
    
    """
    for i in range(n_filters):
        f = filters[:, :, i]
    """
    index1 = 0
    # plot each channel separately
    for j in range(nlayer1_filters):
        if(A1[j] == 1) :
            """
            for i1 in range (0,3):
                for j1 in range(0,3):
                    filters1[:, :, index1][:,:,j][i1][j1] = filters[:, :, i][:,:,j][i1][j1]
            """
            filters1[:, :, :, index1] = filters[:, :, :, j]
            #biases1[:, :, :, index1] = biases[:, :, :, i]
            biases1[index1] = biases[j]
            #print(index1,i)
            index1 += 1
    #print(biases1, biases)        
    model1.layers[1].set_weights([filters1, biases1])
    
    ######################## 1st convolution layer with 256 filters
    filters, biases = model.layers[2].get_weights()
    filters1, biases1 = model1.layers[2].get_weights()
    print('Shape', filters.shape)
    print('Shape', filters1.shape)
    # normalize filter values to 0-1 so we can visualize them
    # plot first few filters
    n_filters, ix = nlayer2_filters, 1
    
    """
    for i in range(n_filters):
        f = filters[:, :, i]
    """
    index1 = 0
    # plot each channel separately
    for j in range(nlayer2_filters):
        if(A2[j] == 1) :
            index2 = 0
            for l in range(nlayer1_filters):
                if(A1[l] == 1):
                    filters1[:, :, index2, index1] = filters[:, :, l, j]
                    index2 += 1
            #biases1[:, :, :, index1] = biases[:, :, :, i]
            biases1[index1] = biases[j]
            #print(index1,i)
            index1 += 1
    #print(biases1, biases)        
    model1.layers[2].set_weights([filters1, biases1])
    
    ######################## 1st convolution layer with 512 filters
    filters, biases = model.layers[4].get_weights()
    filters1, biases1 = model1.layers[4].get_weights()
    print('Shape', filters.shape)
    print('Shape', filters1.shape)
    # normalize filter values to 0-1 so we can visualize them
    # plot first few filters
    n_filters, ix = nlayer3_filters, 1
    
    """
    for i in range(n_filters):
        f = filters[:, :, i]
    """
    index1 = 0
    # plot each channel separately
    for j in range(nlayer3_filters):
        if(A3[j] == 1) :
            index2 = 0
            for l in range(nlayer2_filters):
                if(A2[l] == 1):
                    filters1[:, :, index2, index1] = filters[:, :, l, j]
                    index2 += 1
            biases1[index1] = biases[j]
            #print(index1,i)
            index1 += 1
    #print(biases1, biases)        
    model1.layers[4].set_weights([filters1, biases1])
    
    ######################## 2nd convolution layer with 512 filters
    filters, biases = model.layers[5].get_weights()
    filters1, biases1 = model1.layers[5].get_weights()
    print('Shape', filters.shape)
    print('Shape', filters1.shape)
    # normalize filter values to 0-1 so we can visualize them
    # plot first few filters
    n_filters, ix = nlayer4_filters, 1
    
    """
    for i in range(n_filters):
        f = filters[:, :, i]
    """
    index1 = 0
    # plot each channel separately
    for j in range(nlayer4_filters):
        if(A4[j] == 1) :
            index2 = 0
            for l in range(nlayer3_filters):
                if(A3[l] == 1):
                    filters1[:, :, index2, index1] = filters[:, :, l, j]
                    index2 += 1
            #biases1[:, :, :, index1] = biases[:, :, :, i]
            biases1[index1] = biases[j]
            #print(index1,i)
            index1 += 1
    #print(biases1, biases)        
    model1.layers[5].set_weights([filters1, biases1])
    
    ######################## 1st dense layer with 1024 filters
    filters, biases = model.layers[7].get_weights()
    filters1, biases1 = model1.layers[7].get_weights()
    print('Shape', filters.shape)
    print('Shape', filters1.shape)
    # normalize filter values to 0-1 so we can visualize them
    # plot first few filters
    n_filters, ix = nlayer5_filters, 1
    
    """
    for i in range(n_filters):
        f = filters[:, :, i]
    """
    index1 = 0
    # plot each channel separately
    for j in range(nlayer5_filters):
        if(A5[j] == 1) :
            index2 = 0
            for l in range(nlayer4_filters):
                if(A4[l] == 1):
                    filters1[:, :, index2, index1] = filters[:, :, l, j]
                    index2 += 1
            #biases1[:, :, :, index1] = biases[:, :, :, i]
            biases1[index1] = biases[j]
            #print(index1,i)
            index1 += 1
    #print(biases1, biases)        
    model1.layers[7].set_weights([filters1, biases1])
    
    ######################## conv layer with 256 filters
    filters, biases = model.layers[8].get_weights()
    filters1, biases1 = model1.layers[8].get_weights()
    print('Shape', filters.shape)
    print('Shape', filters1.shape)
    # normalize filter values to 0-1 so we can visualize them
    # plot first few filters
    n_filters, ix = nlayer6_filters, 1
    
    """
    for i in range(n_filters):
        f = filters[:, :, i]
    """
    index1 = 0
    # plot each channel separately
    for j in range(nlayer6_filters):
        if(A6[j] == 1) :
            index2 = 0
            for l in range(nlayer5_filters):
                if(A5[l] == 1):
                    filters1[:, :, index2, index1] = filters[:, :, l, j]
                    index2 += 1
            #biases1[:, :, :, index1] = biases[:, :, :, i]
            biases1[index1] = biases[j]
            #print(index1,i)
            index1 += 1
    #print(biases1, biases)        
    model1.layers[8].set_weights([filters1, biases1])
    
    ######################## conv layer with 256 filters
    filters, biases = model.layers[9].get_weights()
    filters1, biases1 = model1.layers[9].get_weights()
    print('Shape', filters.shape)
    print('Shape', filters1.shape)
    # normalize filter values to 0-1 so we can visualize them
    # plot first few filters
    n_filters, ix = nlayer7_filters, 1
    
    """
    for i in range(n_filters):
        f = filters[:, :, i]
    """
    index1 = 0
    # plot each channel separately
    for j in range(nlayer7_filters):
        if(A7[j] == 1) :
            index2 = 0
            for l in range(nlayer6_filters):
                if(A6[l] == 1):
                    filters1[:, :, index2, index1] = filters[:, :, l, j]
                    index2 += 1
            #biases1[:, :, :, index1] = biases[:, :, :, i]
            biases1[index1] = biases[j]
            #print(index1,i)
            index1 += 1
    #print(biases1, biases)        
    model1.layers[9].set_weights([filters1, biases1])
    
    ######################## conv layer with 512 filters
    filters, biases = model.layers[11].get_weights()
    filters1, biases1 = model1.layers[11].get_weights()
    print('Shape', filters.shape)
    print('Shape', filters1.shape)
    # normalize filter values to 0-1 so we can visualize them
    # plot first few filters
    n_filters, ix = nlayer8_filters, 1
    
    """
    for i in range(n_filters):
        f = filters[:, :, i]
    """
    index1 = 0
    # plot each channel separately
    for j in range(nlayer8_filters):
        if(A8[j] == 1) :
            index2 = 0
            for l in range(nlayer7_filters):
                if(A7[l] == 1):
                    filters1[:, :, index2, index1] = filters[:, :, l, j]
                    index2 += 1
            #biases1[:, :, :, index1] = biases[:, :, :, i]
            biases1[index1] = biases[j]
            #print(index1,i)
            index1 += 1
    #print(biases1, biases)        
    model1.layers[11].set_weights([filters1, biases1])
    
    ######################## conv layer with 512 filters
    filters, biases = model.layers[12].get_weights()
    filters1, biases1 = model1.layers[12].get_weights()
    print('Shape', filters.shape)
    print('Shape', filters1.shape)
    # normalize filter values to 0-1 so we can visualize them
    # plot first few filters
    n_filters, ix = nlayer9_filters, 1
    
    """
    for i in range(n_filters):
        f = filters[:, :, i]
    """
    index1 = 0
    # plot each channel separately
    for j in range(nlayer9_filters):
        if(A9[j] == 1) :
            index2 = 0
            for l in range(nlayer8_filters):
                if(A8[l] == 1):
                    filters1[:, :, index2, index1] = filters[:, :, l, j]
                    index2 += 1
            #biases1[:, :, :, index1] = biases[:, :, :, i]
            biases1[index1] = biases[j]
            #print(index1,i)
            index1 += 1
    #print(biases1, biases)        
    model1.layers[12].set_weights([filters1, biases1])
    
    ######################## conv layer with 512 filters
    filters, biases = model.layers[13].get_weights()
    filters1, biases1 = model1.layers[13].get_weights()
    print('Shape', filters.shape)
    print('Shape', filters1.shape)
    # normalize filter values to 0-1 so we can visualize them
    # plot first few filters
    n_filters, ix = nlayer10_filters, 1
    
    """
    for i in range(n_filters):
        f = filters[:, :, i]
    """
    index1 = 0
    # plot each channel separately
    for j in range(nlayer10_filters):
        if(A10[j] == 1) :
            index2 = 0
            for l in range(nlayer9_filters):
                if(A9[l] == 1):
                    filters1[:, :, index2, index1] = filters[:, :, l, j]
                    index2 += 1
            #biases1[:, :, :, index1] = biases[:, :, :, i]
            biases1[index1] = biases[j]
            #print(index1,i)
            index1 += 1
    #print(biases1, biases)        
    model1.layers[13].set_weights([filters1, biases1])
     
    ######################## conv layer with 512 filters
    filters, biases = model.layers[15].get_weights()
    filters1, biases1 = model1.layers[15].get_weights()
    print('Shape', filters.shape)
    print('Shape', filters1.shape)
    # normalize filter values to 0-1 so we can visualize them
    # plot first few filters
    n_filters, ix = nlayer11_filters, 1
    
    """
    for i in range(n_filters):
        f = filters[:, :, i]
    """
    index1 = 0
    # plot each channel separately
    for j in range(nlayer11_filters):
        if(A11[j] == 1) :
            index2 = 0
            for l in range(nlayer10_filters):
                if(A10[l] == 1):
                    filters1[:, :, index2, index1] = filters[:, :, l, j]
                    index2 += 1
            #biases1[:, :, :, index1] = biases[:, :, :, i]
            biases1[index1] = biases[j]
            #print(index1,i)
            index1 += 1
    #print(biases1, biases)        
    model1.layers[15].set_weights([filters1, biases1])
    
    ######################## conv layer with 512 filters
    filters, biases = model.layers[16].get_weights()
    filters1, biases1 = model1.layers[16].get_weights()
    print('Shape', filters.shape)
    print('Shape', filters1.shape)
    # normalize filter values to 0-1 so we can visualize them
    # plot first few filters
    n_filters, ix = nlayer12_filters, 1
    
    """
    for i in range(n_filters):
        f = filters[:, :, i]
    """
    index1 = 0
    # plot each channel separately
    for j in range(nlayer12_filters):
        if(A12[j] == 1) :
            index2 = 0
            for l in range(nlayer11_filters):
                if(A11[l] == 1):
                    filters1[:, :, index2, index1] = filters[:, :, l, j]
                    index2 += 1
            #biases1[:, :, :, index1] = biases[:, :, :, i]
            biases1[index1] = biases[j]
            #print(index1,i)
            index1 += 1
    #print(biases1, biases)        
    model1.layers[16].set_weights([filters1, biases1])
    
    ######################## conv layer with 512 filters
    filters, biases = model.layers[17].get_weights()
    filters1, biases1 = model1.layers[17].get_weights()
    print('Shape', filters.shape)
    print('Shape', filters1.shape)
    # normalize filter values to 0-1 so we can visualize them
    # plot first few filters
    n_filters, ix = nlayer13_filters, 1
    
    """
    for i in range(n_filters):
        f = filters[:, :, i]
    """
    index1 = 0
    # plot each channel separately
    for j in range(nlayer13_filters):
        if(A13[j] == 1) :
            index2 = 0
            for l in range(nlayer12_filters):
                if(A12[l] == 1):
                    filters1[:, :, index2, index1] = filters[:, :, l, j]
                    index2 += 1
            #biases1[:, :, :, index1] = biases[:, :, :, i]
            biases1[index1] = biases[j]
            #print(index1,i)
            index1 += 1
    #print(biases1, biases)        
    model1.layers[17].set_weights([filters1, biases1])
    
    ######################## 2nd dense layer with 1024 filters
    filters, biases = model.layers[19].get_weights()
    filters1, biases1 = model1.layers[19].get_weights()
    print('Shape', filters.shape)
    print('Shape', filters1.shape)
    # normalize filter values to 0-1 so we can visualize them
    # plot first few filters
    n_filters, ix = nlayer14_filters, 1
    
    """
    for i in range(n_filters):
        f = filters[:, :, i]
    """
    index1 = 0
    # plot each channel separately
    for j in range(nlayer14_filters):
        if(A14[j] == 1) :
            index2 = 0
            for l in range(nlayer13_filters):
                if(A13[l] == 1):
                    filters1[:, :, index2, index1] = filters[:, :, l, j]
                    index2 += 1
            biases1[index1] = biases[j]
            #print(index1,i)
            index1 += 1
    #print(biases1, biases)        
    model1.layers[19].set_weights([filters1, biases1])
    
    ######################## 2nd dense layer with 1024 filters
    filters, biases = model.layers[21].get_weights()
    filters1, biases1 = model1.layers[21].get_weights()
    print('Shape', filters.shape)
    print('Shape', filters1.shape)
    # normalize filter values to 0-1 so we can visualize them
    # plot first few filters
    n_filters, ix = nlayer15_filters, 1
    
    """
    for i in range(n_filters):
        f = filters[:, :, i]
    """
    index1 = 0
    # plot each channel separately
    for j in range(nlayer15_filters):
        if(A15[j] == 1) :
            index2 = 0
            for l in range(nlayer14_filters):
                if(A14[l] == 1):
                    filters1[:, :, index2, index1] = filters[:, :, l, j]
                    index2 += 1
            biases1[index1] = biases[j]
            #print(index1,i)
            index1 += 1
    #print(biases1, biases)        
    model1.layers[21].set_weights([filters1, biases1])
    
    
    history = model1.fit(train_data, train_label, batch_size=batch_size, nb_epoch=50, verbose=1, validation_data=(val_data, val_label))
    
    #model1.load_weights('model_5l_weight_fcn_CamSeq2.hdf5')
    
    #test_data, test_label = prep_data1('test')
    score = model1.evaluate(test_data, test_label, verbose=1)
    print( 'Test score:', score[0])
    print( 'Test accuracy:', score[1])
    
    wt1 = "FCN_pruned_weights_" + str(withtrain+1) + ".h5"

    model1.save_weights(wt1)
    

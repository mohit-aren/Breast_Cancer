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
from tensorflow.keras.optimizers import SGD
#from tensorflow.keras.layers import Input, ZeroPadding2D
#from tensorflow.keras.layers import Activation, Flatten, Reshape
#from tensorflow.keras.layers import Convolution2D, MaxPooling2D, UpSampling2D
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.layers import concatenate  as merge
#from tensorflow.keras.layers import Input, merge, Conv2D, ZeroPadding2D, UpSampling2D, Dense, concatenate, Conv2DTranspose

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

    plt.savefig('result.JPG')
    plt.show()
"""

#########################################################################################################

from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Convolution2D, MaxPooling2D, UpSampling2D
from tensorflow.keras.layers import Activation, Reshape
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.models import Model
#from layers import MaxPoolingWithArgmax2D, MaxUnpooling2D


def SegNet(input_shape=(320, 320, 3), classes=2):
    # c.f. https://github.com/alexgkendall/SegNet-Tutorial/blob/master/Example_Models/bayesian_segnet_camvid.prototxt
    img_input = Input(shape=input_shape)
    x = img_input
    # Encoder
    x = Convolution2D(64, (3, 3), padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    
    x = Convolution2D(128, (3, 3), padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    
    x = Convolution2D(256, (3, 3), padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    
    x = Convolution2D(512, (3, 3), padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    
    # Decoder
    x = Convolution2D(512, (3, 3), padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    
    x = UpSampling2D(size=(2, 2))(x)
    x = Convolution2D(256, (3, 3), padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    
    x = UpSampling2D(size=(2, 2))(x)
    x = Convolution2D(128, (3, 3), padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    
    x = UpSampling2D(size=(2, 2))(x)
    x = Convolution2D(64, (3, 3), padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    
    x = Convolution2D(classes, 1, 1, padding="valid")(x)
    x = Reshape((input_shape[0]*input_shape[1], classes))(x)
    x = Activation("softmax")(x)
    model = Model(img_input, x)
    return model

def SegNet_compress(layer1_filters, layer2_filters, layer3_filters, layer4_filters, layer5_filters, layer6_filters, layer7_filters, layer8_filters, input_shape=(320, 320, 3), classes=2):
    # c.f. https://github.com/alexgkendall/SegNet-Tutorial/blob/master/Example_Models/bayesian_segnet_camvid.prototxt
    img_input = Input(shape=input_shape)
    x = img_input
    # Encoder
    x = Convolution2D(layer1_filters, (3, 3), padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    
    x = Convolution2D(layer2_filters, (3, 3), padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    
    x = Convolution2D(layer3_filters, (3, 3), padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    
    x = Convolution2D(layer4_filters, (3, 3), padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    
    # Decoder
    x = Convolution2D(layer5_filters, (3, 3), padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    
    x = UpSampling2D(size=(2, 2))(x)
    x = Convolution2D(layer6_filters, (3, 3), padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    
    x = UpSampling2D(size=(2, 2))(x)
    x = Convolution2D(layer7_filters, (3, 3), padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    
    x = UpSampling2D(size=(2, 2))(x)
    x = Convolution2D(layer8_filters, (3, 3), padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    
    x = Convolution2D(classes, 1, 1, padding="valid")(x)
    x = Reshape((input_shape[0]*input_shape[1], classes))(x)
    x = Activation("softmax")(x)
    model = Model(img_input, x)
    return model

"""
with open('model_5l.json') as model_file:
    autoencoder = models.model_from_json(model_file.read())
"""

autoencoder = SegNet()

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
autoencoder.load_weights('model_5l_weight_lungs_segnet.hdf5')

#autoencoder.load_weights('model_5l_weight_ep50.hdf5')


layer1_filters = 64
layer2_filters = 128
layer3_filters = 256
layer4_filters = 512
layer5_filters = 512
layer6_filters = 256
layer7_filters = 128
layer8_filters = 64


test_data, test_label = prep_data1('test')
#score = autoencoder.evaluate(test_data, test_label, verbose=1)
#print( 'Test score:', score[0])
#print( 'Test accuracy:', score[1])


for withtrain in range(0,10):
    
    autoencoder = SegNet_compress(layer1_filters, layer2_filters, layer3_filters, layer4_filters, layer5_filters, layer6_filters, layer7_filters, layer8_filters)
    
    print('Start')
    optimizer = SGD(lr=0.001, momentum=0.9, decay=0.0005, nesterov=False)
    autoencoder.compile(loss="categorical_crossentropy", optimizer=optimizer, metrics=['accuracy'])
    print( 'Compiled: OK')
    autoencoder.summary()
    wt = "SegNet_pruned_" + str(withtrain) + "ga.h5"
    autoencoder.load_weights(wt)
    
    nlayer1_filters = layer1_filters
    nlayer2_filters = layer2_filters
    nlayer3_filters = layer3_filters
    nlayer4_filters = layer4_filters
    nlayer5_filters = layer5_filters
    nlayer6_filters = layer6_filters
    nlayer7_filters = layer7_filters
    nlayer8_filters = layer8_filters

    ####################### 1st convolution layer with 64 filters
    print('1st convolution layer with 64 filters')
    A = []
    Acc = []
    
    arr = autoencoder.evaluate(test_data, test_label, verbose=1)
    print(arr)
    
    filters, biases = autoencoder.layers[1].get_weights()
    filters1 = np.copy(filters)
    biases1 = np.copy(biases)
    
    for index in range(0,10):
        # normalize filter values to 0-1 so we can visualize them
        # plot first few filters
        n_filters, ix = len(filters), 1
        B = []
        for j in range(0,layer1_filters):
            x = random.randint(0,1)
            #print(x)
            B.append(x)
        
        for i in range(n_filters):
            f = filters[:, i]
            if(B[i] == 0):
                filters1[:,:,:, i] = 0
                biases1[i] = 0
    
    
        autoencoder.layers[1].set_weights([filters1, biases1])
        
        filters1 = np.copy(filters)
        biases1 = np.copy(biases)
    
        arr = autoencoder.evaluate(test_data, test_label, verbose=1)
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
            for index in range(k, layer1_filters):
                child[index] = par2[index]
                
            #Mutation
            
            for l in range(0,5):
                temp_mut = random.randint(0,layer1_filters-1)
                child[temp_mut] = 1-child[temp_mut]
                
                
            filters1 = np.copy(filters)
            biases1 = np.copy(biases)
            
            for i in range(0,layer1_filters):
                f = filters[:,:,:, i]
                #for j in range(3):
                    #if(child[i] == 0):
                        #filters1[:, :, :, i][:,:,j] = 0
                if(child[i] == 0)        :
                    biases1[i] = 0
                    filters1[:,:,:, i] = 0
            
            autoencoder.layers[1].set_weights([filters1, biases1])
            arr = autoencoder.evaluate(test_data, test_label, verbose=1)
            new_max = 0.99*arr[1]+0.01*len(child)/np.sum(child)
            print (arr)
            temp_index += 1
        if(new_max >= max1 and new_max-max1 < 0.000001):
            temp_sat += 1
        if(temp_sat >0):
            filters1 = np.copy(filters)
            biases1 = np.copy(biases)
            
            for i in range(0,layer1_filters):
                f = filters[:,:,:, i]
                #for j in range(3):
                    #if(par1[i] == 0):
                        #filters1[:, :, :, i][:,:,j] = 0
                if(par1[i] == 0)        :
                    biases1[i] = 0
                    filters1[:,:,:, i] = 0
                    
            autoencoder.layers[1].set_weights([filters1, biases1])
            arr = autoencoder.evaluate(test_data, test_label, verbose=1)
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


    if(new_num > 16):
        layer1_filters = new_num
    else:
        par1 = []
        for k in range(0, layer1_filters):
            par1.append(1)
            
    A1= np.copy(par1)
    
    ####################### 1st convolution layer with 128 filters
    print('1st convolution layer with 128 filters')
    A = []
    Acc = []
    
    arr = autoencoder.evaluate(test_data, test_label, verbose=1)
    print(arr)
    
    filters, biases = autoencoder.layers[5].get_weights()
    filters1 = np.copy(filters)
    biases1 = np.copy(biases)
    
    for index in range(0,10):
        # normalize filter values to 0-1 so we can visualize them
        # plot first few filters
        n_filters, ix = len(filters), 1
        B = []
        for j in range(0,layer2_filters):
            x = random.randint(0,1)
            #print(x)
            B.append(x)
        
        for i in range(n_filters):
            f = filters[:, i]
            if(B[i] == 0):
                filters1[:,:,:, i] = 0
                biases1[i] = 0
    
    
        autoencoder.layers[5].set_weights([filters1, biases1])
        
        filters1 = np.copy(filters)
        biases1 = np.copy(biases)
    
        arr = autoencoder.evaluate(test_data, test_label, verbose=1)
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
            for index in range(k, layer2_filters):
                child[index] = par2[index]
                
            #Mutation
            
            for l in range(0,5):
                temp_mut = random.randint(0,layer2_filters-1)
                child[temp_mut] = 1-child[temp_mut]
                
                
            filters1 = np.copy(filters)
            biases1 = np.copy(biases)
            
            for i in range(0,layer2_filters):
                f = filters[:,:,:, i]
                #for j in range(3):
                    #if(child[i] == 0):
                        #filters1[:, :, :, i][:,:,j] = 0
                if(child[i] == 0)        :
                    biases1[i] = 0
                    filters1[:,:,:, i] = 0
            
            autoencoder.layers[5].set_weights([filters1, biases1])
            arr = autoencoder.evaluate(test_data, test_label, verbose=1)
            new_max = 0.99*arr[1]+0.01*len(child)/np.sum(child)
            print (arr)
            temp_index += 1
        if(new_max >= max1 and new_max-max1 < 0.000001):
            temp_sat += 1
        if(temp_sat >0):
            filters1 = np.copy(filters)
            biases1 = np.copy(biases)
            
            for i in range(0,layer2_filters):
                f = filters[:,:,:, i]
                #for j in range(3):
                    #if(par1[i] == 0):
                        #filters1[:, :, :, i][:,:,j] = 0
                if(par1[i] == 0)        :
                    biases1[i] = 0
                    filters1[:,:,:, i] = 0
                    
            autoencoder.layers[5].set_weights([filters1, biases1])
            arr = autoencoder.evaluate(test_data, test_label, verbose=1)
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


    if(new_num > 16):
        layer2_filters = new_num
    else:
        par1 = []
        for k in range(0, layer2_filters):
            par1.append(1)
            
    A2 = np.copy(par1)
    
    ####################### 1st convolution layer with 256 filters
    print('1st convolution layer with 256 filters')
    A = []
    Acc = []
    
    arr = autoencoder.evaluate(test_data, test_label, verbose=1)
    print(arr)
    
    filters, biases = autoencoder.layers[9].get_weights()
    filters1 = np.copy(filters)
    biases1 = np.copy(biases)
    
    for index in range(0,10):
        # normalize filter values to 0-1 so we can visualize them
        # plot first few filters
        n_filters, ix = len(filters), 1
        B = []
        for j in range(0,layer3_filters):
            x = random.randint(0,1)
            #print(x)
            B.append(x)
        
        for i in range(n_filters):
            f = filters[:, i]
            if(B[i] == 0):
                filters1[:,:,:, i] = 0
                biases1[i] = 0
    
    
        autoencoder.layers[9].set_weights([filters1, biases1])
        
        filters1 = np.copy(filters)
        biases1 = np.copy(biases)
    
        arr = autoencoder.evaluate(test_data, test_label, verbose=1)
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
            for index in range(k, layer3_filters):
                child[index] = par2[index]
                
            #Mutation
            
            for l in range(0,5):
                temp_mut = random.randint(0,layer3_filters-1)
                child[temp_mut] = 1-child[temp_mut]
                
                
            filters1 = np.copy(filters)
            biases1 = np.copy(biases)
            
            for i in range(0,layer3_filters):
                f = filters[:,:,:, i]
                #for j in range(3):
                    #if(child[i] == 0):
                        #filters1[:, :, :, i][:,:,j] = 0
                if(child[i] == 0)        :
                    biases1[i] = 0
                    filters1[:,:,:, i] = 0
            
            autoencoder.layers[9].set_weights([filters1, biases1])
            arr = autoencoder.evaluate(test_data, test_label, verbose=1)
            new_max = 0.99*arr[1]+0.01*len(child)/np.sum(child)
            print (arr)
            temp_index += 1
        if(new_max >= max1 and new_max-max1 < 0.000001):
            temp_sat += 1
        if(temp_sat >0):
            filters1 = np.copy(filters)
            biases1 = np.copy(biases)
            
            for i in range(0,layer3_filters):
                f = filters[:,:,:, i]
                #for j in range(3):
                    #if(par1[i] == 0):
                        #filters1[:, :, :, i][:,:,j] = 0
                if(par1[i] == 0)        :
                    biases1[i] = 0
                    filters1[:,:,:, i] = 0
                    
            autoencoder.layers[9].set_weights([filters1, biases1])
            arr = autoencoder.evaluate(test_data, test_label, verbose=1)
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


    if(new_num > 16):
        layer3_filters = new_num
    else:
        par1 = []
        for k in range(0, layer3_filters):
            par1.append(1)
            
    A3 = np.copy(par1)
    
    ####################### 1st convolution layer with 512 filters
    print('1st convolution layer with 512 filters')
    A = []
    Acc = []
    
    arr = autoencoder.evaluate(test_data, test_label, verbose=1)
    print(arr)
    
    filters, biases = autoencoder.layers[13].get_weights()
    filters1 = np.copy(filters)
    biases1 = np.copy(biases)
    
    for index in range(0,10):
        # normalize filter values to 0-1 so we can visualize them
        # plot first few filters
        n_filters, ix = len(filters), 1
        B = []
        for j in range(0,layer4_filters):
            x = random.randint(0,1)
            #print(x)
            B.append(x)
        
        for i in range(n_filters):
            f = filters[:, i]
            if(B[i] == 0):
                filters1[:,:,:, i] = 0
                biases1[i] = 0
    
    
        autoencoder.layers[13].set_weights([filters1, biases1])
        
        filters1 = np.copy(filters)
        biases1 = np.copy(biases)
    
        arr = autoencoder.evaluate(test_data, test_label, verbose=1)
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
            for index in range(k, layer4_filters):
                child[index] = par2[index]
                
            #Mutation
            
            for l in range(0,5):
                temp_mut = random.randint(0,layer4_filters-1)
                child[temp_mut] = 1-child[temp_mut]
                
                
            filters1 = np.copy(filters)
            biases1 = np.copy(biases)
            
            for i in range(0,layer4_filters):
                f = filters[:,:,:, i]
                #for j in range(3):
                    #if(child[i] == 0):
                        #filters1[:, :, :, i][:,:,j] = 0
                if(child[i] == 0)        :
                    biases1[i] = 0
                    filters1[:,:,:, i] = 0
            
            autoencoder.layers[13].set_weights([filters1, biases1])
            arr = autoencoder.evaluate(test_data, test_label, verbose=1)
            new_max = 0.99*arr[1]+0.01*len(child)/np.sum(child)
            print (arr)
            temp_index += 1
        if(new_max >= max1 and new_max-max1 < 0.000001):
            temp_sat += 1
        if(temp_sat >0):
            filters1 = np.copy(filters)
            biases1 = np.copy(biases)
            
            for i in range(0,layer4_filters):
                f = filters[:,:,:, i]
                #for j in range(3):
                    #if(par1[i] == 0):
                        #filters1[:, :, :, i][:,:,j] = 0
                if(par1[i] == 0)        :
                    biases1[i] = 0
                    filters1[:,:,:, i] = 0
                    
            autoencoder.layers[13].set_weights([filters1, biases1])
            arr = autoencoder.evaluate(test_data, test_label, verbose=1)
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


    if(new_num > 16):
        layer4_filters = new_num
    else:
        par1 = []
        for k in range(0, layer4_filters):
            par1.append(1)
            
    A4 = np.copy(par1)
    
    ####################### 2nd convolution layer with 512 filters
    print('2nd convolution layer with 512 filters')
    A = []
    Acc = []
    
    arr = autoencoder.evaluate(test_data, test_label, verbose=1)
    print(arr)
    
    filters, biases = autoencoder.layers[16].get_weights()
    filters1 = np.copy(filters)
    biases1 = np.copy(biases)
    
    for index in range(0,10):
        # normalize filter values to 0-1 so we can visualize them
        # plot first few filters
        n_filters, ix = len(filters), 1
        B = []
        for j in range(0,layer5_filters):
            x = random.randint(0,1)
            #print(x)
            B.append(x)
        
        for i in range(n_filters):
            f = filters[:, i]
            if(B[i] == 0):
                filters1[:,:,:, i] = 0
                biases1[i] = 0
    
    
        autoencoder.layers[16].set_weights([filters1, biases1])
        
        filters1 = np.copy(filters)
        biases1 = np.copy(biases)
    
        arr = autoencoder.evaluate(test_data, test_label, verbose=1)
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
            for index in range(k, layer5_filters):
                child[index] = par2[index]
                
            #Mutation
            
            for l in range(0,5):
                temp_mut = random.randint(0,layer5_filters-1)
                child[temp_mut] = 1-child[temp_mut]
                
                
            filters1 = np.copy(filters)
            biases1 = np.copy(biases)
            
            for i in range(0,layer5_filters):
                f = filters[:,:,:, i]
                #for j in range(3):
                    #if(child[i] == 0):
                        #filters1[:, :, :, i][:,:,j] = 0
                if(child[i] == 0)        :
                    biases1[i] = 0
                    filters1[:,:,:, i] = 0
            
            autoencoder.layers[16].set_weights([filters1, biases1])
            arr = autoencoder.evaluate(test_data, test_label, verbose=1)
            new_max = 0.99*arr[1]+0.01*len(child)/np.sum(child)
            print (arr)
            temp_index += 1
        if(new_max >= max1 and new_max-max1 < 0.000001):
            temp_sat += 1
        if(temp_sat >0):
            filters1 = np.copy(filters)
            biases1 = np.copy(biases)
            
            for i in range(0,layer5_filters):
                f = filters[:,:,:, i]
                #for j in range(3):
                    #if(par1[i] == 0):
                        #filters1[:, :, :, i][:,:,j] = 0
                if(par1[i] == 0)        :
                    biases1[i] = 0
                    filters1[:,:,:, i] = 0
                    
            autoencoder.layers[16].set_weights([filters1, biases1])
            arr = autoencoder.evaluate(test_data, test_label, verbose=1)
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


    if(new_num > 16):
        layer5_filters = new_num
    else:
        par1 = []
        for k in range(0, layer5_filters):
            par1.append(1)
            
    A5 = np.copy(par1)
    
    ####################### 2nd convolution layer with 256 filters
    print('2nd convolution layer with 256 filters')
    A = []
    Acc = []
    
    arr = autoencoder.evaluate(test_data, test_label, verbose=1)
    print(arr)
    
    filters, biases = autoencoder.layers[20].get_weights()
    filters1 = np.copy(filters)
    biases1 = np.copy(biases)
    
    for index in range(0,10):
        # normalize filter values to 0-1 so we can visualize them
        # plot first few filters
        n_filters, ix = len(filters), 1
        B = []
        for j in range(0,layer6_filters):
            x = random.randint(0,1)
            #print(x)
            B.append(x)
        
        for i in range(n_filters):
            f = filters[:, i]
            if(B[i] == 0):
                filters1[:,:,:, i] = 0
                biases1[i] = 0
    
    
        autoencoder.layers[20].set_weights([filters1, biases1])
        
        filters1 = np.copy(filters)
        biases1 = np.copy(biases)
    
        arr = autoencoder.evaluate(test_data, test_label, verbose=1)
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
            for index in range(k, layer6_filters):
                child[index] = par2[index]
                
            #Mutation
            
            for l in range(0,5):
                temp_mut = random.randint(0,layer6_filters-1)
                child[temp_mut] = 1-child[temp_mut]
                
                
            filters1 = np.copy(filters)
            biases1 = np.copy(biases)
            
            for i in range(0,layer6_filters):
                f = filters[:,:,:, i]
                #for j in range(3):
                    #if(child[i] == 0):
                        #filters1[:, :, :, i][:,:,j] = 0
                if(child[i] == 0)        :
                    biases1[i] = 0
                    filters1[:,:,:, i] = 0
            
            autoencoder.layers[20].set_weights([filters1, biases1])
            arr = autoencoder.evaluate(test_data, test_label, verbose=1)
            new_max = 0.99*arr[1]+0.01*len(child)/np.sum(child)
            print (arr)
            temp_index += 1
        if(new_max >= max1 and new_max-max1 < 0.000001):
            temp_sat += 1
        if(temp_sat >0):
            filters1 = np.copy(filters)
            biases1 = np.copy(biases)
            
            for i in range(0,layer6_filters):
                f = filters[:,:,:, i]
                #for j in range(3):
                    #if(par1[i] == 0):
                        #filters1[:, :, :, i][:,:,j] = 0
                if(par1[i] == 0)        :
                    biases1[i] = 0
                    filters1[:,:,:, i] = 0
                    
            autoencoder.layers[20].set_weights([filters1, biases1])
            arr = autoencoder.evaluate(test_data, test_label, verbose=1)
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


    if(new_num > 16):
        layer6_filters = new_num
    else:
        par1 = []
        for k in range(0, layer6_filters):
            par1.append(1)
            
    A6 = np.copy(par1)
    
    ####################### 2nd convolution layer with 128 filters
    print('2nd convolution layer with 128 filters')
    A = []
    Acc = []
    
    arr = autoencoder.evaluate(test_data, test_label, verbose=1)
    print(arr)
    
    filters, biases = autoencoder.layers[24].get_weights()
    filters1 = np.copy(filters)
    biases1 = np.copy(biases)
    
    for index in range(0,10):
        # normalize filter values to 0-1 so we can visualize them
        # plot first few filters
        n_filters, ix = len(filters), 1
        B = []
        for j in range(0,layer7_filters):
            x = random.randint(0,1)
            #print(x)
            B.append(x)
        
        for i in range(n_filters):
            f = filters[:, i]
            if(B[i] == 0):
                filters1[:,:,:, i] = 0
                biases1[i] = 0
    
    
        autoencoder.layers[24].set_weights([filters1, biases1])
        
        filters1 = np.copy(filters)
        biases1 = np.copy(biases)
    
        arr = autoencoder.evaluate(test_data, test_label, verbose=1)
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
            for index in range(k, layer7_filters):
                child[index] = par2[index]
                
            #Mutation
            
            for l in range(0,5):
                temp_mut = random.randint(0,layer7_filters-1)
                child[temp_mut] = 1-child[temp_mut]
                
                
            filters1 = np.copy(filters)
            biases1 = np.copy(biases)
            
            for i in range(0,layer7_filters):
                f = filters[:,:,:, i]
                #for j in range(3):
                    #if(child[i] == 0):
                        #filters1[:, :, :, i][:,:,j] = 0
                if(child[i] == 0)        :
                    biases1[i] = 0
                    filters1[:,:,:, i] = 0
            
            autoencoder.layers[24].set_weights([filters1, biases1])
            arr = autoencoder.evaluate(test_data, test_label, verbose=1)
            new_max = 0.99*arr[1]+0.01*len(child)/np.sum(child)
            print (arr)
            temp_index += 1
        if(new_max >= max1 and new_max-max1 < 0.000001):
            temp_sat += 1
        if(temp_sat >0):
            filters1 = np.copy(filters)
            biases1 = np.copy(biases)
            
            for i in range(0,layer7_filters):
                f = filters[:,:,:, i]
                #for j in range(3):
                    #if(par1[i] == 0):
                        #filters1[:, :, :, i][:,:,j] = 0
                if(par1[i] == 0)        :
                    biases1[i] = 0
                    filters1[:,:,:, i] = 0
                    
            autoencoder.layers[24].set_weights([filters1, biases1])
            arr = autoencoder.evaluate(test_data, test_label, verbose=1)
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


    if(new_num > 16):
        layer7_filters = new_num
    else:
        par1 = []
        for k in range(0, layer7_filters):
            par1.append(1)
            
    A7 = np.copy(par1)
    
    ####################### 2nd convolution layer with 64 filters
    print('2nd convolution layer with 64 filters')
    A = []
    Acc = []
    
    arr = autoencoder.evaluate(test_data, test_label, verbose=1)
    print(arr)
    
    filters, biases = autoencoder.layers[28].get_weights()
    filters1 = np.copy(filters)
    biases1 = np.copy(biases)
    
    for index in range(0,10):
        # normalize filter values to 0-1 so we can visualize them
        # plot first few filters
        n_filters, ix = len(filters), 1
        B = []
        for j in range(0,layer8_filters):
            x = random.randint(0,1)
            #print(x)
            B.append(x)
        
        for i in range(n_filters):
            f = filters[:, i]
            if(B[i] == 0):
                filters1[:,:,:, i] = 0
                biases1[i] = 0
    
    
        autoencoder.layers[28].set_weights([filters1, biases1])
        
        filters1 = np.copy(filters)
        biases1 = np.copy(biases)
    
        arr = autoencoder.evaluate(test_data, test_label, verbose=1)
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
            for index in range(k, layer8_filters):
                child[index] = par2[index]
                
            #Mutation
            
            for l in range(0,5):
                temp_mut = random.randint(0,layer8_filters-1)
                child[temp_mut] = 1-child[temp_mut]
                
                
            filters1 = np.copy(filters)
            biases1 = np.copy(biases)
            
            for i in range(0,layer8_filters):
                f = filters[:,:,:, i]
                #for j in range(3):
                    #if(child[i] == 0):
                        #filters1[:, :, :, i][:,:,j] = 0
                if(child[i] == 0)        :
                    biases1[i] = 0
                    filters1[:,:,:, i] = 0
            
            autoencoder.layers[28].set_weights([filters1, biases1])
            arr = autoencoder.evaluate(test_data, test_label, verbose=1)
            new_max = 0.99*arr[1]+0.01*len(child)/np.sum(child)
            print (arr)
            temp_index += 1
        if(new_max >= max1 and new_max-max1 < 0.000001):
            temp_sat += 1
        if(temp_sat >0):
            filters1 = np.copy(filters)
            biases1 = np.copy(biases)
            
            for i in range(0,layer8_filters):
                f = filters[:,:,:, i]
                #for j in range(3):
                    #if(par1[i] == 0):
                        #filters1[:, :, :, i][:,:,j] = 0
                if(par1[i] == 0)        :
                    biases1[i] = 0
                    filters1[:,:,:, i] = 0
                    
            autoencoder.layers[28].set_weights([filters1, biases1])
            arr = autoencoder.evaluate(test_data, test_label, verbose=1)
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


    if(new_num > 16):
        layer8_filters = new_num
    else:
        par1 = []
        for k in range(0, layer8_filters):
            par1.append(1)
            
    A8 = np.copy(par1)
    
    #################Compression##############3
    
    autoencoder_compress = SegNet_compress(layer1_filters, layer2_filters, layer3_filters, layer4_filters, layer5_filters, layer6_filters, layer7_filters, layer8_filters)
    
    print('Start')
    optimizer = SGD(lr=0.001, momentum=0.9, decay=0.0005, nesterov=False)
    autoencoder_compress.compile(loss="categorical_crossentropy", optimizer=optimizer, metrics=['accuracy'])
    print( 'Compiled: OK')
    autoencoder_compress.summary()
    
    for k in range(1, 2):
        if(len(autoencoder.layers[k].get_weights()) == 0):
            continue
        filters, biases = autoencoder.layers[k].get_weights()
        filters1, biases1 = autoencoder_compress.layers[k].get_weights()
        print('Shape', filters.shape)
        print('Shape', filters1.shape)
        # normalize filter values to 0-1 so we can visualize them
        # plot first few filters
        n_filters, ix = 64, 1
        
        index1 = 0
        # plot each channel separately
        for j in range(nlayer1_filters):
            if(A1[j] == 1) :
                filters1[:, :, :, index1] = filters[:, :, :, j]
                #biases1[:, :, :, index1] = biases[:, :, :, i]
                biases1[index1] = biases[j]
                #print(index1,i)
                index1 += 1
        #print(biases1, biases)        
        autoencoder_compress.layers[k].set_weights([filters1, biases1])
    
    for k in range(2,5):
        if(len(autoencoder.layers[k].get_weights()) == 0):
            continue
        
        A_1 = autoencoder.layers[k].get_weights()
        A_2 = autoencoder_compress.layers[k].get_weights()
        
        print('Len1', len(A_1))
        print('Len2', len(A_2))
        index1 = 0
        # plot each channel separately
        for j in range(nlayer1_filters):
            if(A1[j] == 1) :
                A_2[0][index1] = A_1[0][j]
                A_2[1][index1] = A_1[1][j]
                A_2[2][index1] = A_1[2][j]
                A_2[3][index1] = A_1[3][j]
                #print(index1,i)
                index1 += 1
        #print(biases1, biases)        
        autoencoder_compress.layers[k].set_weights(A_2)
    
    
    for k in range(5, 6):
        if(len(autoencoder.layers[k].get_weights()) == 0):
            continue
        filters, biases = autoencoder.layers[k].get_weights()
        filters1, biases1 = autoencoder_compress.layers[k].get_weights()
        print('Shape', filters.shape)
        print('Shape', filters1.shape)
        # normalize filter values to 0-1 so we can visualize them
        # plot first few filters
        n_filters, ix = 128, 1
        
        index1 = 0
        # plot each channel separately
        for j in range(nlayer2_filters):
            if(A2[j] == 1) :
                index2 = 0
                for l in range(nlayer1_filters):
                    if(A1[l] == 1):
                        filters1[:, :, index2, index1] = filters[:, :, l, j]
                        index2 += 1
                biases1[index1] = biases[j]
                #print(index1,i)
                index1 += 1
        #print(biases1, biases)        
        autoencoder_compress.layers[k].set_weights([filters1, biases1])
    
    for k in range(6,9):
        if(len(autoencoder.layers[k].get_weights()) == 0):
            continue
        
        A_1 = autoencoder.layers[k].get_weights()
        A_2 = autoencoder_compress.layers[k].get_weights()
        
        print('Len1', len(A_1))
        print('Len2', len(A_2))
        index1 = 0
        # plot each channel separately
        for j in range(nlayer2_filters):
            if(A2[j] == 1) :
                A_2[0][index1] = A_1[0][j]
                A_2[1][index1] = A_1[1][j]
                A_2[2][index1] = A_1[2][j]
                A_2[3][index1] = A_1[3][j]
                #print(index1,i)
                index1 += 1
        #print(biases1, biases)        
        autoencoder_compress.layers[k].set_weights(A_2)
    
    for k in range(9, 10):
        if(len(autoencoder.layers[k].get_weights()) == 0):
            continue
        filters, biases = autoencoder.layers[k].get_weights()
        filters1, biases1 = autoencoder_compress.layers[k].get_weights()
        print('Shape', filters.shape)
        print('Shape', filters1.shape)
        # normalize filter values to 0-1 so we can visualize them
        # plot first few filters
        n_filters, ix = 256, 1
        
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
        autoencoder_compress.layers[k].set_weights([filters1, biases1])
    
    for k in range(10,13):
        if(len(autoencoder.layers[k].get_weights()) == 0):
            continue
        
        A_1 = autoencoder.layers[k].get_weights()
        A_2 = autoencoder_compress.layers[k].get_weights()
        
        print('Len1', len(A_1))
        print('Len2', len(A_2))
        index1 = 0
        # plot each channel separately
        for j in range(nlayer3_filters):
            if(A3[j] == 1) :
                A_2[0][index1] = A_1[0][j]
                A_2[1][index1] = A_1[1][j]
                A_2[2][index1] = A_1[2][j]
                A_2[3][index1] = A_1[3][j]
                #print(index1,i)
                index1 += 1
        #print(biases1, biases)        
        autoencoder_compress.layers[k].set_weights(A_2)
    
    for k in range(13, 14):
        if(len(autoencoder.layers[k].get_weights()) == 0):
            continue
        filters, biases = autoencoder.layers[k].get_weights()
        filters1, biases1 = autoencoder_compress.layers[k].get_weights()
        print('Shape', filters.shape)
        print('Shape', filters1.shape)
        # normalize filter values to 0-1 so we can visualize them
        # plot first few filters
        n_filters, ix = 512, 1
        
        index1 = 0
        # plot each channel separately
        for j in range(nlayer4_filters):
            if(A4[j] == 1) :
                index2 = 0
                for l in range(nlayer3_filters):
                    if(A3[l] == 1):
                        filters1[:, :, index2, index1] = filters[:, :, l, j]
                        index2 += 1
                biases1[index1] = biases[j]
                #print(index1,i)
                index1 += 1
        #print(biases1, biases)        
        autoencoder_compress.layers[k].set_weights([filters1, biases1])
    
    for k in range(14,16):
        if(len(autoencoder.layers[k].get_weights()) == 0):
            continue
        
        A_1 = autoencoder.layers[k].get_weights()
        A_2 = autoencoder_compress.layers[k].get_weights()
        
        print('Len1', len(A_1))
        print('Len2', len(A_2))
        index1 = 0
        # plot each channel separately
        for j in range(nlayer4_filters):
            if(A4[j] == 1) :
                A_2[0][index1] = A_1[0][j]
                A_2[1][index1] = A_1[1][j]
                A_2[2][index1] = A_1[2][j]
                A_2[3][index1] = A_1[3][j]
                #print(index1,i)
                index1 += 1
        #print(biases1, biases)        
        autoencoder_compress.layers[k].set_weights(A_2)
    
    for k in range(16, 17):
        if(len(autoencoder.layers[k].get_weights()) == 0):
            continue
        filters, biases = autoencoder.layers[k].get_weights()
        filters1, biases1 = autoencoder_compress.layers[k].get_weights()
        print('Shape', filters.shape)
        print('Shape', filters1.shape)
        # normalize filter values to 0-1 so we can visualize them
        # plot first few filters
        n_filters, ix = 512, 1
        
        index1 = 0
        # plot each channel separately
        for j in range(nlayer5_filters):
            if(A5[j] == 1) :
                index2 = 0
                for l in range(nlayer4_filters):
                    if(A4[l] == 1):
                        filters1[:, :, index2, index1] = filters[:, :, l, j]
                        index2 += 1
                biases1[index1] = biases[j]
                #print(index1,i)
                index1 += 1
        #print(biases1, biases)        
        autoencoder_compress.layers[k].set_weights([filters1, biases1])
    
    for k in range(17,20):
        if(len(autoencoder.layers[k].get_weights()) == 0):
            continue
        
        A_1 = autoencoder.layers[k].get_weights()
        A_2 = autoencoder_compress.layers[k].get_weights()
        
        print('Len1', len(A_1))
        print('Len2', len(A_2))
        index1 = 0
        # plot each channel separately
        for j in range(nlayer5_filters):
            if(A5[j] == 1) :
                A_2[0][index1] = A_1[0][j]
                A_2[1][index1] = A_1[1][j]
                A_2[2][index1] = A_1[2][j]
                A_2[3][index1] = A_1[3][j]
                #print(index1,i)
                index1 += 1
        #print(biases1, biases)        
        autoencoder_compress.layers[k].set_weights(A_2)
    
    for k in range(20, 21):
        if(len(autoencoder.layers[k].get_weights()) == 0):
            continue
        filters, biases = autoencoder.layers[k].get_weights()
        filters1, biases1 = autoencoder_compress.layers[k].get_weights()
        print('Shape', filters.shape)
        print('Shape', filters1.shape)
        # normalize filter values to 0-1 so we can visualize them
        # plot first few filters
        n_filters, ix = 256, 1
        
        index1 = 0
        # plot each channel separately
        for j in range(nlayer6_filters):
            if(A6[j] == 1) :
                index2 = 0
                for l in range(nlayer5_filters):
                    if(A5[l] == 1):
                        filters1[:, :, index2, index1] = filters[:, :, l, j]
                        index2 += 1
                biases1[index1] = biases[j]
                #print(index1,i)
                index1 += 1
        #print(biases1, biases)        
        autoencoder_compress.layers[k].set_weights([filters1, biases1])
    
    for k in range(21,24):
        if(len(autoencoder.layers[k].get_weights()) == 0):
            continue
        
        A_1 = autoencoder.layers[k].get_weights()
        A_2 = autoencoder_compress.layers[k].get_weights()
        
        print('Len1', len(A_1))
        print('Len2', len(A_2))
        index1 = 0
        # plot each channel separately
        for j in range(nlayer6_filters):
            if(A6[j] == 1) :
                A_2[0][index1] = A_1[0][j]
                A_2[1][index1] = A_1[1][j]
                A_2[2][index1] = A_1[2][j]
                A_2[3][index1] = A_1[3][j]
                #print(index1,i)
                index1 += 1
        #print(biases1, biases)        
        autoencoder_compress.layers[k].set_weights(A_2)
    
    for k in range(24, 25):
        if(len(autoencoder.layers[k].get_weights()) == 0):
            continue
        filters, biases = autoencoder.layers[k].get_weights()
        filters1, biases1 = autoencoder_compress.layers[k].get_weights()
        print('Shape', filters.shape)
        print('Shape', filters1.shape)
        # normalize filter values to 0-1 so we can visualize them
        # plot first few filters
        n_filters, ix = 128, 1
        
        index1 = 0
        # plot each channel separately
        for j in range(nlayer7_filters):
            if(A7[j] == 1) :
                index2 = 0
                for l in range(nlayer6_filters):
                    if(A6[l] == 1):
                        filters1[:, :, index2, index1] = filters[:, :, l, j]
                        index2 += 1
                biases1[index1] = biases[j]
                #print(index1,i)
                index1 += 1
        #print(biases1, biases)        
        autoencoder_compress.layers[k].set_weights([filters1, biases1])
    
    for k in range(25,28):
        if(len(autoencoder.layers[k].get_weights()) == 0):
            continue
        
        A_1 = autoencoder.layers[k].get_weights()
        A_2 = autoencoder_compress.layers[k].get_weights()
        
        print('Len1', len(A_1))
        print('Len2', len(A_2))
        index1 = 0
        # plot each channel separately
        for j in range(nlayer7_filters):
            if(A7[j] == 1) :
                A_2[0][index1] = A_1[0][j]
                A_2[1][index1] = A_1[1][j]
                A_2[2][index1] = A_1[2][j]
                A_2[3][index1] = A_1[3][j]
                #print(index1,i)
                index1 += 1
        #print(biases1, biases)        
        autoencoder_compress.layers[k].set_weights(A_2)
    
    for k in range(28, 29):
        if(len(autoencoder.layers[k].get_weights()) == 0):
            continue
        filters, biases = autoencoder.layers[k].get_weights()
        filters1, biases1 = autoencoder_compress.layers[k].get_weights()
        print('Shape', filters.shape)
        print('Shape', filters1.shape)
        # normalize filter values to 0-1 so we can visualize them
        # plot first few filters
        n_filters, ix = 64, 1
        
        index1 = 0
        # plot each channel separately
        for j in range(nlayer8_filters):
            if(A8[j] == 1) :
                index2 = 0
                for l in range(nlayer7_filters):
                    if(A7[l] == 1):
                        filters1[:, :, index2, index1] = filters[:, :, l, j]
                        index2 += 1
                biases1[index1] = biases[j]
                #print(index1,i)
                index1 += 1
        #print(biases1, biases)        
        autoencoder_compress.layers[k].set_weights([filters1, biases1])
    
    for k in range(29,31):
        if(len(autoencoder.layers[k].get_weights()) == 0):
            continue
        
        A_1 = autoencoder.layers[k].get_weights()
        A_2 = autoencoder_compress.layers[k].get_weights()
        
        print('Len1', len(A_1))
        print('Len2', len(A_2))
        index1 = 0
        # plot each channel separately
        for j in range(nlayer8_filters):
            if(A8[j] == 1) :
                A_2[0][index1] = A_1[0][j]
                A_2[1][index1] = A_1[1][j]
                A_2[2][index1] = A_1[2][j]
                A_2[3][index1] = A_1[3][j]
                #print(index1,i)
                index1 += 1
        #print(biases1, biases)        
        autoencoder_compress.layers[k].set_weights(A_2)
    
    #train_data, train_label = prep_data1('train')
    nb_epoch = 50
    batch_size = 8
    history = autoencoder_compress.fit(train_data, train_label, batch_size=batch_size, nb_epoch=nb_epoch, verbose=1)
    wt = "SegNet_pruned_" + str(withtrain+1) + "ga.h5"
    
    autoencoder_compress.save_weights(wt)
    
    # Model visualization
    #from keras.utils.visualize_util import plot
    #plot(autoencoder, to_file='model.JPG', show_shapes=True)
    
    score = autoencoder_compress.evaluate(test_data, test_label, verbose=1)
    print( 'Test score:', score[0])
    print( 'Test accuracy:', score[1])
    
    output = autoencoder.predict(test_data, verbose=1)
    output = output.reshape((output.shape[0], img_h, img_w, n_labels))
    
    #plot_results(output)


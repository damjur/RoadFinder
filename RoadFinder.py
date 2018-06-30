
# coding: utf-8

# In[1]:



from bs4 import BeautifulSoup
import urllib
import cv2
import matplotlib.pyplot as plt
import numpy as np
import pickle
from tqdm import tqdm
import os

import keras
from keras.models import Model
from keras.layers import Dense, Dropout, Flatten, Input, concatenate, Reshape, BatchNormalization
from keras.layers import Conv2D, MaxPooling2D,UpSampling2D,Lambda, ZeroPadding2D, Activation
from keras import backend as K
from sklearn.model_selection import train_test_split
from keras.regularizers import l1_l2
from keras.utils import to_categorical


import gc
import re

np.errstate(invalid='raise')

class MyException(Exception):
    pass


# In[3]:


HOWMANY = 1100
MAXLINKS = 1109
DEBUG = True
TMP_DIR = 'D:\\tmp'
FORCE_RELOAD = False#True
LOAD = True
PREPROCESS = True#False
batch_size = 32   # ile obrazków przetwarzamy na raz (aktualizacja wag sieci następuje raz na całą grupę obrazków)
epochs = 24         # ile epok będziemy uczyli
SIZE = (750,750)
SIDE = 75
IMPOSITION = 5
HOWMANYPERIMAGE = int(SIZE[0]*SIZE[1]/SIDE/SIDE)
IMAGESPERFILE = 100
CLASS_ZERO = -1
assert int(SIZE[0]*SIZE[1]/SIDE/SIDE)==HOWMANYPERIMAGE


# In[4]:


def loadImage(url):
    raw = urllib.request.urlopen(url).read()
    npraw= np.array(bytearray(raw),dtype=np.uint8)
    return cv2.imdecode(npraw,-1)#-1 -> as is (with the alpha channel)

def getImageName(url):
    return url.split('/').pop().split('.').pop(0)

def pickleBigDataset(prefix,dataset,size):
    j = int(np.ceil(len(dataset)/size))
    # b = getNumber(prefix)
    # for i in range(b,j+b):
        # np.save(os.path.join(TMP_DIR, prefix+str(i)),np.array(dataset[size*(i-1):size*i]))
        # dataset = dataset[size*i:]
        # gc.collect()
    # if len(dataset)>0:
        # np.save(os.path.join(TMP_DIR, prefix+str(j+b)),np.array(dataset))
        # dataset = []
        # gc.collect()
    for i in range(1,j+1):
        np.save(os.path.join(TMP_DIR, prefix+str(i)),np.array(dataset[size*(i-1):size*i]))
        
def getNumber(prefix):
    d = np.dstack(([int(re.findall('\\d+',f)[0]) for f in os.listdir(TMP_DIR) if os.path.isfile(os.path.join(TMP_DIR, f)) and f.startswith(prefix) and f.endswith('.npy')],
    [int(os.path.getsize(os.path.join(TMP_DIR, f))) for f in os.listdir(TMP_DIR) if os.path.isfile(os.path.join(TMP_DIR, f)) and f.startswith(prefix)]))[0]
    return d[d[:,1]==d[:,1].max()][:,0].max()+1 

def unpickleBigDataset(prefix):
    onlyfiles = [f for f in os.listdir(TMP_DIR) if os.path.isfile(os.path.join(TMP_DIR, f))
                 and f.startswith(prefix)]
    dataset = []
    if len(onlyfiles)>0:
        print("Loading...")
        dataset = [x for x in np.load(os.path.join(TMP_DIR, onlyfiles[0]))]
        print("Loaded first")
        for f in tqdm(onlyfiles[1:]):
            dataset += [x for x in np.load(os.path.join(TMP_DIR, f))]
    return dataset
#     return np.load(os.path.join(TMP_DIR, "{}.npy".format(prefix)))
    
            
def loadImagesFromSite(url,prefix):
    onlyfiles = [f for f in os.listdir(TMP_DIR) if os.path.isfile(os.path.join(TMP_DIR, f)) and f.startswith(prefix)]
    if len(onlyfiles)==0 or FORCE_RELOAD:
        imgs = []
        I = None
        
    else:
        imgs = [img for img in unpickleBigDataset(prefix)[:HOWMANY]]
        I = len(imgs)
    print("Cached images {}.".format(I if I is not None else 0))
    
    if (HOWMANY is not None and len(imgs)<HOWMANY and len(imgs)<MAXLINKS) or (len(imgs)<MAXLINKS and HOWMANY is None):
        print("Loading images from {}".format(url))
        print("Proceeding from {} image.".format(I if I is not None else 0))

        s = IMAGESPERFILE

        with urllib.request.urlopen(url) as response:
            html = BeautifulSoup(response.read(),"lxml")
            i = I if I is not None else 0
            links = html.find_all('a')[I:HOWMANY]
            for link in tqdm(links):
                img = loadImage(link.get('href'))  
                img = cv2.resize(img,SIZE)
#                 print(link.get('href'))
                imgs += [cv2.resize(img,SIZE)]
                if i%s==0:
                    pickleBigDataset(prefix,imgs,s)
                i+=1
        pickleBigDataset(prefix,imgs,s)
    
        
    return np.array(imgs)  

def saveDataset(X,Y,prefix=""):
    with open('pickledDatasetX'+prefix,'wb') as f:
        pickle.dump(X,f)
    with open('pickledDatasetY'+prefix,'wb') as f:
        pickle.dump(Y,f)
        
def loadDataset(prefix=""):
    try:
        X = unpickleBigDataset('x')
        Y = unpickleBigDataset('y')
        if len(X) == len(Y) and len(X) == HOWMANY:
            return X,Y
        else:
            print("Failed loading dataset from file system")
            return None,None
    except:
        print("Failed loading dataset from file system")
        return None,None
    
def display(X,Y,howmany=None):
    if howmany is None:
        howmany = X.shape[0]
        
    for i in range(howmany):
        print(X[i].max(),X[i].min())
        plt.figure()
        plt.subplot(1,2,1)
        plt.imshow(X[i])
        plt.subplot(1,2,2)
        plt.imshow(Y[i])
        


# In[5]:


# def get_patches(image,size,side,imposition):
#     patches = []
    
    
#     for i in range(int(size[0]/side)):
#         for j in range(int(size[1]/side)):
#             patches += [image[i*side:(i+1)*side,j*side:(j+1)*side]]
#     return patches

def get_patches(image,size,side,imposition):
    patches = []
    
    if len(image.shape)==3:
        img = np.zeros((image.shape[0]+imposition,image.shape[1]+imposition,3))
        for i in range(3):
            img[...,i] = np.pad(image[...,i],((imposition,0),(imposition,0)),'reflect')
        image = img
    else:
        image = np.pad(image,((imposition,0),(imposition,0)),'reflect')

    for i in range(int(size[0]/side)):
        for j in range(int(size[1]/side)):
            imp1=np.max([i*side-imposition,0])
            imp2=(i+1)*side+imposition if imp1!=0 else (i+1)*side+imposition*2
            imp3=np.max([j*side-imposition,0])
            imp4=(j+1)*side+imposition if imp3!=0 else (j+1)*side+imposition*2
            patches += [image[imp1:imp2,imp3:imp4]]
    return patches

    #v1
# def preprocessorX(image):
    # size,side,imposition = SIZE,SIDE,IMPOSITION
    # image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)[...,2]

    # image = image.astype(np.float32)

    # if image.max() > 1:
        # image /= 255

    # image -= image.mean()
    # image /= image.std()

    # image[image<-3] = -3
    # image[image>3] = 3

    # norm = np.max(np.abs([image.min(),image.max()]))
    # image /= norm if norm!=0 else 3

    # return get_patches(image,size,side,imposition)
    
    #v2
# def preprocessorX(image):
    # size,side,imposition = SIZE,SIDE,IMPOSITION
    # image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)[...,0]

    # image = image.astype(np.uint8)
    # image = cv2.fastNlMeansDenoising(image,None,9,13)
    # image = image**0.9
    
    # patches = get_patches(image,size,side,imposition)

    # for j in range(len(patches)):        
        # patches[j] = patches[j].astype(np.float32)
    
        # dzielnik = patches[j].std()
        # patches[j] -= patches[j].mean()
        # if dzielnik!=0:
            # patches[j] /= dzielnik

        # patches[j][patches[j]<-3] = -3
        # patches[j][patches[j]>3] = 3
        
        # for i in range(3):
            # dzielnik = np.max(np.abs([patches[j].min(),patches[j].max()]))
            # if dzielnik != 0:
                # patches[j] /= dzielnik       
                
    # return patches
    
    #v3
def preprocessorX(image):
    # size,side,imposition = SIZE,SIDE,IMPOSITION
    # image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # mask = np.zeros_like(image).astype(np.int)
    
    # a,b,c,d = 21,12,8,-1
    # mask[image[...,2]<(150+b)]+=1
    # mask[image[...,2]<(110+a)]+=1
    # mask[image[...,1]<(35+d)]+=1
    # mask[image[...,1]<(5+c)]+=1
    # mask[mask!=4] = 0
    # mask[mask==4] = 1
    
    # kernel = np.ones((5,5),np.uint8)
    # opening = cv2.morphologyEx(mask.astype(np.float32),cv2.MORPH_CLOSE,kernel)
    # mask = cv2.morphologyEx(opening.astype(np.float32),cv2.MORPH_OPEN,kernel)
    
    
    # return get_patches(mask,size,side,imposition)
    
    size,side,imposition = SIZE,SIDE,IMPOSITION
    image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)[...,0]

    image = image.astype(np.uint8)
    image = cv2.fastNlMeansDenoising(image,None,9,13)
    image = image**0.9
    
    image[image<50] = 0
    image[image>90] = 0
    # image[image!=0] = 1
    kernel = np.ones((3,3),np.uint8)
    opening = cv2.morphologyEx(image.astype(np.float32),cv2.MORPH_CLOSE,kernel)
    image = cv2.morphologyEx(opening.astype(np.float32),cv2.MORPH_OPEN,kernel)
    
    patches = get_patches(image,size,side,imposition)
    
    for j in range(len(patches)):        
        patches[j] = patches[j].astype(np.float32)
    
        dzielnik = patches[j].max() - patches[j].min()
        patches[j] -= patches[j].min()
        if dzielnik!=0:
            patches[j] /= dzielnik

        
        
        
        
                
    return patches

    
    
def preprocessorY(image):
    size,side,imposition = SIZE,SIDE,IMPOSITION

    image = image.astype(np.float32)
    # if image.max() > 1:
        # image /= 255
    # for i in range(3):
        # image[...,i] = (image[...,i] - image[...,i].min())/(image[...,i].max() - image[...,i].min())
    image[image!=0] = 1# = (image - image.min())/(image.max() - image.min())
    image[image==0] = 0
    return get_patches(image,size,side,imposition)
    
def getRoadStats(arr,mask):
    b = mask.astype(np.bool)
    x = arr[b]
    if len(x) != 0:
        return [x.max(0),x.min(0),x.mean(0),x.std(0),np.median(x,axis=0)]
    else:
        return None

def preprocessXY(X,Y):
    
    r = []
    for i in range(len(X)):
        s = getRoadStats(X[i],Y[i])
        if s is not None:
            r += [s]
            
    return np.array(r).mean(0)
    

def preprocess(images,preprocessor,prefix):
    onlyfiles = [f for f in os.listdir(TMP_DIR) if os.path.isfile(os.path.join(TMP_DIR, f)) and f.startswith(prefix)]
    if len(onlyfiles)==0:
        I = None
        result = []
    else:
        result = unpickleBigDataset(prefix)[:HOWMANY*HOWMANYPERIMAGE]
        I = int(len(result)/HOWMANYPERIMAGE)
    print("Cached images {}.".format(len(result)))
    
    s = int(IMAGESPERFILE )
    if len(result)<HOWMANY*HOWMANYPERIMAGE and images is not None:
        print("Preprocessing images.")
        print("Proceeding from {} image.".format(I if I is not None else 0))
        # i = I if I is not None else 0
        images = images[I:]
        i = 0
        gc.collect()
        # for image in tqdm(images):
        R = len(images)
        for j in tqdm(range(R)):
            if len(images)==0:
                break
            result += preprocessor(images[i])
            if j%s==0:
                images = images[i+1:]
                i = 0
                pickleBigDataset(prefix,result,int(s*HOWMANYPERIMAGE / 2))
            else:
                i += 1
            gc.collect()
        pickleBigDataset(prefix,result,int(s*HOWMANYPERIMAGE / 2))
        images = None
        gc.collect()
    images = None
    gc.collect()
    
    return np.array(result)


# In[6]:


def doSomeDeepLearning(X=None,Y=None,side=85):
    num_classes = 2    # ile klas będziemy rozpoznawali

    # input image dimensions
    img_rows, img_cols = side,side   # takie wymiary mają obrazki w bazie MNIST

    # the data, shuffled and split between train and test sets
    try:
        x_train = np.array(unpickleBigDataset('xain'))
        y_train = np.array(unpickleBigDataset('yain'))
        x_test = np.array(unpickleBigDataset('xest'))
        y_test = np.array(unpickleBigDataset('yest'))
        if len(x_train)==0 or len(y_train)==0 or len(x_test)==0 or len(y_test)==0:
            
            raise Exception
        # if len(x_train) + len(x_test)!=HOWMANY*HOWMANYPERIMAGE:
            # print("Ala ma psa")
            # raise MyException
    except:
        if X is None or Y is None:
            raise MyException
            
        # mask = [not np.all(xyz==xyz.mean()) for xyz in Y]
        # X = X[mask]
        # Y = Y[mask]
        # print(len(X))
        mask = [not np.all(xyz==xyz.mean()) for xyz in X]
        X = X[mask]
        Y = Y[mask]
        print(len(X))
                    
        test_size = 3 * len(X)//10
        
        x_train, x_test, y_train, y_test = X[:-test_size],X[-test_size:],Y[:-test_size],Y[-test_size:]#train_test_split(X, Y, test_size=0.3)
        if K.image_data_format() == 'channels_first':
            x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
            x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
            y_train = y_train.reshape(y_train.shape[0],1, img_rows,img_cols)
            y_test = y_test.reshape(y_test.shape[0],1, img_rows,img_cols)
            input_shape = (1, img_rows, img_cols)
        else:
            x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
            x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
            y_train = y_train.reshape(y_train.shape[0], img_rows,img_cols,1)
            y_test = y_test.reshape(y_test.shape[0], img_rows,img_cols,1)
            input_shape = (img_rows, img_cols, 1)
        # y_train= to_categorical(y_train, num_classes=2)
        # y_test= to_categorical(y_test, num_classes=2)
        s = IMAGESPERFILE * HOWMANYPERIMAGE
        pickleBigDataset('xain',x_train,s)
        pickleBigDataset('yain',y_train,s)
        pickleBigDataset('xest',x_test,s)
        pickleBigDataset('yest',y_test,s)

    print('x_train shape:', x_train.shape)
    print('y_train shape:', y_train.shape)
    print(x_train.shape[0], 'train samples')
    print(x_test.shape[0], 'test samples')
    
    curr_epoch = -1
    onlyfiles = [f for f in os.listdir('.') if os.path.isfile(os.path.join('.', f)) and f.startswith('moj_ulubiony_model') and f.endswith('.h5')]
    o1 = (y_train.sum()+y_test.sum())/(y_train.size+y_test.size)
    print(o1)
    custom_objects={'weighted_binary_crossentropy':create_weighted_binary_crossentropy(o1,1 - o1),'max_pred':max_pred,'min_pred':min_pred,'mean_pred':mean_pred,'relu_advanced':relu_advanced,'weighted_mean_squared_error':create_weighted_mean_squared_error(o1,1-o1)}
    if len(onlyfiles) == 0:
        print("No saved model. Preparing model.")
        imput = Input(shape=(side,side,1))
        activationConv=relu_advanced
        initializationConv='he_normal'
        kernelSizeConv=(5,5)
        conv1 = Conv2D(32, 
                       kernel_size=kernelSizeConv,
                       padding="same", 
                       #activation=activationConv,
                       kernel_initializer=initializationConv,
                       kernel_regularizer=l1_l2(0.01),
                       bias_regularizer=l1_l2(0.01),
                       activity_regularizer=l1_l2(0.01)
                      )(imput)
        # conv1 = Conv2D(32,
                       # kernel_size=kernelSizeConv,
                       # padding="same", 
                       ##activation=activationConv,
                       # kernel_initializer=initializationConv,
                       # kernel_regularizer=l1_l2(0.01),
                       # bias_regularizer=l1_l2(0.01),
                       # activity_regularizer=l1_l2(0.01)
                      # )(conv1)
        conv1 = Conv2D(32,
                       kernel_size=kernelSizeConv,
                       padding="same", 
                       #activation=activationConv,
                       kernel_initializer=initializationConv,
                       kernel_regularizer=l1_l2(0.01),
                       bias_regularizer=l1_l2(0.01),
                       activity_regularizer=l1_l2(0.01)
                      )(conv1)
        conv1 = Conv2D(32, 
                       kernel_size=kernelSizeConv,
                       padding="same", 
                       activation=activationConv,
                       kernel_initializer=initializationConv,
                       kernel_regularizer=l1_l2(0.01),
                       bias_regularizer=l1_l2(0.01),
                       activity_regularizer=l1_l2(0.01)
                      )(conv1)
        # dropout1 = Dropout(0.2)(conv1)
        maxpool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
        bn1 = BatchNormalization()(maxpool1)
        
        # conv2 = Conv2D(64, 
                       # kernel_size=kernelSizeConv,
                       # padding="same", 
                       ##activation=activationConv,
                       # kernel_initializer=initializationConv,
                       # kernel_regularizer=l1_l2(0.01),
                       # bias_regularizer=l1_l2(0.01),
                       # activity_regularizer=l1_l2(0.01)
                      # )(bn1)#maxpool1)
        # conv2 = Conv2D(64, 
                       # kernel_size=kernelSizeConv,
                       # padding="same", 
                       ##activation=activationConv,
                       # kernel_initializer=initializationConv,
                       # kernel_regularizer=l1_l2(0.01),
                       # bias_regularizer=l1_l2(0.01),
                       # activity_regularizer=l1_l2(0.01)
                      # )(conv2)
        # conv2 = Conv2D(64, 
                       # kernel_size=kernelSizeConv,
                       # padding="same", 
                       ##activation=activationConv,
                       # kernel_initializer=initializationConv,
                       # kernel_regularizer=l1_l2(0.01),
                       # bias_regularizer=l1_l2(0.01),
                       # activity_regularizer=l1_l2(0.01)
                      # )(conv2)
        conv2 = Conv2D(64, 
                       kernel_size=kernelSizeConv,
                       padding="same", 
                       activation=activationConv,
                       kernel_initializer=initializationConv,
                       kernel_regularizer=l1_l2(0.01),
                       bias_regularizer=l1_l2(0.01),
                       activity_regularizer=l1_l2(0.01)
                      )(bn1)#(conv2)
        # dropout2 = Dropout(0.25)(conv2)
        maxpool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
        bn2 = BatchNormalization()(maxpool2)
        
        # conv3 = Conv2D(128, 
                       # kernel_size=kernelSizeConv,
                       # padding="same", 
                       ##activation=activationConv,
                       # kernel_initializer=initializationConv,
                       # kernel_regularizer=l1_l2(0.01),
                       # bias_regularizer=l1_l2(0.01),
                       # activity_regularizer=l1_l2(0.01)
                      # )(bn2)#maxpool2)
        # conv3 = Conv2D(128, 
                       # kernel_size=kernelSizeConv,
                       # padding="same", 
                      ## activation=activationConv,
                       # kernel_initializer=initializationConv,
                       # kernel_regularizer=l1_l2(0.01),
                       # bias_regularizer=l1_l2(0.01),
                       # activity_regularizer=l1_l2(0.01)
                      # )(conv3)
        # conv3 = Conv2D(128, 
                       # kernel_size=kernelSizeConv,
                       # padding="same", 
                       # activation=activationConv,
                       # kernel_initializer=initializationConv,
                       # kernel_regularizer=l1_l2(0.01),
                       # bias_regularizer=l1_l2(0.01),
                       # activity_regularizer=l1_l2(0.01)
                      # )(conv3)
        conv3 = Conv2D(128, 
                       kernel_size=kernelSizeConv,
                       padding="same", 
                       activation=activationConv,
                       kernel_initializer=initializationConv,
                       kernel_regularizer=l1_l2(0.01),
                       bias_regularizer=l1_l2(0.01),
                       activity_regularizer=l1_l2(0.01)
                      )(bn2)#(conv3)              

        upsample1 = UpSampling2D(size=(2,2))(conv3)
        
        
        concat1 = concatenate([upsample1,conv2,])#lambda1])
        bn3 = BatchNormalization()(concat1)
        # conv4 = Conv2D(64, 
                       # kernel_size=kernelSizeConv,
                       # padding="same", 
                       ##activation=activationConv,
                       # kernel_initializer=initializationConv,
                       # kernel_regularizer=l1_l2(0.01),
                       # bias_regularizer=l1_l2(0.01),
                       # activity_regularizer=l1_l2(0.01)
                      # )(bn3)#concat1)
        # conv4 = Conv2D(64, 
                       # kernel_size=kernelSizeConv,
                       # padding="same", 
                       ##activation=activationConv,
                       # kernel_initializer=initializationConv,
                       # kernel_regularizer=l1_l2(0.01),
                       # bias_regularizer=l1_l2(0.01),
                       # activity_regularizer=l1_l2(0.01)
                      # )(conv4)
        # conv4 = Conv2D(64, 
                       # kernel_size=kernelSizeConv,
                       # padding="same", 
                       ##activation=activationConv,
                       # kernel_initializer=initializationConv,
                       # kernel_regularizer=l1_l2(0.01),
                       # bias_regularizer=l1_l2(0.01),
                       # activity_regularizer=l1_l2(0.01)
                      # )(conv4)
        conv4 = Conv2D(64, 
                       kernel_size=kernelSizeConv,
                       padding="same", 
                       activation=activationConv,
                       kernel_initializer=initializationConv,
                       kernel_regularizer=l1_l2(0.01),
                       bias_regularizer=l1_l2(0.01),
                       activity_regularizer=l1_l2(0.01)
                      )(bn3)#(conv4)
        # dropout4 = Dropout(0.25)(conv4)
        upsample2 = UpSampling2D(size=(2,2))(conv4)
        zpad1 = ZeroPadding2D(((1,0),(1,0)))(upsample2)
        concat2 = concatenate([zpad1,conv1])
        bn4 = BatchNormalization()(concat2)
        # conv5 = Conv2D(32, 
                       # kernel_size=kernelSizeConv,
                       # padding="same", 
                       ##activation=activationConv,
                       # kernel_initializer=initializationConv,
                       # kernel_regularizer=l1_l2(0.01),
                       # bias_regularizer=l1_l2(0.01),
                       # activity_regularizer=l1_l2(0.01)
                      # )(bn4)#concat2)
        # conv5 = Conv2D(32, 
                       # kernel_size=kernelSizeConv,
                       # padding="same", 
                       ##activation=activationConv,
                       # kernel_initializer=initializationConv,
                       # kernel_regularizer=l1_l2(0.01),
                       # bias_regularizer=l1_l2(0.01),
                       # activity_regularizer=l1_l2(0.01)
                      # )(conv5)
        # conv5 = Conv2D(32, 
                       # kernel_size=kernelSizeConv,
                       # padding="same", 
                       ##activation=activationConv,
                       # kernel_initializer=initializationConv,
                       # kernel_regularizer=l1_l2(0.01),
                       # bias_regularizer=l1_l2(0.01),
                       # activity_regularizer=l1_l2(0.01)
                      # )(conv5)
        conv5 = Conv2D(1, 
                       kernel_size=kernelSizeConv,
                       padding="same", 
                       activation=activationConv,
                       kernel_initializer=initializationConv,
                       kernel_regularizer=l1_l2(0.01),
                       bias_regularizer=l1_l2(0.01),
                       activity_regularizer=l1_l2(0.01)
                      )(bn4)#(conv5)
        model = Model(inputs=imput, outputs=conv5)

        model.compile(loss=create_weighted_binary_crossentropy(o1,1-o1),
        #keras.losses.binary_crossentropy,#keras.losses.mean_squared_error,#
                  optimizer=keras.optimizers.Adam(),       
                  metrics=[max_pred,mean_pred,min_pred,'binary_accuracy'])#'accuracy'])#,'precision','recall'])
    elif len(onlyfiles) == 1:
        print("Saved model:\"{}\"".format(onlyfiles[0]))
        model = keras.models.load_model(onlyfiles[0],custom_objects=custom_objects)
    else:
        # onlyfiles = map(lambda y:filter(lambda x:x is not None and x.startswith('epoch'),y.split('.')[0].split('_')),onlyfiles)
        curr_epoch = max(list(map(lambda x:int(list(filter(lambda x:x.startswith('epoch'),x.split('.')[0].split('_')))[0][5:]),onlyfiles)) )
        model = keras.models.load_model("moj_ulubiony_model_epoch{}.h5".format(curr_epoch),custom_objects=custom_objects)
        print("Saved model:\"moj_ulubiony_model_epoch{}.h5\"".format(curr_epoch))
    
    model.compile(loss=create_weighted_mean_squared_error(o1,1-o1),#create_weighted_binary_crossentropy(o1,1-o1),
        #keras.losses.binary_crossentropy,#keras.losses.mean_squared_error,#
                  optimizer=keras.optimizers.Adam(),       
                  metrics=[max_pred,mean_pred,min_pred,'binary_accuracy'])#'accuracy'])#,'precision','recall'])

    curr_epoch += 1
    print("Current epoch:{}".format(curr_epoch))
    model.summary()
    
    
 
    for i in range(curr_epoch,epochs):
        model.fit(x_train, y_train,
                      batch_size=batch_size,#np.min([batch_size,np.max([batch_size//2**(i-3),1])]),
                      epochs=1,
                      verbose=1,
                      # class_weight = {0:1, 1:20},
                      validation_data=(x_test, y_test))
        model.save("moj_ulubiony_model_epoch{}.h5".format(i))
        # tmp = model.predict(x_test)
        # print("Max:",tmp.max(),"Min:",tmp.min())
    
    score = model.evaluate(x_test, y_test, verbose=0)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])


# In[7]:

def normalize(img,size):
    if img.max() - img.min() != 0:
        return ((img - img.min())/(img.max() - img.min())).reshape(size)
    else:
        img[img!=0] = 1
        return img

def myimshow(x,y,i,size):
    print(x[i].shape)
    print(y[i].shape)
    plt.figure()
    plt.subplot(221)
    if len(x[i].shape)!=2:
        plt.imshow(normalize((x[i])[...,1],size))
    else:
        plt.imshow(normalize(x[i],size))
    plt.subplot(222)
    plt.imshow(normalize(y[i],size))
    plt.show()
    
def load_model(filename):
    o1 = 0.05
    custom_objects={'weighted_binary_crossentropy':create_weighted_binary_crossentropy(o1,1 - o1),'max_pred':max_pred,'min_pred':min_pred,'mean_pred':mean_pred,'relu_advanced':relu_advanced,'weighted_mean_squared_error':create_weighted_mean_squared_error(o1,1-o1)}
    return keras.models.load_model(filename,custom_objects=custom_objects)
    
def max_pred(y_true, y_pred):
    return K.max(y_pred)
    
def min_pred(y_true, y_pred):
    return K.min(y_pred)
    
def mean_pred(y_true, y_pred):
    return K.mean(y_pred)
    
def relu_advanced(x):
    return K.relu(x, alpha=0.001, max_value=1)
    
def create_weighted_binary_crossentropy(zero_weight, one_weight):

    def weighted_binary_crossentropy(y_true, y_pred):

        # Original binary crossentropy (see losses.py):
        # K.mean(K.binary_crossentropy(y_true, y_pred), axis=-1)
        return K.mean((y_true * one_weight + (1. - y_true) * zero_weight)*K.binary_crossentropy(y_true, y_pred))

    return weighted_binary_crossentropy
    
def create_weighted_mean_squared_error(zero_weight, one_weight):

    def weighted_mean_squared_error(y_true, y_pred):

        # Original (see losses.py):
        # K.mean(K.square(y_pred - y_true), axis=-1)
        return K.mean((y_true * one_weight + (1. - y_true) * zero_weight)*K.square(y_pred - y_true))

    return weighted_mean_squared_error
    
if __name__=="__main__":
    try:
        doSomeDeepLearning()
    except MyException as e:
        
        urlX = "https://www.cs.toronto.edu/~vmnih/data/mass_roads/train/sat/index.html"
        urlY = "https://www.cs.toronto.edu/~vmnih/data/mass_roads/train/map/index.html"
        
        if LOAD:
            print("Loading images (X)")
            X = loadImagesFromSite(urlX,'f')
        else:
            X = None
            
        if LOAD:
            print("Loading images (Y)")
            Y = loadImagesFromSite(urlY,'z')
        else:
            Y = None
            
        # for i in range(HOWMANY):
            # myimshow(X,Y,i,(750,750))
            
        if PREPROCESS:
            print("Preprocessing images (X)")
            X = preprocess(X,preprocessorX,'x')
            
        
        
        if PREPROCESS:
            print("Preprocessing images (Y)")
            Y = preprocess(Y,preprocessorY,'y')
        
        

            # r = preprocessXY(X,Y)
            # print("\n\t| {}".format('Value'))
            # l = ['max','min','avg','std','median']
            # for i,c1 in enumerate(r):  
                # print("{}\t| {}".format(l[i],c1))
        # for i in range(HOWMANY*HOWMANYPERIMAGE):
            # myimshow(X,Y,i,(85,85))

        doSomeDeepLearning(X,Y)


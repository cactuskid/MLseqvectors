
# coding: utf-8

# In[1]:


import csv
import keras
from Bio import SeqIO
import numpy as np
import keras
import scipy.signal as sig
import scipy.linalg as linalg
from scipy.fftpack import rfft, fftshift
import glob
from matplotlib import pyplot as plt

import multiprocessing as mp
import functools


# In[2]:


def seq2vec(argvec, hyperparams):
    seqvec = argvec[0]
    propmat = np.zeros((len(propdict),len(seqvec)))
    for i,prop in enumerate(propdict):
        propmat[i,:] = np.vectorize( hyperparams['propdict'][prop].get)(seqvec)
    return [propmat]

def gaussianSmooth(argvec, hyperparams):
    seqvec = argvec[0]
    for i in range(seqvec.shape[0]):
        seqvec[i,:] = sig.fftconvolve(seqvec[i,:], hyperparams['Gaussian'], mode='same')
    return [seqvec]

def fftall(argvec, hyperparams):
    seqvec = argvec[0]
    fftmat = np.zeros( seqvec.shape )
    for row in range( seqvec.shape[0]):
        fftmat[row,:] = rfft( seqvec[row,:] )
    return [fftmat ]


def clipfft(argvec, hyperparams):
    #ony up to a certain frequency pad w zeros if fftmat is too small
    fftmat = argvec[0]
    if fftmat.shape[1]-1 < hyperparams['clipfreq']:
        padded = np.hstack( [fftmat , np.zeros( ( fftmat.shape[0] , hyperparams['clipfreq'] - fftmat.shape[1] ))] )
        return [np.asmatrix(padded.ravel())]
    else:
        return [ np.asmatrix(fftmat[:,:hyperparams['clipfreq']].ravel()) ]


def retfinal_first(argvec, hyperparams):
    #convenience function for unpacking
    return argvec[0]
    
    
def showmat(seqvec):
    plt.imshow(seqvec ,  aspect='auto')
    plt.colorbar()
    plt.show()
    
def loadDict(csvfile):    
    with open(csvfile , 'r') as filestr:
        final = {}
        propdict= csv.DictReader(filestr)
        for row in propdict:
            
            for key in row.keys():
                if key != 'letter Code' and key!= 'Amino Acid Name':
                    if key not in final:
                        final[key]={}
                    final[key][row['letter Code']] = float(row[key])
    return final


def compose(functions):
    def compose2(f, g):
        return lambda x: f(g(x))
    retfunction = functools.reduce(compose2, functions, lambda x: x)

    return retfunction

def seq2numpy(argvec, hyperparams):
    seq = argvec
    return [list(seq)]

def worflow( input1, functions , kwargs):
    for function in functions:
        input1 = function( (input1,kwargs) )  
    return input1

def dataGen( fastas , fulldata = False):
    for fasta in fastas:
        fastaIter = SeqIO.parse(fasta, "fasta")
        for seq in fastaIter:
            if len(seq.seq)>0:
                if fulldata == False:
                    yield seq.seq
                else:
                    yield seq
            



# In[3]:


propdict = loadDict('./physicalpropTable.csv')

print('physical properties of amino acids')
print(propdict)

#gaussian smooth to apply physical properties of neighbors to each residue. tuneable kmer?
nGaussian = 5
stdv = .5
window = sig.gaussian(nGaussian, std=stdv)
window /= np.sum(window)
print('gaussian filter for sequence physical props')
plt.plot(window)
plt.show()


hyperparams={'propdict': propdict  , 'Gaussian':window , 'clipfreq':500 }

seqdir = './datasets/truepositive/'
fastas = glob.glob(seqdir +'*.fasta')


seqIter = dataGen(fastas)
pipeline_functions = [ seq2numpy,  seq2vec , gaussianSmooth, fftall , clipfft , retfinal_first ]
configured = []

for func in pipeline_functions:
    configured.append(functools.partial( func , hyperparams=hyperparams ) )

seq = next(seqIter)
for func in configured:
    seq = func(seq)
    print(seq)

pipeline = compose(reversed(configured))
for i in range(20):
    seq = next(seqIter)
    print(seq)
    res = pipeline(seq)
    print(res)
    print(res.shape)
    


# In[4]:


#make the data
#works
import pickle
import random



folders = ['./datasets/truepositive/' , './datasets/truenegative/' ]

#use random uniclust entries as a negative dataset
def iter_sample_fast(iterator, samplesize):
    results = []
    
    # Fill in the first samplesize elements:
    for _ in range(samplesize):
        results.append(next(iterator))
    random.shuffle(results)  # Randomize their positions
    for i, v in enumerate(iterator, samplesize):
        r = random.randint(0, i)
        if r < samplesize:
            results[r] = v  # at a decreasing rate, replace random items

    if len(results) < samplesize:
        raise ValueError("Sample larger than population.")
    return results


uniclust = '/home/cactuskid/DB/uniclust30_2017_10_seed.fasta'
loadRandom = False
negativesamples = 5000

if loadRandom == True:
    print('loading random entries to negative dataset')
    seqIter = dataGen([uniclust] , True )
    randomentries = iter_sample_fast( seqIter , negativesamples)
    SeqIO.write(randomentries, folders[1] + 'rando_uniclust.fasta', "fasta")
    print('done')

datasets = []
for folder in folders:
    #fourier transform all fastas
    print('fourier transform of '+folder)
    fastas = glob.glob(folder +'*.fasta')
    seqIter = dataGen(fastas)
    x = np.array( np.vstack(list(map(pipeline, seqIter ))) )
    #remove nans
    x = x[~np.isnan(x).any(axis=1)]
    y = [ folder ]*(x.shape[0])
    datasets.append( ( x,y) )
    print('DONE')
    
x,y = zip(*datasets)
Xtotal = np.vstack(x)
Ytotal= np.concatenate(y)


print('xmat')
print(Xtotal.shape)
print('ymat')
print(Ytotal.shape)

with open( './xdata.pkl' , 'wb') as handle:
    pickle.dump( Xtotal, handle, -1)

with open( './ydata.pkl' , 'wb') as handle:
    pickle.dump( Ytotal, handle,-1)
    


# In[31]:


#learn and validate 
#works

from keras.models import Sequential
from keras.layers import Dense, Conv1D
from keras.wrappers.scikit_learn import KerasClassifier
from keras.utils import np_utils
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold , train_test_split
from sklearn.preprocessing import LabelEncoder , robust_scale , normalize
from sklearn.pipeline import Pipeline


# define model. this could probably be a lot fancier... just used out of the box stuff
#isnt overfitting so we can def up the number of layers/nodes
def baseline_model(nlayers ,firstsize, hiddensize,  inputdim, outputdim , modeltype= 'normal' ):
    
    #try convolutional layer?
    #try dropout layer
    
    model = Sequential()
    
    if modeltype == 'convolutional':
        #finish me
        model.add(Conv1D(filters=30 , kernel_size=50 , strides=1, input_shape= ( 1, inputdim) ,  activation='relu'))
    else:
        model.add(Dense(firstsize , input_dim= inputdim  ))

    for i in range(nlayers):
        model.add(Dense(hiddensize ))
    
    model.add(Dense(outputdim , activation = 'softmax'))
    # Compile model
    model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
    
    return model



with open( './xdata.pkl' , 'rb') as handle:
    X = pickle.load(handle)

with open( './ydata.pkl' , 'rb') as handle:
    Y = pickle.load(handle)
    
#X = robust_scale(X)
X = normalize(X)

np.random.seed(0)

# encode class values as integers
encoder = LabelEncoder()
encoder.fit(Y)
encoded_Y = encoder.transform(Y)
dummy_y = np_utils.to_categorical(encoded_Y)


nlayers =3
firstsize = 3000
hiddensize = 300
inputdim=X.shape[1] 
outputdim = dummy_y.shape[1]

#output a configured model function with no inputs
retmodel = functools.partial( baseline_model , nlayers , firstsize, hiddensize, inputdim, outputdim , modeltype= 'normal' )    


#set up the problem
estimator = KerasClassifier(build_fn=retmodel, epochs=20, batch_size=15, verbose=1)
#X_train, X_test, y_train, y_test = train_test_split( X, dummy_y, test_size=0.2 , random_state=0)


# In[ ]:


#kfold validation of the estimator
kfold = KFold(n_splits=10, shuffle=True, random_state=0)
results = cross_val_score(estimator, X, dummy_y , cv=kfold)
print("Baseline: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))





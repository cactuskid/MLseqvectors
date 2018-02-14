

import csv
import numpy as np
import scipy.signal as sig

import scipy.linalg as linalg
from scipy.fftpack import rfft, fftshift
import glob

import functools


import pickle
import random


import re, string 

from keras.models import Sequential
from keras.layers import Dense, Conv1D, Dropout
from keras.wrappers.scikit_learn import KerasClassifier
from keras.utils import np_utils
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold , train_test_split
from sklearn.preprocessing import LabelEncoder , robust_scale , normalize
from sklearn.pipeline import Pipeline
from subprocess import Popen
import os
import multiprocessing as mp
from Bio import SeqIO

import pandas as pd
import shlex, subprocess
import config
import dask.dataframe as dd
import dask.array as da
from dask.delayed import delayed


def hd5save(df, f ):
    for col in df.columns:
        print(col)
        array = np.vstack( df[col].values )
        print(array.shape)

            


def applypipeline_to_series(series, pipeline, hyperparams):
    newseries = series.map( pipeline )
    if hyperparams['printResult']== True:
        print(newseries)
    return newseries

def openprocess(args , inputstr =None , verbose = False ):
    args = shlex.split(args)
    p = subprocess.Popen(args,  stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr= subprocess.PIPE)
    if verbose == True:
        print(inputstr)
    if inputstr != None:
        p.stdin.write(inputstr.encode())
    output = p.communicate()
    if verbose == True:
        print(output)
    p.wait()
    return output[0].decode()

def parsephobius( phobiusstr,hyperparams ):
    maxlen = 0 
    lines = phobiusstr.split('\n')
    for i,line in enumerate(lines):
        vals = line.split()
        if i > 0:
            try:
                end = int(vals[3])
                if maxlen < end:
                    maxlen = end
            except:
                pass
    domains =  {'SIGNAL':0, 'CYTOPLASMIC':1, 'NON CYTOPLASMIC':2, 'NON CYTOPLASMIC':3, 'TRANSMEMBRANE':4}
    propmat = np.zeros((len(domains),maxlen))

    for i,line in enumerate(lines):
        vals = line.split()
        if i > 0:
            key = None
            if 'SIGNAL' in line:
                key = 'SIGNAL'
            elif 'TRANSMEMBRANE' in line:
                key = 'TRANSMEMBRANE'
            elif 'NON' in line:
                key = 'NON CYTOPLASMIC'
            elif 'CYTOPLASMIC' in line:
                key = 'CYTOPLASMIC'
            if key != None:
                start = int(vals[2])
                end = int(vals[3])
                propmat[ domains[key] , start:end ] = 1
    
    if hyperparams['verbose'] == True:
        print(propmat)


    return [propmat]


def runphobius(seqstr , hyperparams):

    return openprocess(config.phobius, seqstr, hyperparams['verbose'])


def runpsipred(seqstr):
    #run psipred and collect output
    pass

def hourglass():
    #define NN architectcure with classic bottleneck shape
    #pepper in some dropouts
    pass

def seq2vec(argvec, hyperparams):
    
    if hyperparams['verbose'] == True:
        print(argvec[0].shape)


    seqvec = argvec[0]

    propmat = np.zeros((len(hyperparams['propdict']),len(seqvec)))
    
    for i,prop in enumerate(hyperparams['propdict']):
        vals = np.vectorize( hyperparams['propdict'][prop].get)(seqvec)
        propmat[i,:] = vals.ravel()

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
        if hyperparams['verbose'] == True:
            print ('DONE')
        return padded.ravel()
    else:
        if hyperparams['verbose'] == True:
            
            print ('DONE')

        return  fftmat[:,:hyperparams['clipfreq']].ravel()


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

def retx(x):
    return x

def compose(functions):
    def compose2(f, g):
        def fOg(x):
            return f(g(x))
        return fOg
    retfunction = functools.reduce(compose2, functions, retx )

    return retfunction

def seq2numpy(argvec, hyperparams):
    seq = argvec
    return [np.asarray( [char for char in seq] )]

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

        
def fastasToDF(fastas ,DDF = None):

    regex = re.compile('[^a-zA-Z1-9]')
    regexfast = re.compile('[^ARDNCEQGHILKMFPSTWYV]')

    DFdict={}
    for fasta in fastas:
        fastaIter = SeqIO.parse(fasta, "fasta")
        for seq in fastaIter:
            if len(seq.seq)>0:
                seqstr = regexfast.sub('', str(seq.seq))
                desc = regex.sub(' ', str(seq.description))
                DFdict[seq.description] = {'seq':seqstr , 'fasta':'>'+desc+'\n'+seqstr+'\n'}  
    
    df = pd.DataFrame.from_dict(DFdict, orient = 'index')
    if DDF == None:
        DDF = dd.from_pandas(df , npartitions = mp.cpu_count() )
    else:
        DDF.append(dd.from_pandas(df , npartitions = mp.cpu_count() ))
    return DDF

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



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
import dask
from dask.delayed import delayed
import h5py


def hd5save(df, name , overwrite ,verbose = False):
    #dataframe columns should all be arrays or bytestrings w ascii encoding

    if overwrite == True:
        f = h5py.File(name,'w')
    else:
        f = h5py.File(name,'a')
    f.create_group('datasets')

    for col in df.columns:
        try:
            array = np.vstack( df[col].values )
        except:
            maxlen = 0
            for array in df[col].values:
                if len(array) > maxlen :
                    maxlen=len(array)
            #pad bytestrings with spaces to maxlen

            array = np.vstack( [ np.string_(np.pad(array,((0, maxlen - len(array))) , mode='constant' , constant_values=20 )) for array in df[col].values ]  )
        
        if col not in f['datasets']:
            try:
                dset = f['datasets'].create_dataset(col, data=array ,chunks=True)
            except :
                dset = f['datasets'].create_dataset(col, data=array,  dtype="S" , chunks=True)
        else:
            dset = f['datasets'][col]
            x,y = dset.shape
            inx,iny = array.shape
            #resize dataset for new data.
            dset.resize(inx+x, y + max(0,iny-y) )
            dset[x:inx + x , : ] = array
    f.close()

def DaskArray_hd5loadDataset(files , verbose = False ):
    #load to dask arrays, all arrays passed should have the same dataset names
    datasets = {}
    for name in files:
        f = h5py.File(name,'r')
        print(list(f['datasets'].keys()) ) 
        for dataset in f['datasets'].keys():
            chunksize = [  max(1, int(chunk/2) ) for chunk in f['datasets'][dataset].chunks ]
            
            if verbose == True:
                print('chunking smaller than hdf5')
                print( chunksize)
                print( f['datasets'][dataset].chunks)
                #print(f['datasets'][dataset][0:10])
            
            if dataset not in datasets:
                datasets[dataset] = da.from_array(f['datasets'][dataset], chunks=chunksize )
                if verbose==True:
                    f['datasets'][dataset][0:2]
                    print(datasets[dataset][0:2].compute(get=dask.get) )
            else:
                array = da.from_array(f['datasets'][dataset], chunks=chunksize )
                datasets[dataset] = da.concatenate([array, datasets[dataset]], axis=0)
                if verbose ==True:
                    print('append')
                    print(datasets[dataset])
                    print(datasets[dataset][0:10].compute(get=dask.get) )
    return datasets

def applypipeline_to_series(series, pipeline, hyperparams):
    newseries = series.map( pipeline )
    if hyperparams['printResult']== True:
        print(newseries)
    return newseries

def openprocess(args , inputstr =None , verbose = False ):
    args = shlex.split(args)
    p = subprocess.Popen(args,  stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr= subprocess.PIPE)
    if verbose == True:
        print(inputstr.decode())
    if inputstr != None:
        p.stdin.write(inputstr)
        
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
    #return generic output to get the dask DF set up. 
    if seqstr == 'foo':
        return '''ID   MTH_DROMEa signal peptide
                FT   SIGNAL        1     24       
                FT   REGION        1      3       N-REGION.
                FT   REGION        4     19       H-REGION.
                FT   REGION       20     24       C-REGION.
                FT   TOPO_DOM     25    218       NON CYTOPLASMIC.
                FT   TRANSMEM    219    238       
                FT   TOPO_DOM    239    249       CYTOPLASMIC.
                FT   TRANSMEM    250    269       
                FT   TOPO_DOM    270    280       NON CYTOPLASMIC.
                FT   TRANSMEM    281    302       
                FT   TOPO_DOM    303    321       CYTOPLASMIC.
                FT   TRANSMEM    322    342       
                FT   TOPO_DOM    343    371       NON CYTOPLASMIC.
                FT   TRANSMEM    372    391       
                FT   TOPO_DOM    392    421       CYTOPLASMIC.
                FT   TRANSMEM    422    439       
                FT   TOPO_DOM    440    450       NON CYTOPLASMIC.
                FT   TRANSMEM    451    476       
                FT   TOPO_DOM    477    514       CYTOPLASMIC.
                //''' 
    else:
        return openprocess(config.phobius, seqstr, hyperparams['verbose'])

"""
def rungarnier(seqstr):
    #run psipred and collect output
    pass


def bezier(stop, start):
    for i in range(100):
        n= i/100
        (1-n)**2 p0 + 2*(1-n)*n p1 + n**2 p2  """

def hourglass(nlayers, mlayers, chokept, startneurons, endneurons):
    
    layersizes = np.zeros( (mlayers+nlayers,1 ))

    #define NN architectcure with classic bottleneck shape
    #pepper in some dropouts
    pass

def seq2vec(argvec, hyperparams):
    
    if hyperparams['verbose'] == True:
        print('argvec')
        print(argvec[0])
    seqvec =  argvec[0]
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
    if hyperparams['verbose']== True:
        print(fftmat.shape)

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
    try:
        seq = argvec.decode('ascii')
    except AttributeError:
        seq=argvec

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
            seqstr = regexfast.sub('', str(seq.seq))
            desc =regex.sub(' ', str(seq.description))
            fastastr = '>'+desc+'\n'+seqstr+'\n'
            DFdict[desc] = {'seq':seqstr.encode() , 'fasta': fastastr.encode() }
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

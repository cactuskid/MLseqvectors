

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

########################hdf5 save and load #################################################################
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
    
    for key in datasets:
        print(datasets[key][0:10].compute(get=dask.get))
    return datasets

def applypipeline_to_series(series, pipeline, hyperparams):
    newseries = series.map( pipeline )
    if hyperparams['printResult']== True:
        print(newseries)
    return newseries
 
##########################################Garnier pipeline################################################## 
def retFastaTmp(df):
    fastastr = ''.join( [ string for string in df['fasta'] ] )
    temp = tempfile.NamedTemporaryFile(dir=config.savedir)
    temp.write(fastastr)
    #make tempfiles
    return temp

def runGarnier(fasta , hyperparams):
    
    if hyperparams['verbose'] == True:
        print(fasta)

    if fasta == 'foo':
        default = str("""########################################
        # Program: garnier
        # Rundate: Mon 26 Feb 2018 16:24:35
        # Commandline: garnier
        #    -filter
        # Report_format: tagseq
        # Report_file: stdout
        ########################################

        #=======================================
        #
        # Sequence: SAMEA2619974_10776_4     from: 1   to: 604
        # HitCount: 163
        #
        # DCH = 0, DCS = 0
        # 
        #  Please cite:
        #  Garnier, Osguthorpe and Robson (1978) J. Mol. Biol. 120:97-120
        # 
        #
        #=======================================

                  .   10    .   20    .   30    .   40    .   50
              MYKRKIIIPILLFVILSLIVSAFSTLSLDRVDFSTRGDEFDQQWLLLISE
        helix HHHH                                    HHHHH     
        sheet     EEEEEEEEEEEEEEEEEEE    E EEEE            EEE  
        turns                             T       TT T         T
         coil                        CCCC      CCC  C         C 
                  .   60    .   70    .   80    .   90    .  100
              DGRADKAVVTKKAEEIKDDKIHAKNDLTIKTEIDKNSCVYTIQNINQPIS
        helix HHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHH                 
        sheet                                      EEEEEE     EE
        turns                                  TTTT      TT     
         coil                                              CCC  
                  .  110    .  120    .  130    .  140    .  150
              RLDYTKKTVWKFWDIPNEIKSCEQREGYYWAFNLGFDFNVYCFFTPAGKD
        helix                      HH                           
        sheet EE      EEE                   EEE      E   E      
        turns   TTTTTT   TT   TTTTT  TTTTTTT    TTTTT TTT   TTT 
         coil              CCC                 C          CC   C
                  .  160    .  170    .  180    .  190    .  200
              SFVGRISDKKYDFNTKITLTSGGESKSVDISNSNRVSNSVPDYFVAKWHG
        helix                                               H   
        sheet  EEEE        EEEEEE      EEEEE        E   EEEE    
        turns       TTTTTTT                    TTTTT   T     T T
         coil C    C             CCCCCC     CCC      CC       C 
                  .  210    .  220    .  230    .  240    .  250
              SLSTGAECPDITSLSPIYDRDSGEWKLGEKSSISNYQSFHSSGMKECLNS
        helix                        HHHHH                HHHHHH
        sheet       EEE  E EEE E                  E             
        turns    TTT   TT T   T TTT       T  TT TT  TTT TT      
         coil CCC                  CC      CC  C   C   C        
                  .  260    .  270    .  280    .  290    .  300
              YIIARTTGGLIVETPDSCKKRYNDLVEKAQSPEVFNIKGSTGSMTTQSSE
        helix                       HHHHHHHHHHHH                
        sheet EEEEEE   EEEEE                    EEE             
        turns               TTTTTTTT               TT           
         coil       CCC                              CCCCCCCCCCC
                  .  310    .  320    .  330    .  340    .  350
              LGELVVEDVQPFQFPVISLKINADWIGIVQPITKPTILSCESSKFKQGEI
        helix   HHHHHHH                              HH    HH  H
        sheet             EEEEE        EEEEE     EEEE           
        turns           TT           TT      T  T      TTTT     
         coil CC       C       CCCCCC       C CC             CC 
                  .  360    .  370    .  380    .  390    .  400
              GEIDVEVKNKGESGSVSISTVCLSPFKSVGTTPIIRLKKGESKSITVPIT
        helix HHHHHHHH                                          
        sheet                EEEEEEEE        EEEEEE       EEEEEE
        turns         T  T TT         TTTTTT       T TTTTT      
         coil          CC C          C      C       C           
                  .  410    .  420    .  430    .  440    .  450
              VSTKEDVSKTCSVTVQDESDPSVRVSKRCDVSASGVVLCEAGKKRCSGRF
        helix   HH                  H              HHHHH        
        sheet E          EEEEE       EEEEEE E   EEE            E
        turns     TTTTTTT     TT  TT       T TTT        TTTTTTT 
         coil  C                CC                              
                  .  460    .  470    .  480    .  490    .  500
              IEQCKSSGSEYGLIEECELDCKLDKYGQPFCPEVTPPPPPPPNGNGDKCE
        helix             HHHHHHHHHHH                           
        sheet EE                            EE                EE
        turns   TTT  T               TTTT TT  TT       TTTTTTT  
         coil      CC CCCC               C      CCCCCCC         
                  .  510    .  520    .  530    .  540    .  550
              PIWAIAKITIIPDFICEMEKVPYLKEGISGLVGFVMFIILIFMLKNLIGF
        helix HHHHHH       HHHHHHHHHHHHH          HHHHHHHHH     
        sheet       EEEEE                EE EEEEEE              
        turns             T             T                       
         coil            C                 C               CCCCC
                  .  560    .  570    .  580    .  590    .  600
              ENIPQRLMVLGFSLIIAILLGFLFYYLFWFGVALIVALVVMFFVIKIILG
        helix            HHH               HHHHHHHHHHHHHHHHHHHH 
        sheet      EEEEEE   EEEEEEEEEEEE                        
        turns     T                     TTT                    T
         coil CCCC                                              
              0
              KVGL
        helix     
        sheet    E
        turns T   
         coil  CC 

        #---------------------------------------
        #
        #  Residue totals: H:158   E:177   T:162   C:107
        #         percent: H: 26.9 E: 30.1 T: 27.6 C: 18.2
        #
        #---------------------------------------

        #---------------------------------------
        # Total_sequences: 1
        # Total_length: 604
        # Reported_sequences: 1
        # Reported_hitcount: 163
        #---------------------------------------
        """)
        if hyperparams['verbose'] ==True:
            print(default)

        return default

        
    else:
        return openprocess(config.garnier, fasta , hyperparams['verbose'])

    

def GarnierParser(garnierStr,hyperparams):
    #parse garnier output. return matrix with alpha, beta and loop topology.

    helixstr=''
    sheetstr=''
    trunstr=''
    coilstr=''

    
    for line in garnierStr.split('\n'):

        if 'helix' in line:
            helixstr += line[5:55]
        elif 'sheet' in line:
            sheetstr += line[5:55]
        elif 'turns' in line:
            trunstr+= line[5:55]
        elif 'coil' in line:
            coilstr+= line[5:55]

    features = [helixstr, sheetstr , trunstr , coilstr ]
    if hyperparams['verbose'] == True:
        for feat in features:
            print(feat)
    
    veclen = max( [len(stringdata) for stringdata in  features] )
    if  hyperparams['verbose'] == True:
        print(veclen)
    featmat = np.zeros( (len(features) , veclen))
    
    for i , seqstr in enumerate(features):
        index = [i for i, letter in enumerate(seqstr) if letter != ' ']
        featmat[i , index ] = 1 

    return [featmat]

##################################run processes, apply to dataframe partitions etc###########################################

def runOnDelayed(DF , pipeline):
    #split df into temp fastas
    dfs = DF.to_delayed()
    retdfs = [functions.delayed(pipeline) (df) for df in dfs]
    DDF =None
    for df in retdfs:
        if DDF == None:
            DDF = dd.from_pandas(df , npartitions = mp.cpu_count() )
        else:
            DDF.append(dd.from_pandas(df))
    
    return DDF


def openprocess(args , inputstr =None , verbose = False ):
    args = shlex.split(args)
    p = subprocess.Popen(args,  stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr= subprocess.PIPE )
    if verbose == True:
        print(inputstr.decode())
    if inputstr != None:
        p.stdin.write(inputstr)
        
    output = p.communicate()
    if verbose == True:
        print(output)
    p.wait()
    return output[0].decode()



######################################################################phobius tm topology predicition###############################################
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
    #return dummy output to get the dask DF set up. 
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



######################################neural network functions ################################################33



from keras.models import Sequential
from keras.layers import Dense, Conv1D, Dropout
from keras.wrappers.scikit_learn import KerasClassifier
from keras.utils import np_utils
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold , train_test_split
from sklearn.preprocessing import LabelEncoder , robust_scale , normalize
from sklearn.pipeline import Pipeline

def baseline_model(layers, inputdim, outputdim, dropout_interval ):
    #try convolutional layer?
    #try dropout layer
    model = Sequential()
    for i,size in enumerate(layers):
        if i ==0 :
            model.add(Dense(size , input_dim= inputdim  ))
        else:
            model.add(Dense(size ))
        if i % dropout_interval == 0:
        #added a dropout layer so it would generalize better
            model.add(Dropout(.5))    
    model.add(Dense(outputdim , activation = 'softmax'))
    # Compile model
    model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
    return model

def hourglass(nlayers, mlayers, chokept, startneurons, endneurons):
    layersizes = []
    step1 = (chokept - startneurons )/ nlayers    
    step2 = (endneurons - chokept)/nlayers
    for i in range(nlayers):
        layersizes.append( int(startneurons + step1*i))    
    for i in range(mlayers):
        layersizes.append( int(chokept + step2*i))    
    return layersizes


#####################################physical props pipeline###########################################################

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

def seq2numpy(argvec, hyperparams):
    try:
        seq = argvec.decode('ascii')
    except AttributeError:
        seq=argvec

    return [np.asarray( [char for char in seq] )]


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


########################## signal processing functions #####################################
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
    

#######################################pipeline building functions##############################################
def retx(x):
    return x

def compose(functions):
    def compose2(f, g):
        def fOg(x):
            return f(g(x))
        return fOg
    retfunction = functools.reduce(compose2, functions, retx )

    return retfunction

def dataGen( fastas , fulldata = False):
    for fasta in fastas:
        fastaIter = SeqIO.parse(fasta, "fasta")
        for seq in fastaIter:
            if len(seq.seq)>0:
                if fulldata == False:
                    yield seq.seq
                else:
                    yield seq

###########################################################dataframe / dataset building ##########################################
def fastasToDF(fastas , DDF = None, verbose=False):
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
    if verbose == True:
        print(df)

    try:
        DDF= dd.concat( [DDF,dd.from_pandas(df, npartitions = mp.cpu_count() )] , axis =0  )
    except:
        DDF = dd.from_pandas(df , npartitions = mp.cpu_count()  )
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



from keras.models import Sequential
from keras.layers import Dense, Conv1D, Dropout
from keras.wrappers.scikit_learn import KerasClassifier
from keras.utils import np_utils
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold , train_test_split
from sklearn.preprocessing import LabelEncoder , robust_scale , normalize
from sklearn.pipeline import Pipeline
from subprocess import popen

import shlex, subprocess


def prepCL(commandstr):
    args = shlex.split(commandstr)


def runphobius(seqstr):
    #run phobius on a sequence and collect output

def runpsipred(seqstr):
    #run psipred and collect output
    pass

def hourglass():
    #define NN architectcure with classic bottleneck shape

    #pepper in some dropouts
    pass

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

        
def fastasToDF(fastas)
    DFdict={}
    for fasta in fastas:
        fastaIter = SeqIO.parse(fasta, "fasta")
        for seq in fastaIter:
            if len(seq.seq)>0:
                DFdict.update({seq.description : {'seq':seq.seq} } )
    return pd.from_dict(DFdict)

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

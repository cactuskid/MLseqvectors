#test functions
import functions
import config
import os
import glob
from dask.multiprocessing import get
import dask

dask.set_options(get=dask.multiprocessing.get)


testOne = False
nGaussian = 7
stdv=.5
window = functions.sig.gaussian(nGaussian, std=stdv)
window /= functions.np.sum(window)

positive_datasets = config.workingdir + 'datasets/'

positives = [x[0] for x in os.walk(positive_datasets)]

propdict = functions.loadDict('./physicalpropTable.csv')
hyperparams={'propdict': propdict  , 'Gaussian':window , 'clipfreq':500, 'verbose' : True , 'printResult' : True  }


physical_pipeline_functions = [ functions.seq2numpy,  functions.seq2vec , functions.gaussianSmooth, functions.fftall , functions.clipfft ]
configured = []

for func in physical_pipeline_functions:
    configured.append(functions.functools.partial( func , hyperparams=hyperparams ) )
physicalProps_pipeline  = functions.compose(reversed(configured))


phobius_pipeline_functions = [  functions.runphobius,  functions.parsephobius, functions.gaussianSmooth, functions.fftall , functions.clipfft ]
configured = []
for func in phobius_pipeline_functions:
    configured.append(functions.functools.partial( func , hyperparams=hyperparams ) )
phobius_pipeline = functions.compose(reversed(configured))



applyphobiustoseries = functions.functools.partial( functions.applypipeline_to_series , pipeline=phobius_pipeline  , hyperparams=hyperparams ) 
applyphysicaltoseries = functions.functools.partial( functions.applypipeline_to_series , pipeline=physicalProps_pipeline  , hyperparams=hyperparams ) 

for folder in positives:
	fastas = glob.glob(folder+'/*fasta')

	if len(fastas)>0:
		if testOne == True:
			regex = functions.re.compile('[^a-zA-Z1-9]')
			#First parameter is the replacement, second parameter is your input string
			regex = functions.re.compile('[^a-zA-Z1-9]')
			regexfast = functions.re.compile('[^ARDNCEQGHILKMFPSTWYV]')
			for fasta in fastas:
				fastaIter = functions.SeqIO.parse(fasta, "fasta")
				for seq in fastaIter:
					seqstr = regexfast.sub('', str(seq.seq))
					desc = regex.sub(' ', seq.description)
					fastastr = '>'+desc+'\n'+seqstr
					print(fastastr)
					#result = physicalProps_pipeline(seqstr)
					#print (result)
					result = phobius_pipeline(fastastr)
					print (result)

		if testOne == False:
			print(fastas)
			df = functions.fastasToDF(fastas)
			df['folder'] = folder
			#df['physical'] = df['seq'].map_partitions( applyphysicaltoseries).compute(get=get)
			print(df)
			df['phobius'] = df['fasta'].map_partitions( applyphobiustoseries ).compute(get=get)
			print(df)


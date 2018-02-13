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
chunks = functions.mp.cpu_count()
window = functions.sig.gaussian(nGaussian, std=stdv)
window /= functions.np.sum(window)

save_data_tohdf5 = True
savedir = './'

positive_datasets = config.workingdir + 'datasets/'
positives = [x[0] for x in os.walk(positive_datasets)]





propdict = functions.loadDict('./physicalpropTable.csv')
hyperparams={'propdict': propdict  , 'Gaussian':window , 'clipfreq':500, 'verbose' : False , 'printResult' : False  }


#####configure pipelines #######################
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


##### final functions to be mapped #####

applyphobiustoseries = functions.functools.partial( functions.applypipeline_to_series , pipeline=phobius_pipeline  , hyperparams=hyperparams ) 
applyphysicaltoseries = functions.functools.partial( functions.applypipeline_to_series , pipeline=physicalProps_pipeline  , hyperparams=hyperparams ) 


pipelines={'physical':applyphysicaltoseries}#, 'phobius': applyphobiustoseries }
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
			
			for name in pipelines:
				meta = functions.dd.utils.make_meta( {name: object }, index=None)
				df[name] = df['seq'].map_partitions( pipelines[name] ).compute(get=get)
				print(df)

			if save_data_tohdf5 == True:
				dfs = df.to_delayed()
				#save datsets to hdf5 format in chunks
				filenames=[]
				handles = []
				overwrite = False

				for i in range(len(dfs)):
					name = savedir + 'hdf5_'+str(i)+'.hd5'
					filenames.append(name)

				"""if overwrite == True:
																	for i in range(chunks):
																		name = savedir + 'hdf5'+str(i)+'.hd5'
																		filenames.append(name)
																		handles.append(h5py.File(name,'w'))
																else:
																	filenames = glob.glob(savedir + '*.hd5')
																	for name in filenames:
																		handles.append(h5py.File(name,'w'))
												"""
				
				print(filenames)
				writes = [functions.delayed(functions.hd5save)(df , fn ) for df, fn in zip(dfs, filenames)]
				functions.dd.compute(*writes)

		
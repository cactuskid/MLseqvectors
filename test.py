#test functions
import functions
import config
import os
import glob
from dask.multiprocessing import get
import dask

dask.set_options(get=dask.multiprocessing.get)

if config.create_data == True:
	chunks = functions.mp.cpu_count()
	window = functions.sig.gaussian(config.nGaussian, std=config.stdv)
	window /= functions.np.sum(window)
	positives = [x[0] for x in os.walk(config.positive_datasets)]
	print(positives)

	propdict = functions.loadDict(config.proptable)
	hyperparams={'propdict': propdict  , 'Gaussian':window , 'clipfreq':500, 'verbose' : config.verbose , 'printResult' : True  }

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
	#use pipelines to generate feature vecotrs for fastas 
	#store matrices in hdf5 files
	pipelines={'physical':applyphysicaltoseries, 'phobius': applyphobiustoseries }
	inputData = {'physical':'seq', 'phobius': 'fasta' }
	df = None

	if config.generate_negative == True:
		print('generating negative sample fasta')
		fastaIter = functions.SeqIO.parse(config.uniclust, "fasta")
		sample = functions.iter_sample_fast(fastaIter, config.NegSamplesize)
		samplename = config.negative_dataset+str(config.NegSamplesize)+'rand.fasta'
		
		seqIO.write(sample , samplename , format = 'fasta')

	for folder in positives + [config.negative_dataset]:
		fastas = glob.glob(folder+'/*fasta')
		if len(fastas)>0:
			if config.testOne == True:
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
			else:
				df = functions.fastasToDF(fastas, df)
				df['folder'] = functions.np.string_(folder)

	for name in pipelines:
		meta = functions.dd.utils.make_meta( {name: object }, index=None)

		df[name] = df[inputData[name]].map_partitions( pipelines[name] ).compute(get=get)

	if config.save_data_tohdf5 == True:
		dfs = df.to_delayed()
		#save datsets to hdf5 format in chunks
		filenames=[]
		if config.overwrite == True:
			print('overwriting hdf5 datasets')
			for i in range(len(dfs)):
				name = config.savedir + config.dataname+str(i)+'.hd5'
				filenames.append(name)
		else:
			print('appending hdf5 datasets')
			filenames = glob.glob(config.savedir + '*.hd5')		
		inputlist = list(zip(dfs, filenames))
		writes = [functions.delayed(functions.hd5save)(df , fn , config.overwrite ) for df, fn in inputlist]
		functions.dd.compute(*writes , get=get)

if config.load_data == True:
	#create dask arrays for X and Y to feed to keras
	print('loading dataset')
	filenames = glob.glob(config.savedir + '*.hd5')
	print(filenames)
	dataset = functions.DaskArray_hd5loadDataset(filenames , verbose=config.verbose)
	print('dataset loaded with keys:')
	print(dataset.keys())

if config.make_networkmodel ==True:
	pass

if config.learn == True:
	pass


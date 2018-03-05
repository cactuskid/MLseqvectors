#test functions
import functions
import config
import os
import glob
from dask.multiprocessing import get
import dask

dask.set_options(get=dask.get)

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



	garnier_pipeline_functions = [  functions.runGarnier,  functions.GarnierParser, functions.gaussianSmooth, functions.fftall , functions.clipfft ]
	configured = []
	for func in garnier_pipeline_functions:
	    configured.append(functions.functools.partial( func , hyperparams=hyperparams ) )
	garnier_pipeline = functions.compose(reversed(configured))

	##### final functions to be mapped #####
	applyGarniertoseries = functions.functools.partial( functions.applypipeline_to_series , pipeline=garnier_pipeline  , hyperparams=hyperparams ) 
	applyphobiustoseries = functions.functools.partial( functions.applypipeline_to_series , pipeline=phobius_pipeline  , hyperparams=hyperparams ) 
	applyphysicaltoseries = functions.functools.partial( functions.applypipeline_to_series , pipeline=physicalProps_pipeline  , hyperparams=hyperparams ) 
	
	#use pipelines to generate feature vecotrs for fastas 
	#store matrices in hdf5 files
	
	pipelines={'physical':applyphysicaltoseries, 'phobius': applyphobiustoseries , 'garnier': applyGarniertoseries }

	inputData = {'physical':'seq', 'phobius': 'fasta' , 'garnier' : 'fasta' }

	df = None
	if config.generate_negative == True:
		print('generating negative sample fasta')
		fastaIter = functions.SeqIO.parse(config.uniclust, "fasta")
		sample = functions.iter_sample_fast(fastaIter, config.NegSamplesize)
		samplename = config.negative_dataset+str(config.NegSamplesize)+'rand.fasta'
		with open(samplename, "w") as output_handle:
			functions.SeqIO.write(sample , output_handle , format = 'fasta')

	for folder in positives + [config.negative_dataset]:
		fastas = glob.glob(folder+'/*fasta')
		print(folder)
		print(fastas)

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
						fastastr = '>'+seq.description+'\n'+seqstr
						print(fastastr)
						#result = physicalProps_pipeline(seqstr)
						#print (result)
						result = phobius_pipeline(fastastr)
						print (result)
			else:
				df = functions.fastasToDF(fastas, df , config.verbose)
				df['folder'] = functions.np.string_(folder)
	print(df)
	
	print('loaded fastas with categories:')
	print(df['folder'].unique().compute(get=get ) )


	for name in pipelines:
		print('calculating ' + name)

		meta = functions.dd.utils.make_meta( {name: object }, index=None)

		df[name] = df[inputData[name]].map_partitions( pipelines[name] ).compute(get=get)

	if config.verbose == True:
		print(df)

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
	
	X = functions.da.concatenate( [ dataset['physical'] , dataset['phobius'] , dataset['garnier'] ] , axis = 1  )
	Y = dataset['folder']

	print(X[0:10,:].compute(get=dask.get))
	print(Y[0:10].compute(get=dask.get))

if config.make_networkmodel ==True:
	nlayers = 10
	mlayers = 20
	choke = 900
	start = 3000
	end = 3000 
	dropout_interval = 15

	#X = robust_scale(X)
	X = functions.normalize(X)
	functions.np.random.seed(0)
	# encode class values as integers
	encoder = functions.LabelEncoder()
	
	print(Y)

	encoder.fit(Y)
	encoded_Y = encoder.transform(Y)
	
	print(encoded_Y)

	dummy_y = functions.np_utils.to_categorical(encoded_Y)

	print(dummy_y)

	inputdim=X.shape[1] 
	outputdim = dummy_y.shape[1]

	print(inputdim)
	print(outputdim)

	layers = functions.hourglass(nlayers, mlayers, choke, start, end)
	
	print('network shape')
	print(layers)

	model = functions.baseline_model(layers, inputdim, outputdim, dropout_interval )
	#output a configured model function with no inputs
	retmodel = functions.functools.partial( functions.baseline_model , layers, inputdim, outputdim, dropout_interval )   
	#set up the problem
	estimator = functions.KerasClassifier(build_fn=retmodel, epochs=15, batch_size=15, verbose=1)

if config.learn == True:
	from sklearn.model_selection import KFold
	kf= KFold(n_splits=5, shuffle=True, random_state=0)

	results = []
	for train, test in kf.split(X):
		X_train, X_test, y_train, y_test = X[train], X[test], dummy_y[train], dummy_y[test]
		encoded_Y_train, encoded_Y_test = encoded_Y[train], encoded_Y[test]
		estimator.fit( X_train, y_train)
		y_pred = estimator.predict( X_test)

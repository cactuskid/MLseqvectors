#test functions
import functions
import config
import os
import glob
from dask.multiprocessing import get
import dask
import gc




dask.set_options(get=dask.threaded.get)

if config.create_data == True:
	chunks = functions.mp.cpu_count()
	window = functions.sig.gaussian(config.nGaussian, std=config.stdv)
	window /= functions.np.sum(window)
	
	propdict = functions.loadDict(config.proptable)
	hyperparams={'propdict': propdict  , 'Gaussian':window , 'clipfreq':800, 'verbose' : config.verbose , 'printResult' : True  }

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
	
	pipelines={'physical':applyphysicaltoseries,  'garnier': applyGarniertoseries , 'phobius': applyphobiustoseries }
	inputData = {'physical':'seq', 'phobius': 'fasta' , 'garnier' : 'fasta' }
	df = None

	######## load fasta into dataframe ############################################
	if config.RunWithEcod==True:
		print('learning on protein structure clusters')
		fastas = [config.ecodFasta]
		df = functions.fastasToDF(fastas, None , config.verbose , config.RunWithEcod )
		if config.verbose == True:
			print(df)
		df['superfam'] = df['ECOD domain'].map( lambda x : x.split('.')[0] )
		gc.collect()

	else:
		positives = [x[0] for x in os.walk(config.positive_datasets)]
		print(positives)

		if config.generate_negative == True:
			print('generating negative sample fasta')
			fastaIter = functions.SeqIO.parse(config.uniclust, "fasta")
			sample = functions.iter_sample_fast(fastaIter, config.NegSamplesize)
			samplename = config.negative_dataset+str(config.NegSamplesize)+'rand.fasta'
			with open(samplename, "w") as output_handle:
				functions.SeqIO.write(sample , output_handle , format = 'fasta')

		dfs=[]
		for folder in positives + [config.negative_dataset]:
			fastas = glob.glob(folder+'/*fasta')
			print(folder)
			print(fastas)

			if len(fastas)>0:
				df = functions.fastasToDF(fastas, df , config.verbose)
				df['folder'] = functions.np.string_(folder)
				dfs.append(df)
		df= functions.dd.concat( dfs , axis =0 , interleave_partitions=True )


	print('loaded fastas with categories:')
	if config.RunWithEcod == False:
		print(len(df['folder'].unique().compute(get=get )) )
	else:
		print( len(df['superfam'].unique().compute(get=get ) ) )

	################################# run pipelines on sequences ###########################

	for name in config.pipelines:
		print('calculating ' + name)

		meta = functions.dd.utils.make_meta( {name: object }, index=None)
		
		df[name] = df[inputData[name]].map_partitions( pipelines[name] ).compute(get=get)
		gc.collect()








	if config.verbose == True:
		print(df)

	##############################save###########################################################3
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

##################################load ###############################################################
if config.load_data == True:
	#create dask arrays for X and Y to feed to keras
	print('loading dataset')
	filenames = glob.glob(config.savedir + config.dataname +'*.hd5')
	print(filenames)
	dataset = functions.DaskArray_hd5loadDataset(filenames , verbose=config.verbose)
	print('dataset loaded with keys:')
	print(dataset.keys())
	data = []
	for pipeline in config.pipelines:
		data.append( dataset[pipeline] )
	X = functions.da.concatenate(   data , axis = 1  )
	if config.RunWithEcod == True:
		Y= dataset['superfam']
	else:	
		Y = dataset['folder']

	print(X[0:10,:].compute(get=dask.get))
	print(Y[0:10].compute(get=dask.get))

######################################make NN ################################################################
if config.make_networkmodel ==True:
	#X = robust_scale(X)
	#X = functions.normalize(X)
	functions.np.random.seed(0)
	# encode class values as integers
	encoder = functions.LabelEncoder()
	
	encoder.fit(Y)
	encoded_Y = encoder.transform(Y)

	dummy_y = functions.utils.to_categorical(encoded_Y)
	
	print(dummy_y)

	inputdim=X.shape[1] 
	outputdim = dummy_y.shape[1]

	print(inputdim)
	print(outputdim)


	#output a configured model function with no inputs
	retmodel = functions.functools.partial( functions.selu_network , config.nnProps, inputdim, outputdim)   
	#set up the problem
	estimator = functions.KerasClassifier(build_fn=retmodel, epochs=5, batch_size=100, verbose=1)



###############################################learn ###########################################################
if config.learn == True:
	from sklearn.model_selection import KFold
	from sklearn.metrics import confusion_matrix
	from matplotlib import pyplot as plt

	kf= KFold(n_splits=3, shuffle=True, random_state=0)

	results = []
	
	for i , (train, test) in enumerate(kf.split(X)):
		X_train, X_test, y_train, y_test = X[train], X[test], dummy_y[train], dummy_y[test]
		encoded_Y_train, encoded_Y_test = encoded_Y[train], encoded_Y[test]
		estimator.fit( X_train, y_train)
		y_pred = estimator.predict( X_test)
		cnf_matrix = confusion_matrix(encoded_Y_test, y_pred)
		print(cnf_matrix)
		results.append(cnf_matrix)
    	#save network here
	cfn_final = np.sum(results)
	



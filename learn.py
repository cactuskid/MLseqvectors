import config
import functions
import keras_model

if config.load_data == True:
	from matplotlib import pyplot as plt
	import random
	#create dask arrays for X and Y to feed to keras
	print('loading dataset')
	filenames = glob.glob(config.savedir + config.dataname +'*.hd5')
	print(filenames)
	dataset = functions.DaskArray_hd5loadDataset(filenames , verbose=config.verbose)
	print('dataset loaded with keys:')
	print(dataset.keys())
	Y= dataset['Y']
	data = []
	for pipeline in config.pipelines:
		data.append( dataset[pipeline] )
		if config.visualize == True:
			for j in range(5):
				i = random.randint( 0, dataset[pipeline].shape[0] )
				plt.title(pipeline + str(Y[i].compute()))
				plt.plot(dataset[pipeline][i,:].compute())
				plt.save_fig( './figures/'+pipeline + str(Y[i].compute()+'.png'))
	X = functions.da.concatenate(data, axis = 1  )

######################################make NN ################################################################
if config.make_networkmodel ==True:

	from dask_ml.preprocessing import StandardScaler
	scaler = StandardScaler()
	functions.np.random.seed(0)
	# encode class values as integers
	encoder = functions.LabelEncoder()
	encoder.fit(Y.ravel())
	encoded_Y = encoder.transform(Y.ravel())
	dummy_y = functions.utils.to_categorical(encoded_Y)
	print(dummy_y)
	inputdim=X.shape[1]
	print('scaling X')
	X = scaler.fit_transform( X, Y )
	print('Done')
	outputdim = dummy_y.shape[1]
	print(inputdim)
	print(outputdim)
	#output a configured model function with no inputs
	retmodel = functions.functools.partial( functions.selu_network , config.nnProps, inputdim, outputdim)   
	#set up the problem

	#if distributed here
	#configure tensorflow estimator to use worker and ps tasks
	#setup some workers to deal with the tensorflow computations


	retmodel = configure_model(keras_model.selu_network, input_dim , output_dim , retmodel )
	
	if config.distributed == True:
		pass

	else:
		model,x,y,lossfun, tfOptimizer = retmodel()
		###############################################learn ###########################################################
	if config.learn == True and config.distributed ==False :
		from sklearn.model_selection import KFold
		from sklearn.metrics import confusion_matrix
		results = []
		kf= KFold(n_splits=3, shuffle=True, random_state=0)
		for i , (train, test) in enumerate(kf.split(X)):
			X_train, X_test, y_train, y_test = X[train], X[test], dummy_y[train], dummy_y[test]
			encoded_Y_train, encoded_Y_test = encoded_Y[train], encoded_Y[test]
			model.fit(self, x=X_train, y=y_train, batch_size=None, epochs=1, verbose=1, validation_split=0.0, validation_data=None,  class_weight=None, sample_weight=None)
			y_pred = model.predict( X_test)
			
			cnf_matrix = confusion_matrix(encoded_Y_test, y_pred)
			print(cnf_matrix)
			if config.visualize == True:
				functions.plot_confusion_matrix(cnf_matrix , encoder.classes_ )
				functions.plot_confusion_matrix(cnf_matrix , encoder.classes_ , normalize = True )
			results.append(cnf_matrix)
		cfn_final = functions.np.sum(results)
		functions.plot_confusion_matrix(cnf_matrix , encoder.classes_ )
		functions.plot_confusion_matrix(cnf_matrix , encoder.classes_ , normalize = True )
		model.save(config.model_path)




	###############################################wisdom ###########################################################

	if config.visualize_model == True:
		pass 
		#show model output at different layes to learn something from the features
		
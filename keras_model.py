
######################################neural network functions ################################################33


from keras import utils
from keras.models import Sequential
from keras.layers import Dense, Conv1D, Dropout, Input
from keras.wrappers.scikit_learn import KerasClassifier
from keras.utils import np_utils
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold , train_test_split
from sklearn.preprocessing import LabelEncoder , robust_scale , normalize
from sklearn.pipeline import Pipeline
from keras import regularizers
from keras.objectives import categorical_crossentropy
from keras import losses
from keras import optimizers
from keras.layers import Dense, Activation, Dropout , PReLU
from keras.layers.noise import AlphaDropout
import tensorflow as tf

#build a funtion to see the output of a layer
def get_output_function_N(model , layer):
	f = K.function([model.layers[0].input], [model.layers[layer].output])
	return f

def selu_network(  input_dim, output_dim , retmodel):
	"""
	# Returns
		A Keras model instance (compiled).
	"""

	nnProps = { 'nlayers':2 , 'mlayers':2, 'choke':40 , 'start':100 , 'end':50 , 'dropout_rate':.1  , 'activation':'selu',				   
				   "kernel_initializer":'lecun_normal',
				   'optimizer':'adam'
					}

	
	layers = hourglass( nnProps['nlayers'] , nnProps['mlayers'] ,nnProps['choke'] , nnProps['start'],nnProps['end'])
	model = Sequential()

	for i,size in enumerate(layers):
		if i==0:
			#add selu layer 1st
			model.add(Dense(size, kernel_initializer=nnProps['kernel_initializer'] , input_shape=(input_dim,) ))
			model.add(Activation(nnProps['activation']))
			model.add(AlphaDropout(nnProps['dropout_rate']))
		else:
			#add rectified linear units
			model.add(PReLU(alpha_initializer='zeros', alpha_regularizer=None, alpha_constraint=None, shared_axes=None))
			model.add(Dropout(nnProps['dropout_rate']))
	#output layer is categorical
	model.add(Dense(output_dim , activation= 'softmax'))
	#compile
	
	x = tf.placeholder(tf.float32, shape=(None, input_dim))

	#placeholder y
	y = model(x)
	#use tensorflow optimizer
	tfOptimizer = tf.train.AdamOptimizer
	optimizer = optimizers.TFOptimizer(tfOptimizer)
	lossfun = losses.categorical_crossentropy

	model.compile(loss=lossfun , optimizer=optimizer, metrics=['accuracy'])


	if retmodel == True:
		return model
	else:
		return model,x,y,lossfun, tfOptimizer

def configure_model(networkfun, input_dim , output_dim , retmodel ):
	return partial( networkfun , input_dim , output_dim , retmodel )


def hourglass(nlayers, mlayers, chokept, startneurons, endneurons):
	layersizes = []
	step1 = (chokept - startneurons )/ nlayers    
	step2 = (endneurons - chokept)/nlayers
	for i in range(nlayers):
		layersizes.append( int(startneurons + step1*i))    
	for i in range(mlayers):
		layersizes.append( int(chokept + step2*i))    
	return layersizes


def input_function(features,labels=None,shuffle=False):
    input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"posts_input": features},
        y=labels,
        shuffle=shuffle
    )
    #tensorflow input function
    return input_fn
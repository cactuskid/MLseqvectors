import functions
import config





propdict = loadDict('./physicalpropTable.csv')
print('physical properties of amino acids')
print(propdict)
window = sig.gaussian(nGaussian, std=stdv)
window /= np.sum(window)
print('gaussian filter for sequence physical props')
print(window)

hyperparams={'propdict': propdict  , 'Gaussian':window , 'clipfreq':500 }
pipeline_functions = [ seq2numpy,  seq2vec , gaussianSmooth, fftall , clipfft , retfinal_first ]
configured = []

for func in pipeline_functions:
    configured.append(functools.partial( func , hyperparams=hyperparams ) )

physicalProps_pipeline = compose(reversed(configured))

postives = [x[0] for x in os.walk(positive_datasets)]
folders = [ negative_dataset ]+ positives
if loadRandom == True:
    print('loading random entries to negative dataset')
    seqIter = dataGen([uniclust] , True )
    randomentries = iter_sample_fast( seqIter , negativesamples)
    SeqIO.write(randomentries, datadir + 'rando_uniclust.fasta', "fasta")
    print('done')


if Generate_XY = True
    datasets = []
    for folder in folders:
        #fourier transform all fastas
        fastas = glob.glob(folder +'*.fasta')
        seqDF = fastasToDF(fastas)
        seqDF['physical'] = seqDF['seq'].apply(physicalProps_pipeline)
        seqDF['phobius'] = seqDF['seq'].apply(phobius_pipeline)
        seqDF['psipred'] = seqDF['seq'].apply(psipred_pipeline)
        


        x = np.array( np.vstack(list(map(pipeline, seqIter ))) )
        #remove nans

        x = x[~np.isnan(x).any(axis=1)]
        y = [ folder ]*(x.shape[0])
        datasets.append( ( x,y) )
        print('DONE')
    x,y = zip(*datasets)
    Xtotal = np.vstack(x)
    Ytotal= np.concatenate(y)
    print('xmat')
    print(Xtotal.shape)
    print('ymat')
    print(Ytotal.shape)

    with open( datadir +'xdata.pkl' , 'wb') as handle:
        pickle.dump( Xtotal, handle, -1)

    with open( datadir+'ydata.pkl' , 'wb') as handle:
        pickle.dump( Ytotal, handle,-1)
        



#learn and validate 
#works

# define model. this could probably be a lot fancier... just used out of the box stuff
#isnt overfitting so we can def up the number of layers/nodes
def baseline_model(nlayers ,firstsize, hiddensize,  inputdim, outputdim , modeltype= 'normal' ):
    
    #try convolutional layer?
    #try dropout layer
    
    model = Sequential()
    
    if modeltype == 'convolutional':
        #finish me
        model.add(Conv1D(filters=30 , kernel_size=50 , strides=1, input_shape= ( 1, inputdim) ,  activation='relu'))
    else:
        model.add(Dense(firstsize , input_dim= inputdim  ))

    #added a dropout layer so it would generalize better
    
    model.add(Dropout(.5))

    for i in range(nlayers):
        model.add(Dense(hiddensize ))
    
    
    model.add(Dense(outputdim , activation = 'softmax'))
    # Compile model
    model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
    
    return model



with open( datadir + 'xdata.pkl' , 'rb') as handle:
    X = pickle.load(handle)

with open( datadir '/ydata.pkl' , 'rb') as handle:
    Y = pickle.load(handle)
    
#X = robust_scale(X)
X = normalize(X)

np.random.seed(0)

# encode class values as integers
encoder = LabelEncoder()
encoder.fit(Y)
encoded_Y = encoder.transform(Y)
dummy_y = np_utils.to_categorical(encoded_Y)


nlayers =3
firstsize = 3000
hiddensize = 300
inputdim=X.shape[1] 
outputdim = dummy_y.shape[1]

#output a configured model function with no inputs
retmodel = functools.partial( baseline_model , nlayers , firstsize, hiddensize, inputdim, outputdim , modeltype= 'normal' )    


#set up the problem
estimator = KerasClassifier(build_fn=retmodel, epochs=15, batch_size=15, verbose=1)


# In[6]:



from sklearn.metrics import confusion_matrix

from sklearn.model_selection import KFold

import itertools

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues ,outfile = './confusion.png'):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.save_fig(outfile, format ='png')


kf= KFold(n_splits=5, shuffle=True, random_state=0)

results = []
for train, test in kf.split(X):
    X_train, X_test, y_train, y_test = X[train], X[test], dummy_y[train], dummy_y[test]
    encoded_Y_train, encoded_Y_test = encoded_Y[train], encoded_Y[test]
    estimator.fit( X_train, y_train)
    y_pred = estimator.predict( X_test)
    
    
    # Compute confusion matrix
    cnf_matrix = confusion_matrix(encoded_Y_test, y_pred)
    results.append(cnf_matrix)
    plt.figure()
    plot_confusion_matrix(results[0], classes=folders.reverse() ,
                      title='Confusion matrix, without normalization')
    
    plt.show()

    plt.figure()

    plot_confusion_matrix(results[0], classes=folders.reverse() , normalize = True )

    plt.show()
    



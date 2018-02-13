#test functions
import functions
import config
import os
import glob

testOne = True
nGaussian = 7
stdv=.5
window = functions.sig.gaussian(nGaussian, std=stdv)
window /= functions.np.sum(window)

positive_datasets = config.workingdir + 'datasets/'

positives = [x[0] for x in os.walk(positive_datasets)]

propdict = functions.loadDict('./physicalpropTable.csv')
hyperparams={'propdict': propdict  , 'Gaussian':window , 'clipfreq':500, 'verbose' : True }


physical_pipeline_functions = [ functions.seq2numpy,  functions.seq2vec , functions.gaussianSmooth, functions.fftall , functions.clipfft , functions.retfinal_first ]
configured = []

for func in physical_pipeline_functions:
    configured.append(functions.functools.partial( func , hyperparams=hyperparams ) )
physicalProps_pipeline  = functions.compose(reversed(configured))


phobius_pipeline_functions = [ functions.runphobius,  functions.parsephobius, functions.gaussianSmooth, functions.fftall , functions.clipfft , functions.retfinal_first ]
configured = []
for func in phobius_pipeline_functions:
    configured.append(functions.functools.partial( func , hyperparams=hyperparams ) )
phobius_pipeline = functions.compose(reversed(configured))

for folder in positives:
	fastas = glob.glob(folder+'/*fasta')

	if len(fastas)>0:
		
		if testOne == True:
			for fasta in fastas:
				fastaIter = functions.SeqIO.parse(fasta, "fasta")
				for seq in fastaIter:
					seqstr = str(seq.seq)
					fastastr = '>'+str(seq.description)+'\n'+str(seq.seq)
					print(fastastr)
					#result = physicalProps_pipeline(seqstr)
					#print (result)
					result = phobius_pipeline(fastastr)
					print (result)

		if testOne == False:
			print(fastas)
			df = functions.fastasToDF(fastas)
			df['folder'] = folder
			print(df)
			df['physical'] = df['seq'].apply( physicalProps_pipeline )
			df['phobius'] = df['fasta'].apply( phobius_pipeline )
			print(df)



loadRandom = True
negativesamples = 10000
verbose = False

#gaussian smooth to apply physical properties of neighbors to each residue. tuneable kmer?
nGaussian = 7
stdv = .5

create_data = True
dataname = 'firstdataset'
#test one fasta through pipelines

testOne = False
#save dataset to hdf5
save_data_tohdf5 = True
#overwrite Hdf5 data
overwrite = True


load_data = True

make_networkmodel = True

learn = True


#generate a fasta with random entries from uniclust
generate_negative = False
#size of negative sample
NegSamplesize = 100000



#local folders
workingdir = './'

#physical properties table to use in dataset generation
proptable = './physicalpropTable.csv'
#save hdf5 matrices here
#savedir = '/scratch/cluster/monthly/dmoi/MachineLearning/'

savedir = './'


#where to find uniclust and a few other things
datadir = '/scratch/cluster/monthly/dmoi/MachineLearning/'

positive_datasets = workingdir + 'datasets/'
negative_dataset = datadir + 'truenegative/'

uniclust = datadir+ '/uniclust/uniclust30_2017_10/uniclust30_2017_10_seed.fasta'
scop = ''

#programs for topology prediction, used in dataset generation
phobius = './phobius/phobius.pl  '
garnier = ' garnier -filter '

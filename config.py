

verbose = False


#####run in distributed mode with a client, scheduler and workers######
#scheduler should be initialized outside of this script#
distributed = True
schedulerIP = "130.223.123.113:8786"
schedulerFile = ''



#gaussian smooth to apply physical properties of neighbors to each residue. tuneable kmer?
nGaussian = 7
stdv = .25

create_data = False
dataname = 'testdb'
ecodFasta = './datasets/ECOD/ecod.latest.F99.fasta'
RunWithEcod = False





pipelines = ['physical', 'garnier', 'phobius']

save_data_tohdf5 = True
#overwrite Hdf5 data
overwrite = True
load_data = True
visualize = False
make_networkmodel = True
learn = True


model_dir = './'


#generate a fasta with random entries from uniclust
generate_negative = False
#size of negative sample
NegSamplesize = 10000




#local folders
workingdir = '/home/cactuskid/Dropbox/IIB/archaeaReboot/Machinelearning/'

#output the trained neural network to file
model_path = '.NN.hdf'


#physical properties table to use in dataset generation
proptable = './physicalpropTable.csv'
#save hdf5 matrices here
#savedir = '/scratch/cluster/monthly/dmoi/MachineLearning/'

savedir = '/home/cactuskid/Dropbox/IIB/archaeaReboot/Machinelearning/'


#where to find uniclust and a few other things
datadir = '/scratch/cluster/monthly/dmoi/MachineLearning/'

positive_datasets = workingdir + 'datasets/'
negative_dataset = datadir + 'truenegative/'

uniclust = datadir+ '/uniclust/uniclust30_2017_10/uniclust30_2017_10_seed.fasta'
scop = ''

#programs for topology prediction, used in dataset generation
phobius = './phobius/phobius.pl  '
garnier = ' garnier -filter '
coils = ' ./coils/ncoils-osf '

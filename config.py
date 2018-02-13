
loadRandom = True
negativesamples = 10000

#gaussian smooth to apply physical properties of neighbors to each residue. tuneable kmer?
nGaussian = 7
stdv = .5


workingdir = './'
datadir = '/scratch/cluster/monthly/dmoi/MachineLearning/'
positive_datasets = workingdir + 'datasets/truepositive/'
negative_dataset = datadir + 'truenegative/'


uniclust = datadir+ 'uniclust/uniclust30_2017_10_seed.fasta'
scop = ''
phobius = './phobius/phobius.pl  '

psipred = ''

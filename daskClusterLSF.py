#dask script to setup dask cluster on LSF
#args are num workers, wall for workers
#loop keeps workers alive
#tensorflow cluster available as well\

from functools import partial
import keras_model
import distributed
import tensorflow as tf

from distributed.deploy import Adaptive
from distributed import Client , Scheduler , LocalCluster

from dask_tensorflow import start_tensorflow
import json
import time

from keras import backend as K

###################################################################Adaptive clsuter##################################################################################

def start_cluster(clustertype='local' , n = 8 , Adaptive_deploy = False):
	#add the json config file ?
	if clustertype == 'local':
		cluster = LocalCluster( n_workers = n )
	elif clustertype =='LSF':
		cluster = MyLSFCluster( n_workers = n )
	adapative_cluster = None
	if Adaptive_deploy == True:
		#turns on cluster size management
		adapative_cluster = Adaptive(cluster.scheduler, cluster)
	cluster.scheduler.start()
	client = Client(cluster)
	return client,cluster, adapative_cluster

###########################################################################tensorflow specific#####################################################################

def start_psworkers(client,tf_spec, dask_spec ,   verbose  ):
	#send tasks to dask workers	
	ps_tasks = [client.submit(ps_task , tf_spec = tf_spec, verbose= verbose , workers=worker ) for worker in dask_spec['ps']   ]

def start_workers(client,tf_spec, dask_spec , keras_model , verbose  ):
	worker_tasks = [client.submit(worker_task, tf_spec = tf_spec, verbose= verbose , keras_model = keras_model, workers=worker , pure=False) for worker in dask_spec['worker']]

def start_scorer(client,tf_spec, dask_spec , verbose  ):
	scorer_task = client.submit(scoring_task, tf_spec = tf_spec, verbose= verbose , workers=dask_spec['scorer'][0])

"""
def add_worker( client , keras_model , verbose , worker):
	worker_task = client.submit(workerfun, tf_spec = tf_spec, verbose= verbose , workers=worker , pure=False)

def add_ps( client,  tf_spec , verbose , worker):
	worker_task = client.submit(psfun, tf_spec = tf_spec, verbose= verbose , workers=worker , pure=False)
"""

def setup_tensorflow_cluster(clustertype='local' , TFservers= None, scale_up = False):
	client,dask_cluster, adapative_cluster = start_cluster(clustertype=clustertype, Adaptive_deploy = False)
	if TFservers == None :
		if 6 > len(dask_cluster.workers):
			cluster.scale_up(6)
		tf_spec, dask_spec = start_tensorflow(client, ps=1, worker=4, scorer=1)
	else:
		if len(TFservers) > len(dask_cluster.workers):
			cluster.scale_up(len(TFservers))
		tf_spec, dask_spec = start_tensorflow(client, ps=TFservers['ps'], worker=TFservers['worker'], scorer=TFservers['scorer'])
	return client, dask_cluster,  tf_spec, dask_spec

def ps_task(tf_spec, verbose = False):
	worker = distributed.get_worker()
	server = worker.tensorflow_server
	ps_device = "/job:%s/task:%d" % (server.server_def.job_name, server.server_def.task_index)
	if verbose == True:
		print ('PS task')
		print (ps_device)
	worker.tensorflow_server.join()

def scoring_task(tf_spec , xval, yval, keras_model , verbose = False):
	#run partial of this to configure it to xval and yval
	with local_client() as c:
		# Scores Channel
		scores = c.channel('scores', maxlen=10)
		worker = distributed.get_worker()
		queue = worker.tensorflow_queue
		server = worker.tensorflow_server
		# Make Model
		sess, _, _, _, _, loss = model(server, tf_spec , keras_model)

		# Testing Data
		test_data = {x: xval, y_: yval}
		# Main Loop
		while True:
			score = sess.run(loss, feed_dict=test_data)
			scores.append(float(score))
			time.sleep(1)

def worker_task(tf_spec, dask_spec , verbose , keras_model):                                                           
	if verbose == True:
		print('init worker')	
	worker = distributed.get_worker()
	queue = worker.tensorflow_queue
	server = worker.tensorflow_server
	worker_device = "/job:%s/task:%d" % (server.server_def.job_name, server.server_def.task_index)
	task_index = server.server_def.task_index
	is_chief = task_index == 0
	if is_chief == True:
		sess = tf.train.ChiefSessionCreator()
		sess.create_session()
		if verbose == True:
			print('IM THE CHIEF')
			print(worker.address)
			print('session')
			print(sess)
	else:            
		sess = tf.train.WorkerSessionCreator()
		sess.create_session()
		if verbose == True:
			print('IM A WORKER')
			print(worker.address)
			print('session')
			print(sess)

	if verbose == True:
		print(task_index)
		print(worker_device)

	sess, x, y_, train_step, global_step, loss = model(server,tf_spec ,dask_spec, modelfun )

	#todo change stopping condition here
	while not scores or scores.data[-1] > 1000:
		try:
			batch = queue.get(timeout=0.5)
		except Empty:
			continue

		train_data = {x: batch[0], y_: batch[1]}
		sess.run([train_step, global_step], feed_dict=train_data)

	
#adapt this for the keras models
def model(server, tf_spec , dask_spec , modelfun , sync_replicas = False , chkptdir='./' ):

	replicas_to_aggregate = len(dask_spec['worker'])	
	worker_device = "/job:%s/task:%d" % (server.server_def.job_name,
										 server.server_def.task_index)
	task_index = server.server_def.task_index
	is_chief = task_index == 0
	with tf.device(tf.train.replica_device_setter(worker_device=worker_device, cluster=tf_spec)):
		
		global_step = tf.Variable(0, name="global_step", trainable=False)
		
		model,x,y,lossfun, optimizer = modelfun()
		opt = model.optimizer
		labels = tf.placeholder(tf.float32, shape=(None, output_dim))
		loss = lossfun(labels, y)

		if sync_replicas:
			if replicas_to_aggregate is None:
				replicas_to_aggregate = num_workers
			else:
				replicas_to_aggregate = replicas_to_aggregate

			opt = tf.train.SyncReplicasOptimizer(
					  opt,
					  replicas_to_aggregate=replicas_to_aggregate,
					  total_num_replicas=num_workers,
					  name="sync_replicas")

		train_step = opt.minimize(loss , global_step=global_step)
		if sync_replicas:
			local_init_op = opt.local_step_init_op
			if is_chief:
				local_init_op = opt.chief_init_op

			ready_for_local_init_op = opt.ready_for_local_init_op

			# Initial token and chief queue runners required by the sync_replicas mode
			chief_queue_runner = opt.get_chief_queue_runner()
			sync_init_op = opt.get_init_tokens_op()

		init_op = tf.global_variables_initializer()

		train_dir = chkptdir

		if sync_replicas:
		  sv = tf.train.Supervisor(
			  is_chief=is_chief,
			  logdir=train_dir,
			  init_op=init_op,
			  local_init_op=local_init_op,
			  ready_for_local_init_op=ready_for_local_init_op,
			  recovery_wait_secs=1,
			  global_step=global_step)
		else:
		  sv = tf.train.Supervisor(
			  is_chief=is_chief,
			  logdir=train_dir,
			  init_op=init_op,
			  recovery_wait_secs=1,
			  global_step=global_step)

		sess_config = tf.ConfigProto(
			allow_soft_placement=True,
			log_device_placement=False,
			device_filters=["/job:ps", "/job:worker/task:%d" % task_index])

		# The chief worker (task_index==0) session will prepare the session,
		# while the remaining workers will wait for the preparation to complete.
		if is_chief:
			print("Worker %d: Initializing session..." % task_index)
		else:
			print("Worker %d: Waiting for session to be initialized..." % task_index)

		sess = sv.prepare_or_wait_for_session(server.target, config=sess_config)

		if sync_replicas and is_chief:
			# Chief worker will start the chief queue runner and call the init op.
			sess.run(sync_init_op)
			sv.start_queue_runners(sess, [chief_queue_runner])


		return sess, x, y_, train_step, global_step, loss

def transfer_dask_to_tensorflow(batch):
	worker = distributed.get_worker()
	worker.tensorflow_queue.put(batch)

def trainbatches(X,Y , dask_spec):
	#split up dask arrays 
	chunksX = X.to_delayed()
	chunksY = Y.to_delayed()
	batches = [batch for batch in zip(chunksX, chunksY)]
	dump = client.map(transfer_dask_to_tensorflow, batches, workers=dask_spec['worker'], pure=False)

def test():
	TFservers = {'ps':1 , 'worker':4 ,'scorer':1 }
	client, dask_cluster , tf, ds = setup_tensorflow_cluster('local', TFservers )

	model = keras_model
	return client, dask_cluster , tf, ds



def check_dead_tasks():
	pass

def relaunch():
	pass


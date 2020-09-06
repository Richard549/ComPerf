import sys, os, imp, gc, time, math, random, logging, json
import subprocess, signal, psutil
import numpy as np

from ComperfUtil import ensure_exists
from ComperfDataset import ComperfDataset
from ComperfModelling import ComperfModeller, Regularisation, convert_regularisation_enum_to_str
from ComperfAnalysis import run_sample_weighting_and_pca_transformation

from hyperopt import fmin, tpe, hp, Trials, STATUS_OK
from hyperopt.mongoexp import MongoTrials
from functools import partial
from hyperopt.pyll.base import scope

"""
	This objective is called by hyperopt, and itself calls a function in another file
	The function is defined in another file because weirdness occurred when this file imported itself
"""
def objective(hyperparams, params):

	import imp, os, logging, sys
	sys.path.insert(0,params['src_sys_path'])

	import ComperfOpt
	from hyperopt import STATUS_OK

	average_mse = ComperfOpt.run_model_evaluate_params(hyperparams, params)
	return {'loss': average_mse,  'status': STATUS_OK}

"""
	This is the wrapper around the ComperfOpt function that runs models and returns generalisation performance
"""
def find_best_hyperparams(
		experiment_name, # for saving the database (e.g. strassen)
		dataset_pickle_file,
		experiment_folder,
		models_folder,
		results_folder,
		experiment_repeat_index,
		k,
		should_standardise,
		pca_transformation,
		input_events,
		response_event,
		regularisation,
		symbols,
		should_sum,
		exclusive_cache_events,
		total_experiment_indexes, # list of indexes to sample from
		modeling_indexes_batch=5,
		modeling_indexes_start=5,
		hyperparams_filename = "hyperparams.json",
		num_parallel_trials = 5,
		port="1234"
	):

	experiment_repeat_index = 0
	ensure_exists(experiment_folder + "/" + results_folder + "/" + str(experiment_repeat_index))

	num_modeling_indexes = modeling_indexes_start
	current_modeling_indexes_best = 999.0
	global_best_hyperparameters = {}
	global_improvement = True
	while(global_improvement):

		num_modeling_indexes += modeling_indexes_batch
		if num_modeling_indexes > len(total_experiment_indexes):
			logging.info("Maximum number of modeling indexes reached in the hyperparamter optimisation wrapper.")
			break

		sample_modeling_indexes = np.random.choice(total_experiment_indexes,num_modeling_indexes,replace=False).tolist()
		
		logging.info("Running new hyperparameter optimisation using %d modeling indexes: %s", num_modeling_indexes, str(sample_modeling_indexes))

		params = {}
		params['src_sys_path'] = sys.path[0]
		params['dataset_pickle'] = dataset_pickle_file
		params['experiment_folder'] = experiment_folder
		params['models_folder'] = models_folder
		params['experiment_index'] = experiment_repeat_index
		params['k'] = 5
		params['should_standardise'] = should_standardise
		params['pca_transform'] = pca_transformation
		params['input_events'] = input_events
		params['response_event'] = response_event
		params['symbols'] = symbols
		params['should_sum'] = should_sum
		params['exclusive_cache_events'] = exclusive_cache_events
		params['modeling_indexes'] = sample_modeling_indexes
		params['timeout_seconds'] = 1800 # seconds per model (which we do k times)
		params['regulariser_str'] = convert_regularisation_enum_to_str(regulariser)

		opt_fn = partial(objective, params=params)

		space = {
			'num_layers': scope.int(hp.quniform('num_layers', 1, 5, 1)),
			'neurons_per_layer': scope.int(hp.quniform('neurons_per_layer', 5, 100, 1)),
			'learning_rate': hp.loguniform('learning_rate', -11.51, -6.9),
			'batch_size': scope.int(hp.uniform('batch_size', 0.5, 100.0)) # 10 = 1 percent, so 100 is 10% at a time
		}

		# start the database
		# start N workers with sufficient timeout
		# run the trials
		# stop the database

		database_directory = experiment_folder + "/mongodb"
		ensure_exists(database_directory)

		start_cmd = "mongod --dbpath " + database_directory + " --port " + str(port) + " --directoryperdb --journal --nohttpinterface --noprealloc --fork --logpath " + database_directory + "/db.log"
		stop_cmd = "mongod --shutdown --dbpath " + database_directory
		worker_start_cmd = "COMPERF_DEBUG=1 PYTHONPATH=" + str(params['src_sys_path']) + " hyperopt-mongo-worker --mongo=localhost:" + str(port) + "/" + str(experiment_name) + "_db --reserve-timeout=3600" # --workdir=/tmp"

		start_cmd_list = start_cmd.split(" ")
		stop_cmd_list = stop_cmd.split(" ")
		worker_start_cmd_list = worker_start_cmd.split(" ")

		database_proc_log = open(experiment_folder + "/mongodb/" + str(experiment_repeat_index) + "_start.log", 'w')
		start_proc = subprocess.Popen(start_cmd, shell=True, stdout=database_proc_log)
		start_proc.wait()
		database_proc_log.close()

		worker_procs = []

		for worker_id in range(num_parallel_trials):

			worker_log_filename = experiment_folder + "/mongodb/" + str(experiment_repeat_index) + "_worker_" + str(worker_id) + ".log"
			worker_cmd = worker_start_cmd
			logging.info("Running worker with: %s", worker_cmd)
			worker_proc = subprocess.Popen(worker_cmd, shell=True, stdout=open(worker_log_filename,'w'), stderr=subprocess.STDOUT)
			worker_procs.append(worker_proc)

		trials = MongoTrials('mongo://localhost:' + str(port) + '/' + str(experiment_name) + '_db/jobs', exp_key=str(experiment_repeat_index) + "_" + str(num_modeling_indexes))

		local_improvement = True

		try:

			trials_batch_size = 10
			local_best = 999.0
			local_best_hyperparameters = {}
			number_of_trials = 0 
			number_of_failed_batches = 0
			while(local_improvement):

				number_of_trials += trials_batch_size

				best = fmin(fn=opt_fn,
						space=space,
						algo=tpe.suggest,
						trials=trials,
						max_evals=number_of_trials)

				best_error = trials.best_trial['result']['loss']
				logging.info("The best hyperparameters were: " + str(best) + " with loss " + str(best_error))

				if best_error < local_best:
					logging.info("We locally improved from %f to %f", local_best, best_error)
					local_best = best_error
					local_best_hyperparameters = dict(best)
					number_of_failed_batches = 0
				else:
					logging.info("We haven't locally improved. This is the %d failure in a row.", number_of_failed_batches+1)
					# if we haven't done any better in this batch
					number_of_failed_batches += 1
					if number_of_failed_batches == 2:
						local_improvement = False

			# Has the local optimisation improved on global best significantly?
			difference = current_modeling_indexes_best - local_best

			if difference > 0.01:
				logging.info("Moving from %d to %d modelling indexes globally improved the generalisation error from %f to %f", num_modeling_indexes - modeling_indexes_batch, num_modeling_indexes, current_modeling_indexes_best, local_best)
				current_modeling_indexes_best = local_best
				global_best_hyperparameters = local_best_hyperparameters
				global_best_hyperparameters["num_modeling_indexes"] = num_modeling_indexes
			else:
				if difference > 0.0:
					current_modeling_indexes_best = local_best
					global_best_hyperparameters = local_best_hyperparameters
					global_best_hyperparameters["num_modeling_indexes"] = num_modeling_indexes
				logging.info("Moving from %d to %d modelling indexes did not significantly improve the global generalisation error, which ended at %f", num_modeling_indexes - modeling_indexes_batch, num_modeling_indexes, current_modeling_indexes_best)
				# stop increasing num_modeling_indexes
				global_improvement = False 

			logging.info("Trials after %d modelling indexes:", num_modeling_indexes)
			for trial in trials:
				logging.info(str(trial["misc"]["vals"].to_dict()) + " produced generalisation loss: " + str(trial["result"]["loss"]))

		except (AttributeError, ValueError) as e:
			logging.error("Hyperopt returned no results from trials")
			global_improvement = False
		except KeyboardInterrupt as e:
			logging.error("Handling interrupt, will now shutdown algorithm.")
			global_improvement = False

		# now stop the database
		logging.info("Closing hyperopt mongo database")
		stop_proc = subprocess.Popen(stop_cmd, shell=True)
		stop_proc.wait()

		# ensure the workers have terminated (they will timeout)
		logging.info("Waiting for workers to terminate.")
		for worker_proc in worker_procs:
			try:
				p = psutil.Process(worker_proc.pid)
				logging.debug("Attempting to kill %d", worker_proc.pid)
				# this is the shell that popen started

				for child in p.children():
					# this is the executed shell that runs the monogo worker script
					for sub_child in child.children():
						# this is the mongo worker process
						logging.debug("Killing %d", sub_child.pid)
						sub_child.terminate()
						sub_child.kill()
					logging.debug("Killing %d", child.pid)
					child.terminate()
					child.kill()
				p.terminate()
				p.kill()
				worker_proc.terminate() # just in case
				logging.debug("Killing %d", worker_proc.pid)
			except OSError as e:
				logging.debug("Killed a process twice.")
				pass
			except psutil.NoSuchProcess as e:
				logging.debug("Killed a process twice.")
				pass
				
			worker_proc.wait()

	# Now save the best hyperparameters into the experiment folder
	logging.info("Saving the best hyperparameters %s which gave weighted generalisation loss: %f"+ str(global_best_hyperparameters), current_modeling_indexes_best)
	with open(experiment_folder + "/" + hyperparams_filename,'w') as f:
		json.dump(global_best_hyperparameters,f)

	return global_best_hyperparameters

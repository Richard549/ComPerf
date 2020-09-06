import imp, logging, json
import numpy as np

from ComperfUtil import initialise_logging, LogLevelOption, ensure_exists
from ComperfDataset import ComperfDataset, Target
from ComperfModelling import ComperfModeller, get_experiment_indexes_fixed_partitioning, Regularisation, convert_regularisation_str_to_enum
from ComperfOptWrapper import find_best_hyperparams
from ComperfAnalysis import run_sample_weighting_and_pca_transformation

"""
	TODO this should be merged with the ComperfRunner calls so we only define this process (or part of this process once)
"""
def run_model_evaluate_params(hyperparams, params):

	dataset_pickle = params['dataset_pickle']
	experiment_folder = params['experiment_folder']
	models_folder = params['models_folder']
	experiment_index = params['experiment_index']
	k = params['k']
	should_standardise = params['should_standardise']
	pca_transform = params['pca_transform']
	input_events = params['input_events']
	response_event = params['response_event']
	symbols = params['symbols']
	should_sum = params['should_sum']
	exclusive_cache_events = params['exclusive_cache_events']
	timeout_seconds = params['timeout_seconds']
	regulariser_str = params['regulariser_str']
	regularisation = convert_regularisation_str_to_enum(regularizer_str)

	modeling_indexes = params['modeling_indexes'] # while this is a hyperparameter, it is controlled by a custom wrapper, not hyperopt

	# hyperparameters:
	num_layers = hyperparams['num_layers']
	neurons_per_layer = hyperparams['neurons_per_layer']
	learning_rate = hyperparams['learning_rate']
	batch_size_pc = hyperparams['batch_size']

	logging.info("Running model evaluation for hyperparameters: %s", str(hyperparams))

	# Do not need to pass in the parameters to load the dataset, because we expect an already correct pickled version to exist
	dataset = ComperfDataset(
		input_events=input_events,
		response_event=response_event,
		require_pickle=True)
	
	architecture = []
	for layer_idx in range(num_layers):
		architecture.append(neurons_per_layer)
	
	modeller = ComperfModeller(
		experiment_folder + "/" + models_folder + "_" + str(experiment_index),
		architecture=architecture,
		learning_rate=learning_rate,
		batch_size=batch_size_pc,
		patience=50,
		max_training_epochs=10000,
		regulariser=regularisation)

	saved_training_event_stats = []
	testing_results_across_kfolds = []

	random.shuffle(modeling_indexes)
	num_modeling_indexes = len(modeling_indexes)

	partitions_seen_so_far = {} # dict where key is the validation set as a sorted string, where the values is a dict where key is the training set as a sorted string, whose value is then the number of required models

	for k_fold_idx in range(k):

		if k_fold_idx % num_modeling_indexes == 0:
			random.shuffle(modeling_indexes)

		k_fold = k_fold_idx % num_modeling_indexes

		# the testing set will have 1 index until we have 10 modeling indexes, at which point we will then have 2
		# this means that between 6 and 9 for example, we randomly select a subset of all the possible holdout sets
		# this is also the case with the validation indexes - we randomly select a validation set (without replacement) that is the same size as the testing set each time

		num_holdout_indexes = int(float(num_modeling_indexes)/k)
		testing_indexes = modeling_indexes[k_fold*num_holdout_indexes:(k_fold+1)*num_holdout_indexes] # this will never overflow, but there might be some remaining at the end not used for testing

		validation_indexes = modeling_indexes[(k_fold+1)*num_holdout_indexes:(k_fold+2)*num_holdout_indexes]
		if len(validation_indexes) < num_holdout_indexes:
			additional_required = num_holdout_indexes - len(validation_indexes)
			validation_indexes = validation_indexes + modeling_indexes[:additional_required]

		training_indexes = [index for index in modeling_indexes if index not in testing_indexes and index not in validation_indexes]
		
		validation_set_identifier = str(sorted(validation_indexes))
		training_set_identifier = str(sorted(training_indexes))
		if validation_set_identifier in partitions_seen_so_far:
			if training_set_identifier in partitions_seen_so_far[validation_set_identifier]:
				partitions_seen_so_far[validation_set_identifier][training_set_identifier] += 1
			else:
				partitions_seen_so_far[validation_set_identifier][training_set_identifier] = 1
		else:
			partitions_seen_so_far[validation_set_identifier] = {training_set_identifier:1}

		num_required_models = partitions_seen_so_far[validation_set_identifier][training_set_identifier]

		constant_events = []

		params = {}
		params["input_events"] = input_events
		params["output_event"] = response_event
		params["specific_symbols"] = symbols
		params["standardised"] = should_standardise
		params["summed"] = should_sum
		params["exclusive_cache_events"] = exclusive_cache_events

		params["constant_events"] = constant_events
		params["is_training_set"] = True
		params["set_indexes"] = training_indexes

		training_taskset, training_event_stats, ranks_to_label_strings = dataset.get_taskset(**params)

		model_input_events = [event for event in model_input_events if event not in constant_events]
		saved_training_event_stats.append(training_event_stats)

		params["input_events"] = model_input_events
		params["is_training_set"] = False
		params["training_event_stats"] = training_event_stats

		params["set_indexes"] = validation_indexes
		validation_taskset, _, _ = dataset.get_taskset(**params)

		params["set_indexes"] = testing_indexes
		testing_taskset, _, _ = dataset.get_taskset(**params)

		training_responses = training_taskset[2]
		validation_responses = validation_taskset[2]
		testing_responses = testing_taskset[2]

		batch_size = int((float(batch_size_pc)/1000.0) * len(training_taskset[2])) # batch_size_pc 10 means 1 percent
		logging.debug("Batch size is %d of %d total examples", batch_size, len(training_taskset[2]))
		modeller.batch_size = batch_size

		# Run the PCA transformations if set
		transformation_results = run_sample_weighting_and_pca_transformation(training_taskset,
			validation_taskset,
			testing_taskset,
			should_standardise,
			pca_transformation,
			training_event_stats)

		training_taskset = transformation_results[0]
		validation_taskset = transformation_results[1]
		testing_taskset = transformation_results[2]
		training_sample_weights = transformation_results[3]
		validation_sample_weights = transformation_results[4]
		testing_sample_weights = transformation_results[5]
		training_pca_matrix = transformation_results[6]
	
		# Now get the models or train them if they don't exist:
		models = modeller.get_trained_models(
			training_set=training_taskset,
			validation_set=validation_taskset,
			testing_set=testing_taskset)

		if len(models) < num_required_models:

			# We only train one new model
			models = modeller.train_models(
				training_set=training_taskset,
				validation_set=validation_taskset,
				testing_set=testing_taskset,
				repeats=1,
				mem_alloc=0.048,
				training_sample_weights=training_sample_weights,
				validation_sample_weights=validation_sample_weights)
		
		test_mse = models[-1].errors[str(sorted(testing_indexes))]
		testing_results_across_kfolds.append(test_mse)
		logging.info("Hyperparameter testing set results are MSE %f, RMSE %f, RMSE destandardised %f.", test_mse, np.sqrt(test_mse), np.sqrt(test_mse)*training_event_stats[-1][1])

	# I now have testing results across all k-folds

	average_mse = np.mean(testing_results_across_kfolds)
	logging.info("The testing results for %s were %s, with average MSE %f", str(hyperparams), str(testing_results_across_kfolds), average_mse)

	return average_mse

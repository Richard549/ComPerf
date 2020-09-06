import imp, logging, json
import numpy as np
from argparse import ArgumentParser, ArgumentTypeError

from ComperfUtil import initialise_logging, LogLevelOption, ensure_exists
from ComperfDataset import ComperfDataset, Target
from ComperfModelling import ComperfModeller, get_experiment_indexes_fixed_partitioning, Regularisation, convert_regularisation_str_to_enum, run_linear_regression
from ComperfOptWrapper import find_best_hyperparams
from ComperfAnalysis import run_sample_weighting_and_pca_transformation, calculate_mae, performance_comparison
import ComperfResults

def run_comperf(
		experiment_folder,
		Config,
		configurations,
		target,
		should_reload,
		dataset_pickle_file,
		total_experiment_indexes,
		hyperparameter_filename,
		k,
		num_parallel_trials,
		regularisation,
		experiment_repeat_index,
		k_fold_idx,
		num_repeat_models_per_k_fold
		):

	# Load the dataset
	dataset = ComperfDataset(Config.input_events,
		Config.response_event,
		configurations,
		Config.datasets_prefix,
		Config.max_num_datasets,
		Config.dataset_delimiter,
		Config.filename_suffix,
		Config.should_sum,
		target,
		dataset_pickle_file,
		should_reload,
		Config.should_preload)

	model_input_events = Config.input_events + Config.rusage_events + Config.syscall_events + Config.runtime_events
	response_event = Config.response_event.lower()
	all_events = dataset.events

	# Reduce the model input events to only those that we have in the dataset
	unavailable_events = [event.lower() for event in model_input_events if event.lower() not in dataset.events]
	if len(unavailable_events) > 0:
		logging.debug("There are %s requested input events that are unavailable in the dataset, so removing them from the model input.")
		logging.debug("They were: %s", len(unavailable_events), str(unavailable_events))
		model_input_events = [event.lower() for event in model_input_events if event.lower() in dataset.events]

	# Get hyperparams or run hyperopt if hyperparams are unavailable
	if os.path.isfile(hyperparameter_filename) == False:
		logging.info("Could not find existing hyperparameter file %s. Running hyperparameter optimisation", hyperparameter_filename)

		best_hyperparameters = find_best_hyperparams(
			experiment_name=Config.experiment_name, 
			dataset_pickle_file=dataset_pickle_file,
			experiment_folder=experiment_folder,
			models_folder=Config.models_folder,
			results_folder=Config.results_folder,
			experiment_repeat_index=0,
			k=k,
			should_standardise=Config.should_standardise,
			pca_transformation=Config.pca_transformation,
			input_events=model_input_events,
			response_event=Config.response_event,
			regularisation=regularisation,
			symbols=Config.symbols,
			should_sum=Config.should_sum,
			exclusive_cache_events=Config.exclusive_cache_events,
			total_experiment_indexes=total_experiment_indexes,
			modeling_indexes_batch=Config.num_profiles_batch_size,
			modeling_indexes_start=Config.num_profiles_start_size,
			hyperparams_filename=Config.hyperparams_filename,
			num_parallel_trials=num_parallel_trials,
			port=Config.mongodb_port)

		logging.info("Aftering running hyperparameter optimisation, the best hyperparameters were: %s", str(best_hyperparameters))

	hyperparams = {}
	with open(hyperparameter_filename,'r') as f:
		hyperparams = json.load(f)
		if len(hyperparams) == 0:
			logging.error("The loaded hyperparamters from %s are invalid.", hyperparameter_filename)
			raise ValueError()
		
	logging.info("Loaded hyperparameters: %s", str(hyperparams))

	num_modeling_indexes = int(hyperparams["num_modeling_indexes"])
	num_validation_indexes = int(num_modeling_indexes/k)
	num_repeat_models_per_k_fold = 1

	architecture = [int(hyperparams["neurons_per_layer"]) for layer in range(int(hyperparams["num_layers"]))]
	learning_rate = float(hyperparams["learning_rate"])
	batch_size_pc = float(hyperparams["batch_size"])

	# Build the modeller

	modeller = ComperfModeller(
		experiment_folder + "/" + Config.models_folder + "_" + str(experiment_repeat_index),
		architecture=architecture,
		learning_rate=learning_rate,
		batch_size=batch_size_pc,
		patience=50,
		max_training_epochs=10000,
		regulariser=regularisation)

	# Get the task set indexes that are appropriate for the requested experiment repeat and k-fold index
	testing_indexes, training_indexes, validation_indexes = get_experiment_indexes_fixed_partitioning(
		experiment_repeat_index,
		k_fold_idx,
		num_total_modeling_indexes=dataset.get_num_tasksets(),
		experiment_modeling_size=num_modeling_indexes,
		num_validation_indexes=num_validation_indexes,
		num_experiments=30,
		folder=experiment_folder)
	
	logging.info("Experiment repeat %d and k-fold index %d has %d testing indexes: %s", experiment_repeat_index, k_fold_idx, len(testing_indexes), testing_indexes)
	logging.info("Experiment repeat %d and k-fold index %d has %d validation indexes: %s", experiment_repeat_index, k_fold_idx, len(validation_indexes), validation_indexes)
	logging.info("Experiment repeat %d and k-fold index %d has %d training indexes: %s", experiment_repeat_index, k_fold_idx, len(training_indexes), training_indexes)

	constant_events = []

	# Get the task sets for these indexes
	# Each set should be formed as (list_taskset_indexes, standardised examples, responses, label_ranks)

	params = {}
	params["input_events"] = model_input_events
	params["output_event"] = response_event
	params["specific_symbols"] = Config.symbols
	params["standardised"] = Config.should_standardise
	params["summed"] = Config.should_sum
	params["exclusive_cache_events"] = Config.exclusive_cache_events

	params["constant_events"] = constant_events
	params["is_training_set"] = True
	params["set_indexes"] = training_indexes

	training_taskset, training_event_stats, ranks_to_label_strings = dataset.get_taskset(**params)

	model_input_events = [event for event in model_input_events if event not in constant_events]
	
	# Load the training indexes but get stats for *all* events (not just the model inputs)
	# This is simply to dump information to file for later analysis, does not affect the modelling:
	params["input_events"] = all_events
	_, all_event_stats, _ = dataset.get_taskset(**params)

	# I use the statistics generated from the training sets to standardise the validation/testing sets
	# So that the training set is the only data source for any manipulation of validation/testing sets
	params["input_events"] = model_input_events
	params["is_training_set"] = False
	params["training_event_stats"] = training_event_stats

	params["set_indexes"] = validation_indexes
	validation_taskset, _, _ = dataset.get_taskset(**params)

	params["set_indexes"] = testing_indexes
	testing_taskset, _, _ = dataset.get_taskset(**params)

	batch_size = int((float(batch_size_pc)/1000.0) * len(training_taskset[2])) # batch_size_pc 10 means 1 percent
	logging.debug("Batch size is %d of %d total examples", batch_size, len(training_taskset[2]))
	modeller.batch_size = batch_size

	# Run the PCA transformations if set
	transformation_results = run_sample_weighting_and_pca_transformation(training_taskset,
		validation_taskset,
		testing_taskset,
		Config.should_standardise,
		Config.pca_transformation,
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

	num_already_trained = len(models)

	if num_already_trained < num_repeat_models_per_k_fold:

		num_required_models = num_repeat_models_per_k_fold - num_already_trained

		new_models = modeller.train_models(
			training_set=training_taskset,
			validation_set=validation_taskset,
			testing_set=testing_taskset,
			repeats=num_required_models,
			mem_alloc=0.048,
			training_sample_weights=training_sample_weights,
			validation_sample_weights=validation_sample_weights)

		models.extend(new_models)

	logging.info("Finished loading %d models and training %d models!", num_already_trained, len(models) - num_already_trained)

	# Now run linear regressions
	# des meaning "destandardised"
	lin_mae, lin_mse, lin_rmse, lin_coeff_fs_des = run_linear_regression(
		training_taskset,
		testing_taskset,
		pca_matrix,
		training_event_stats)

	# Then collect the accuracy results for both the linear regressions and the neural networks
	ann_rmses = []
	for model in models:
		mse = model.errors[str(sorted(testing_indexes))]
		rmse = math.sqrt(mse)
		ann_rmses.append(rmse)

	ann_rmse = np.mean(ann_rmses)

	lin_rmse_des = lin_rmse * training_event_stats[-1][1]
	ann_rmse_des = ann_rmse * training_event_stats[-1][1]

	lin_mae_des = lin_mae * training_event_stats[-1][1]
	ann_mae_des = calculate_mae(models, testing_taskset) * training_event_stats[-1][1]

	# Then write the accuracy results, linear coefficients, and (all) event stats to file
	results_folder_full = experiment_folder + "/" + Config.results_folder

	ComperfResults.write_event_stats_to_file(results_folder_full,
		experiment_repeat_index,
		k_fold_idx,
		all_events,
		all_event_stats)

	ComperfResults.write_model_performance_to_file(results_folder_full,
		experiment_repeat_index,
		k_fold_idx,
		ann_rmse_des,
		lin_rmse_des,
		ann_mae_des,
		lin_mae_des)

	ComperfResults.write_linreg_coefficients_to_file(results_folder_full,
		experiment_repeat_index,
		k_fold_idx,
		model_input_events,
		lin_coeff_fs_des)

	# Then run the comparisons across configurations, using the neural networks (via DeepLIFT) and the linear regression coefficients

	# Get the reference

	params = {}
	params["input_events"] = model_input_events
	params["output_event"] = response_event
	params["specific_symbols"] = Config.symbols
	params["standardised"] = Config.should_standardise
	params["summed"] = Config.should_sum
	params["exclusive_cache_events"] = Config.exclusive_cache_events
	params["constant_events"] = constant_events
	params["is_training_set"] = False
	params["training_event_stats"] = training_event_stats
	params["set_indexes"] = testing_indexes

	reference_configuration_idx = 0 # reference is the first configuration in the list
	params["benchmark_id"] = reference_configuration_idx

	reference_taskset, reference_taskset_fs, reference_taskset_all_events_fs_destandardised = get_configuration_tasksets(
		dataset,
		all_events,
		training_pca_matrix,
		params)

	# Then for each target configuration, get the target taskset, apply the comparative analysis, then report it
	for configuration_idx, configuration_identifier in enumerate(configurations):
	
		logging.info("Running comparative analysis between configuration %d and configuration %d of %d: %s", reference_configuration_idx, configuration_idx, len(configurations), str(configuration_identifier))

		params["benchmark_id"] = configuration_idx
		target_taskset, target_taskset_fs, target_taskset_all_events_fs_destandardised = get_configuration_tasksets(
			dataset,
			all_events,
			training_pca_matrix,
			params)

		comparison_results = performance_comparison(
			reference_taskset,
			reference_taskset_fs,
			reference_taskset_all_events_fs_destandardised,
			target_taskset,
			target_taskset_fs,
			target_taskset_all_events_fs_destandardised,
			models,
			training_event_stats,
			training_pca_matrix,
			lin_coeff_fs_des)

		average_contributions_per_event = comparison_results[0]
		average_linreg_contributions_per_event = comparison_results[1]
		average_coefficients_per_event = comparison_results[2]
		average_input_event_value_variations_per_event = comparison_results[3]
		average_all_event_value_variations_per_event = comparison_results[4]
		average_pred_duration_variation = comparison_results[5]
		average_true_duration_variation = comparison_results[6]

		ComperfResults.write_performance_comparison_results_to_file(
			results_folder_full,
			experiment_repeat_index,
			k_fold_idx,
			configuration_identifier,
			input_events,
			all_events,
			average_contributions_per_event,
			average_linreg_contributions_per_event,
			average_coefficients_per_event,
			average_input_event_value_variations_per_event,
			average_all_event_value_variations_per_event,
			average_pred_duration_variation,
			average_true_duration_variation)

hyperparams = {}
with open(hyperparameter_filename,'r') as f:
	hyperparams = json.load(f)
	if len(hyperparams) == 0:
		log.fatal("Loaded hyperparamters " + str(hyperparameter_filename) + " are invalid")
		exit(1)

def log_level_option(input_string):
	log_level_option_int = 0
	try:
		log_level_option_int = int(input_string)
		enum_value = LogLevelOption(log_level_option_int)
	except ValueError:
		raise ArgumentTypeError("Log level option not recognised. See usage.")
	return enum_value

def parse_args():

	parser = ArgumentParser()
	optional = parser._action_groups.pop() # will return the default 'optional' options-group (including the 'help' option)

	required = parser.add_argument_group('required arguments')
	required.add_argument('-f', '--experiment_folder', required=True, help="Experiment base folder (containing config.py).")
	required.add_argument('-r', '--experiment_repeat_index', required=True, help="Assuming repeating the modelling and analysis multiple times, provide the repeat index (to get the correct train/val/test partitions).")
	required.add_argument('-i', '--k_fold_idx', required=True, help="For the repeat index, what k-fold index are we using in this run?")
	
	ll_options_str = ", ".join([str(member.value) + "=" + str(name) for name,member in LogLevelOption.__members__.items()])

	optional.add_argument('-d', '--log_level', type=log_level_option, default=LogLevelOption.INFO, help="Logging level. Options are:" + ll_options_str + ".")
	optional.add_argument('--reload', action='store_true', help="Reload the dataset, even if a serialised version already exists in the experiment folder.")
	optional.add_argument('--tee', action='store_true', help="Pipe logging messages to stdout as well as the log file.") 

	parser._action_groups.append(optional)

	args = parser.parse_args()

	logfile = args.experiment_folder + "/log.txt" # hardcoded log file
	return args.experiment_folder, logfile, args.log_level, args.tee, args.reload, args.experiment_repeat_index, args.k_fold_idx

def parse_target(target_str):
	if target_str == "work":
		return Target.WORK
	elif target_str == "nonwork":
		return Target.NONWORK
	elif target_str == "work_and_nonwork":
		return Target.ALL
	else:
		logging.error("Do not recognise target: %s", target_str)
		raise ValueError()

# Load the user-options and experiment base folder

experiment_folder, logfile, log_level, tee_mode, should_reload, experiment_repeat_index, k_fold_idx = parse_args()
initialise_logging(logfile, log_level, tee_mode)

Experiment = imp.load_source("Experiment", experiment_folder + "/config.py")
from Experiment import Config

# Get the experiment parameters

configurations = Config.get_configurations()
target = parse_target(Config.target)
regularisation = convert_regularisation_str_to_enum(Config.regularisation)
total_experiment_indexes = list(range(Config.num_profiles_per_experiment))
num_repeat_models_per_k_fold = 1 # hardcoded

dataset_pickle_file = experiment_folder + "/" + Config.pickle_file
hyperparameter_filename = experiment_folder + "/" + Config.hyperparams_filename

# Make sure the folders exist

ensure_exists(experiment_folder + "/" + Config.results_folder)
ensure_exists(experiment_folder + "/" + Config.models_folder)

k = 5
num_parallel_trials = 5

np.set_printoptions(suppress=True,precision=3)
logging.info("Running ComPerf for the experiment folder: %s", experiment_folder)

run_comperf(
	experiment_folder,
	Config,
	configurations,
	target,
	should_reload,
	dataset_pickle_file,
	total_experiment_indexes,
	hyperparameter_filename,
	k,
	num_parallel_trials,
	regularisation,
	experiment_repeat_index,
	k_fold_idx,
	num_repeat_models_per_k_fold)

logging.info("Done")

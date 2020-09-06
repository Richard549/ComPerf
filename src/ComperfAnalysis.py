import logging
import numpy as np

from ComperfModelling import predict_from_model

def run_kernel_density_estimation(values, normalised=False, normalisation_statistic=0):

	values = values.reshape(-1,1)

	# What should bandwidth be? Add as part of experiment configuration?
	bandwidth = 100000
	evaluation_gap = 50000

	if normalised:
		bandwidth = float(bandwidth)/normalisation_statistic
		evaluation_gap = float(evaluation_gap)/normalisation_statistic

	kde = KernelDensity(kernel='gaussian', bandwidth=bandwidth).fit(values)

	return kde, bandwidth, evaluation_gap

def compute_sample_weights_from_kde(kde, values, normalised=False, normalisation_statistic=0.0):

	# It is very expensive to evaluate every single example, so bin the examples into bins of 100k cycles, and apply the same weight to all examples within a given bin

	range_max = np.amax(values)
	range_min = np.amin(values)

	# Making the bin size equal to the bandwidth of the KDE
	bin_size = 100000
	if normalised:
		bin_size = bin_size/normalisation_statistic

	bins = int((float(range_max) - range_min)/float(bin_size))

	bin_positions = np.linspace(range_min,range_max,bins) + (bin_size/2.0) # i.e. evaluate the middle of each bin

	density_estimations = np.exp(kde.score_samples(bin_positions.reshape(-1,1)))

	# clip density estimations for bins that very likely don't contain any examples, as otherwise sample weights explode
	density_estimations[density_estimations < 0.1] = np.nan
	sample_weights = np.reciprocal(density_estimations)
	sample_weights[np.isnan(sample_weights)] = 10

	# now, apply the correct weight to each original estimation_position
	estimation_position_corresponding_bins = ((values - range_min) / bin_size).astype(np.int32)
	estimation_position_corresponding_bins[estimation_position_corresponding_bins >= bins] = bins-1

	per_example_sample_weights = sample_weights[estimation_position_corresponding_bins]

	logging.debug("Clipped %d examples to have a maximum sample weight of 10", len(per_example_sample_weights[per_example_sample_weights == 10]))
	
	return per_example_sample_weights

def kde_clustering_get_indexes(values, normalised=False, normalisation_statistic=0):

	counts = values.reshape(-1,1)

	kde, bandwidth, evaluation_gap = run_kernel_density_estimation(counts, normalised, normalisation_statistic)

	num_divisions = int(np.amax(counts)-np.amin(counts))/evaluation_gap
	if num_divisions < 20:
		# i.e. if we evaluate the density at less than this number of divisions, then we can't possibly cluster...
		return None, [], []

	estimation_positions = np.linspace(np.amin(counts),np.amax(counts),num_divisions)
	estimation_positions = estimation_positions.reshape(-1,1)

	estimations = np.exp(kde.score_samples(estimation_positions))

	# form a minimum by starting at 1.0 then descending to 0.0, then continuing with the histogram
	estimations = np.insert(estimations,0,0.0,axis=0)
	estimation_positions = np.insert(estimation_positions,0,np.amin(counts)-evaluation_gap,axis=0)
	estimations = np.insert(estimations,0,1.0,axis=0)
	estimation_positions = np.insert(estimation_positions,0,np.amin(counts)-(2*evaluation_gap),axis=0)

	mi, ma = argrelextrema(estimations, np.less_equal)[0], argrelextrema(estimations, np.greater)[0]

	bounds = []

	populated_cluster_id = 0

	cluster_indexes = []
	cluster_bounds = []

	for idx in np.arange(1,len(mi)):

		lower_bound = estimation_positions[mi[idx-1]]

		if idx == (len(mi)-1):
			upper_bound = np.amax(counts)+1
		else:
			upper_bound = estimation_positions[mi[idx]]

		clustered_task_indexes = np.where(np.logical_and(counts >= lower_bound, counts < upper_bound))[0]

		if(len(clustered_task_indexes) > 0):
			cluster_indexes.append(clustered_task_indexes)
			cluster_bounds.append([lower_bound,upper_bound])

	return kde, cluster_indexes, cluster_bounds

"""
	This function aims to identify significant differences between the responses, and saving the examples to later investigate as a comparative study
	A comparison may be between individual tasks or two task sets, where we necessarily average the latter
"""
def get_comparisons(responses):
	
	# should also save the cluster details so we can later use them to repeat the equivalent analysis on separate testing sets

	individual_comparisons = []
	cluster_comparisons = []
	cluster_comparison_bounds = []
		
	central_cluster_indexes = []
	central_cluster_bounds = []
	individual_comparisons = []
	cluster_comparisons = []
	kde = None

	# return a list of pairs, where each element is a list of task indexes to compare against eachother
	# return two types of comparisons: closest-to-mean task, tasks within 1*std-deviation of mean

	kde, clustered_task_indexes, cluster_bounds = kde_clustering_get_indexes(responses)
	if kde is None:
		log.info("Could not cluster instances as there is no significant variation.")
		return central_cluster_indexes, central_cluster_bounds, individual_comparisons, cluster_comparisons, kde

	# filter to only the significant clusters?
	large_clusters = []
	minimum_cluster_size = max([int(0.01*len(responses)),10])
	for cluster_idx, cluster_indexes in enumerate(clustered_task_indexes):
		if len(cluster_indexes) >= minimum_cluster_size:
			large_clusters.append(cluster_idx)

	logging.info("Found %d significant clusters (minimum cluster size was: %d)", len(large_clusters), minimum_cluster_size)

	clustered_task_indexes = [cluster_indexes for cluster_idx, cluster_indexes in enumerate(clustered_task_indexes) if cluster_idx in large_clusters]
	cluster_bounds = [cluster_indexes for cluster_idx, cluster_indexes in enumerate(cluster_bounds) if cluster_idx in large_clusters]

	if len(clustered_task_indexes) < 2:
		logging.info("There were fewer than two clusters, so cannot do a comparative analysis")
		return central_cluster_indexes, central_cluster_bounds, individual_comparisons, cluster_comparisons, kde

	fast_cluster = clustered_task_indexes[0]
	slow_clusters = clustered_task_indexes[1:]

	fast_durations = responses[fast_cluster]
	mean_fast_task_idx = (np.abs(fast_durations - np.mean(fast_durations))).argmin()
	mean_fast_task_idx_original_set = fast_cluster[mean_fast_task_idx]

	constraint_coefficient = 0.5
	fast_lower_central_bound = np.mean(fast_durations) - constraint_coefficient * np.std(fast_durations)
	fast_upper_central_bound = np.mean(fast_durations) + constraint_coefficient * np.std(fast_durations)
	central_fast_task_idxs = np.where(np.logical_and(fast_durations>=fast_lower_central_bound, fast_durations<=fast_upper_central_bound))[0]

	fast_durations_within_constraint = fast_durations[central_fast_task_idxs].flatten()
	distances_from_mean = np.absolute(fast_durations_within_constraint - np.mean(fast_durations))
	distances_sorted_indexes = np.argsort(distances_from_mean)

	# now filter the central indexes to only those that are within the bounds and are the closest to mean
	central_fast_task_idxs = central_fast_task_idxs[distances_sorted_indexes[:50]]

	# update the bounds
	fast_lower_central_bound = np.amin(fast_durations[central_fast_task_idxs])
	fast_upper_central_bound = np.amax(fast_durations[central_fast_task_idxs])

	central_fast_task_idxs_original_set = fast_cluster[central_fast_task_idxs]

	central_cluster_bounds = []
	central_cluster_bounds.append([fast_lower_central_bound, fast_upper_central_bound])

	central_cluster_indexes = []
	central_cluster_indexes.append(central_fast_task_idxs_original_set)

	for slow_cluster_idx, slow_cluster in enumerate(slow_clusters):
		
		slow_durations = responses[slow_cluster]
		mean_slow_task_idx = (np.abs(slow_durations - np.mean(slow_durations))).argmin()
		mean_slow_task_idx_original_set = slow_cluster[mean_slow_task_idx]
		lower_central_bound = np.mean(slow_durations) - constraint_coefficient * np.std(slow_durations)
		upper_central_bound = np.mean(slow_durations) + constraint_coefficient * np.std(slow_durations)
		central_slow_task_idxs = np.where(np.logical_and(slow_durations>=lower_central_bound, slow_durations<=upper_central_bound))[0]
		central_slow_task_idxs_original_set = slow_cluster[central_slow_task_idxs]

		slow_durations_within_constraint = slow_durations[central_slow_task_idxs].flatten()
		slow_distances_from_mean = np.absolute(slow_durations_within_constraint - np.mean(slow_durations))
		slow_distances_sorted_indexes = np.argsort(slow_distances_from_mean)
		
		# now filter the central indexes to only those that are within the bounds and are the closest to mean
		central_slow_task_idxs = central_slow_task_idxs[slow_distances_sorted_indexes[:50]]
		central_slow_task_idxs_original_set = slow_cluster[central_slow_task_idxs]
	
		# update the bounds
		lower_central_bound = np.amin(slow_durations[central_slow_task_idxs])
		upper_central_bound = np.amax(slow_durations[central_slow_task_idxs])
		
		individual_comparisons.append([mean_fast_task_idx_original_set, mean_slow_task_idx_original_set])
		central_cluster_bounds.append([lower_central_bound, upper_central_bound])
		central_cluster_indexes.append(central_slow_task_idxs_original_set)
		cluster_comparisons.append([0,slow_cluster_idx+1])

	return central_cluster_indexes, central_cluster_bounds, individual_comparisons, cluster_comparisons, kde

def run_pca_transformation(taskset, pca):

	taskset = list(taskset)
	taskset[1] = np.array([sum((taskset[1][i]*np.array(pca.components_[:])).transpose()) for i in range(len(taskset[1]))])
	taskset = tuple(taskset)

def run_sample_weighting_and_pca_transformation(
		training_taskset,
		validation_taskset,
		testing_taskset,
		should_standardise,
		pca_transform,
		training_event_stats):

	training_responses = training_taskset[2]
	validation_responses = validation_taskset[2]
	testing_responses = testing_taskset[2]

	kde_params = {}
	kde_params["normalised"] = should_standardise
	kde_params["normalisation_statistic"] = training_event_stats[-1][1]

	kde_params["values"] = training_responses
	kde, _, _ = run_kernel_density_estimation(**kde_params)

	kde_params["kde"] = kde
	training_sample_weights = compute_sample_weights_from_kde(**kde_params)

	kde_params["values"] = validation_responses
	validation_sample_weights = compute_sample_weights_from_kde(**kde_params)

	kde_params["values"] = testing_responses
	testing_sample_weights = compute_sample_weights_from_kde(**kde_params)

	num_input_events = len(training_taskset[1][0])

	if pca_transform == True:
		from sklearn.decomposition import PCA
		pca = PCA(n_components=num_input_events)
		pca.fit_transform(training_taskset[1])
	else:
		pca = pca_struct()
		pca.components_ = np.identity(num_input_events)

	run_pca_transformation(training_taskset, pca)
	run_pca_transformation(validation_taskset, pca)
	run_pca_transformation(testing_taskset, pca)

	return training_taskset, validation_taskset, testing_taskset, training_sample_weights, validation_sample_weights, testing_sample_weights, pca

def calculate_mae(models, testing_taskset):

	testing_examples = testing_taskset[1]
	testing_responses_true = testing_taskset[2][:,0]

	maes = []
	for model in models:	
		testing_responses_pred = predict_from_model(model.model_file, testing_examples)
		mae = np.sum(np.absolute(testing_responses_pred - testing_responses_true)) / float(len(testing_responses_true))
		maes.append(mae)

	mean_mae = np.mean(maes)
	return mean_mae

def get_analysis_tasksets_for_configuration(
		dataset,
		all_events,
		pca_component_struct,
		taskset_kwargs
		):
	
	# Which configuration to get from the dataset is within the "benchmark_id" kwarg 

	taskset, _, _ = dataset.get_taskset(**taskset_kwargs)
	run_pca_transformation(taskset, pca_component_struct)

	taskset_fs, _, _ = dataset.get_taskset(**taskset_kwargs) # No pca transformation here

	taskset_kwargs["input_events"] = all_events
	taskset_kwargs["standardised"] = False
	
	taskset_all_events_fs_destandardised, _, _ = dataset.get_taskset(**taskset_kwargs) # No pca transformation here

	return taskset, taskset_fs, taskset_all_events_fs_destandardised

def analyse_tasks_across_benchmarks(self,
		models,
		testing_tasksets, # array of length two, for two sets of tasks to compare to eachother
		scores,
		predicted_responses,
		std_devs,
		reference_responses,
		true_responses,
		reference_benchmark_idx, # i.e. what index of testing_tasksets is the reference
		reference_strategy="minimum_predicted_response",
		reference_indexes=[],
		just_get_values=False # If True, then deeplift won't be invoked at all, and all produced contributions/coefficients will be 0
		): 

	target_benchmark_idx = [benchmark_idx for benchmark_idx in range(len(testing_tasksets)) if benchmark_idx != reference_benchmark_idx][0]

	if just_get_values:
		models = [0] # We don't need the models if we just want to get the value variations, but we do need an element in the list
	else:
		backend.set_floatx('float32')
		backend.set_learning_phase(False) # this is some keras funkery when using models with backend: https://github.com/fchollet/keras/issues/2310
		config = tf.ConfigProto()
		config.gpu_options.allow_growth = True
		backend.set_session(tf.Session(config=config))

	reference_example_idx = -1

	for model in models:

		if just_get_values:

			reference_example = np.zeros(len(testing_tasksets[reference_benchmark_idx][1][0])).tolist()
			reference_response = 0.0
			deeplift_results = np.array([[0.0 for event in reference_example] for task in testing_tasksets[target_benchmark_idx][1]])
			
			model_reference_predictions = np.array([0.0 for task in testing_tasksets[target_benchmark_idx][1]])
			model_target_predictions = np.array([0.0 for task in testing_tasksets[target_benchmark_idx][1]])

			if reference_responses == None:
				reference_responses = [reference_response]
			else:
				reference_responses.append(reference_response)

			reference_example_idx = 0
			reference_indexes.append(reference_example_idx)

		else:
			
			model_reference_predictions = predict_from_model(model.model_file, testing_tasksets[reference_benchmark_idx][1])
			model_target_predictions = predict_from_model(model.model_file, testing_tasksets[target_benchmark_idx][1])

			# If the comparison is between two groups of examples, which ones do we compare?
			# After this switch, we will have one reference example and one target example
			if reference_strategy == "minimum_predicted_response":
				reference_example_idx = np.argmin(model_reference_predictions)
				reference_example = testing_tasksets[reference_benchmark_idx][1][reference_example_idx]
				reference_response = model_reference_predictions[reference_example_idx]

			elif reference_strategy == "closest_to_mean_predicted_response":
				difference_from_mean = 9999999
				reference_example_idx = 0
				mean = np.mean(model_reference_predictions)
				for task_idx, duration in enumerate(model_reference_predictions):
					if abs(mean - duration) < difference_from_mean:
						difference_from_mean = abs(mean - duration)
						reference_example_idx = task_idx
				
				reference_example = testing_tasksets[reference_benchmark_idx][1][reference_example_idx]
				reference_response = model_reference_predictions[reference_example_idx]

			elif reference_strategy == "minimum_true_response":
				reference_example_idx = np.argmin(testing_tasksets[reference_benchmark_idx][2])
				reference_example = testing_tasksets[reference_benchmark_idx][1][reference_example_idx]
				reference_response = keras_model.predict(np.array([reference_example]))[:,0]

			elif reference_strategy == "minimum_example_inputs":
				reference_example = [np.amin(event_counts) for event_idx, event_counts in enumerate(testing_tasksets[reference_benchmark_idx][1].transpose())]
				reference_response = keras_model.predict(np.array([reference_example]))[:,0]

			elif reference_strategy == "zeros":
				reference_example = np.zeros(len(testing_tasksets[reference_benchmark_idx][1][0])).tolist()
				reference_response = keras_model.predict(np.array([reference_example]))[:,0]

			else:
				logging.error("Invalid reference strategy '%s'", reference_strategy)
				raise ValueError()

			if reference_responses == None:
				reference_responses = [reference_response]
			else:
				reference_responses.append(reference_response)

			del model_reference_predictions

			reference_indexes.append(reference_example_idx)
	
			target_example = testing_tasksets[target_benchmark_idx][1][0]

			# Now apply deeplift
			deeplift_results = run_deeplift_comparison(model.model_file,
				target_example,
				reference_example)

		# deeplift results is of format [task_idx][event_idx] = score
		# Aim to return a scores format of: [label_rank][input_event_idx] = list of scores, one per model

		if scores is None:
			# a list of scores for every task label and event
			scores = [[[] for event_idx in range(len(reference_example))] for i in list(range(int(np.amax(testing_tasksets[target_benchmark_idx][3]))))]
			# a list of predicted responses for every task label
			predicted_responses = [[] for i in list(range(int(np.amax(testing_tasksets[target_benchmark_idx][3]))))] 
			# a list of input values for every task label and event
			std_devs = [[[] for event_idx in range(len(reference_example))] for i in list(range(int(np.amax(testing_tasksets[target_benchmark_idx][3]))))]
		
		add_true_responses = False
		if true_responses is None:
			add_true_responses = True
			true_responses = [[] for i in list(range(int(np.amax(testing_tasksets[target_benchmark_idx][3]))))] 

		for task_idx, rank in enumerate(testing_tasksets[target_benchmark_idx][3]): # the true number of analysed tasks (we have scores for all)

			label_rank = int(rank)
			if label_rank >= len(scores):
				difference = (len(scores) - label_rank) + 1
				for i in range(difference + 1):
					scores.append([[] for event_idx in range(len(reference_example))])
					predicted_responses.append([])
					std_devs.append([[] for event_idx in range(len(reference_example))])
					if add_true_responses:	
						true_responses.append([])
			
			for event_idx, score in enumerate(deeplift_results[task_idx]):
				scores[label_rank][event_idx].append(score)
				std_devs[label_rank][event_idx].append(testing_tasksets[target_benchmark_idx][1][task_idx][event_idx])

			predicted_responses[label_rank].append(model_target_predictions[task_idx])

			if add_true_responses:	
				true_responses[label_rank].append(testing_tasksets[target_benchmark_idx][2][task_idx])

	return scores, predicted_responses, std_devs, reference_responses, true_responses

def get_delta_values(
		tasksets,
		reference_benchmark_idx,
		reference_index): 
		
	target_benchmark_idx = [benchmark_idx for benchmark_idx in range(len(tasksets)) if benchmark_idx != reference_benchmark_idx][0]

	reference_example = tasksets[reference_benchmark_idx][1][reference_index]

	values = [[[] for event_idx in range(len(reference_example))] for i in list(range(int(np.amax(tasksets[target_benchmark_idx][3]))))]
		
	for task_idx, rank in enumerate(tasksets[target_benchmark_idx][3]):

		label_rank = int(rank)
		if label_rank >= len(values):
			difference = (len(values) - label_rank) + 1
			for i in range(difference + 1):
				values.append([[] for event_idx in range(len(reference_example))])
		
		for event_idx in range(len(tasksets[target_benchmark_idx][1][task_idx])):
			values[label_rank][event_idx].append(tasksets[target_benchmark_idx][1][task_idx][event_idx] - reference_example[event_idx])

	return values

def performance_comparison(
		reference_taskset,
		reference_taskset_fs,
		reference_taskset_all_events_fs_destandardised,
		target_taskset,
		target_taskset_fs,
		target_taskset_all_events_fs_destandardised,
		models,
		training_event_stats,
		training_pca_matrix,
		linreg_coefficients_fs_des
		):

	# The performance comparison may operate as a 1-1 comparison, or a many-many comparison
	# For the latter case, we average across multiple individual comparisons (all-all)

	linreg_contributions_across_comparisons = []
	contributions_across_comparisons = []
	coefficients_across_comparisons = []
	event_variations_across_comparisons = []
	all_event_variations_across_comparisons = []
	pred_duration_variations_across_comparisons = []
	true_duration_variations_across_comparisons = []

	logging.debug("For the performance comparison, there are %d reference examples and %d target examples.", len(reference_taskset[1]))

	reference_instance_indexes_to_use_for_comparison = list(range(len(reference_taskset[1])))[:]
	
	# At the moment, all reference instances are compared to all target instances
	"""
	num_reference_instances_to_use = 4 # TODO parameterise this arbitrary number!
	if len(reference_instance_indexes_to_use_for_comparison) > num_reference_instances_to_use:
		selection = np.random.choice(np.arange(len(reference_instance_indexes_to_use_for_comparison)),num_reference_instances_to_use,replace=False)
		reference_instance_indexes_to_use_for_comparison = np.array(reference_instance_indexes_to_use_for_comparison)[selection].tolist()
	"""

	# For each reference example, compare it with the corresponding target example
	for reference_instance_idx in reference_instance_indexes_to_use_for_comparison:

		# Build a pseudo taskset containing the one reference example that we are comparing
		reference_taskset_temp = (reference_taskset[0], reference_taskset[1][[reference_instance_idx]],reference_taskset[2][[reference_instance_idx]],reference_taskset[3][[reference_instance_idx]])
	
		# Build a pseudo taskset containing the one target example that we are comparing
		target_instance_idx = np.where(target_taskset[3] == reference_taskset_temp[3][0])[0] # i.e. find the matching label
		target_taskset_temp = (target_taskset[0], target_taskset[1][[target_instance_idx]],target_taskset[2][[target_instance_idx]],target_taskset[3][[target_instance_idx]])
		
		# reference and target tasksets in FS
		reference_instance_idx_fs = np.where(reference_taskset_fs[3] == reference_taskset_temp[3][0])[0]
		target_instance_idx_fs = np.where(target_taskset_fs[3] == reference_taskset_temp[3][0])[0]
		reference_instance_idx_fs_all_events = np.where(reference_taskset_all_events_fs_destandardised[3] == reference_taskset_temp[3][0])[0]
		target_instance_idx_fs_all_events = np.where(target_taskset_all_events_fs_destandardised[3] == reference_taskset_temp[3][0])[0]
		target_taskset_fs_temp = (target_taskset_fs[0], target_taskset_fs[1][[target_instance_idx_fs]],target_taskset_fs[2][[target_instance_idx_fs]],target_taskset_fs[3][[target_instance_idx_fs]])
		reference_taskset_fs_temp = (reference_taskset_fs[0], reference_taskset_fs[1][[reference_instance_idx_fs]],reference_taskset_fs[2][[reference_instance_idx_fs]],reference_taskset_fs[3][[reference_instance_idx_fs]])

		comparison_tasksets = [reference_taskset_temp,target_taskset_temp]
		comparison_tasksets_fs = [reference_taskset_fs_temp,target_taskset_fs_temp]

		reference_indexes = []
		task_scores = None
		task_std_devs = None
		predicted_durations = None
		true_durations = None
		reference_response = None

		# TODO this function (and thus the surrounding code) is in need of significant refactoring!
		scores, predicted_durations_tmp, task_std_devs_tmp, reference_responses_tmp, true_durations_tmp = analyse_tasks_across_benchmarks(models,
			comparison_tasksets,
			task_scores,
			predicted_durations,
			task_std_devs,
			reference_response,
			true_durations,
			reference_benchmark_idx=0,
			reference_strategy="closest_to_mean_predicted_response",
			reference_indexes=reference_indexes)
		
		event_variations_fs = get_delta_values(comparison_tasksets_fs,
			reference_benchmark_idx=0,
			reference_index=reference_indexes[0])

		reference_example = reference_taskset_temp[1][reference_indexes[0]]
		reference_duration_true = reference_taskset_temp[2][reference_indexes[0]]
		reference_duration_pred = reference_responses_tmp[0]

		reference_example_all_events = reference_taskset_all_events_fs_destandardised[1][reference_instance_idx_fs_all_events]
		target_example_all_events = target_taskset_all_events_fs_destandardised[1][target_instance_idx_fs_all_events]

		duration_variation_true = np.array([np.mean(the_durations) for task_idx, the_durations in enumerate(true_durations_tmp) if len(the_durations) > 0]) - reference_duration_true
		duration_variation_destandardised_true = (duration_variation_true * training_event_stats[-1][1]) 
		
		duration_variation_pred = np.array([np.mean(the_durations) for task_idx, the_durations in enumerate(predicted_durations_tmp) if len(the_durations) > 0]) - reference_duration_pred
		duration_variation_destandardised_pred = (duration_variation_pred * training_event_stats[-1][1]) 

		event_variations_fs = np.array([np.mean(delta_values,axis=1) for task_idx, delta_values in enumerate(event_variations_fs) if any(len(s) > 0 for s in delta_values)])
		event_variations_fs_destandardized = event_variations_fs * np.array(training_event_stats).transpose()[1].transpose()[:-1]
		event_variations = np.array([np.mean(the_inputs,axis=1) for task_idx, the_inputs in enumerate(task_std_devs_tmp) if any(len(s) > 0 for s in the_inputs)]) - reference_example

		all_event_variations = np.array([target_value - reference_value for reference_value,target_value in zip(reference_example_all_events,target_example_all_events)])
		
		scores_pc = np.array([np.mean(the_scores,axis=1) for task_idx, the_scores in enumerate(scores) if any(len(s) > 0 for s in the_scores)])
		coefficients_pc = np.divide(scores_pc, event_variations, out=np.zeros_like(scores_pc), where=event_variations!=0)
		coefficients_fs = np.matmul(coefficients_pc, training_pca_matrix.components_)
		coefficients_fs_destandardized = ((coefficients_fs * np.array(training_event_stats)[-1][1])) / np.array(training_event_stats).transpose()[1].transpose()[:-1]
		scores_fs = coefficients_fs * event_variations_fs 
		scores_fs_destandardized = scores_fs * training_event_stats[-1][1]

		# Get the average over the compared examples (though there should only be one comparison here)
		average_scores = scores_fs_destandardized[0]
		average_coefficients = coefficients_fs_destandardized[0]
		average_values = event_variations_fs_destandardized[0]
		
		linreg_contributions = np.array(average_values) * linreg_coefficients_fs_des
		linreg_contributions_across_comparisons.append(linreg_contributions)
		
		if configuration_idx == reference_benchmark_id:
			# Record the actual reference values so we can later reconstruct absolute counts from deltas if we want
			average_values = np.matmul(reference_example, training_pca_matrix.components_) * np.array(training_event_stats).transpose()[1].transpose()[:-1] + np.array(training_event_stats).transpose()[0][:-1]
			predicted_durations_destandardised = reference_duration_pred * np.array(training_event_stats[-1][1]) + training_event_stats[-1][0]
			duration_variation_destandardised_pred = predicted_durations_destandardised
			durations_destandardised = reference_duration_true * np.array(training_event_stats[-1][1]) + training_event_stats[-1][0]
			duration_variation_destandardised_true = durations_destandardised

		all_event_variations_across_comparisons.append(all_event_variations)
		contributions_across_comparisons.append(average_scores)
		coefficients_across_comparisons.append(average_coefficients)
		event_variations_across_comparisons.append(average_values)
		pred_duration_variations_across_comparisons.append(np.mean(duration_variation_destandardised_pred))
		true_duration_variations_across_comparisons.append(np.mean(duration_variation_destandardised_true))

	# And I need to get value variations for *all* events
	average_all_event_variations = np.mean(np.array(all_event_variations_across_comparisons),axis=0)[0]

	# Now I have the computed data for all the comparisons between the reference set and target set
	average_linreg_contributions = np.mean(np.array(linreg_contributions_across_comparisons),axis=1)

	average_contributions = np.mean(np.array(contributions_across_comparisons),axis=0)
	average_coefficients = np.mean(np.array(coefficients_across_comparisons),axis=0)
	average_event_variations = np.mean(np.array(event_variations_across_comparisons),axis=0)
	average_predicted_duration_variations = np.mean(np.array(pred_duration_variations_across_comparisons),axis=0)
	average_true_duration_variations = np.mean(np.array(true_duration_variations_across_comparisons),axis=0)

	return average_contributions, average_linreg_contributions, average_coefficients, average_event_variations, average_all_event_variations, np.mean(average_predicted_duration_variations), np.mean(average_true_duration_variations)
	

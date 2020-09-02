import logging
import numpy as np

from sklearn.neighbors import KernelDensity
from scipy.signal import argrelextrema

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

	training_taskset = list(training_taskset)
	training_taskset[1] = np.array([sum((training_taskset[1][i]*np.array(pca.components_[:])).transpose()) for i in range(len(training_taskset[1]))])
	training_taskset = tuple(training_taskset)

	validation_taskset = list(validation_taskset)
	validation_taskset[1] = np.array([sum((validation_taskset[1][i]*np.array(pca.components_[:])).transpose()) for i in range(len(validation_taskset[1]))])
	validation_taskset = tuple(validation_taskset)

	testing_taskset = list(testing_taskset)
	testing_taskset[1] = np.array([sum((testing_taskset[1][i]*np.array(pca.components_[:])).transpose()) for i in range(len(testing_taskset[1]))])
	testing_taskset = tuple(testing_taskset)

	return training_taskset, validation_taskset, testing_taskset, training_sample_weights, validation_sample_weights, testing_sample_weights, pca


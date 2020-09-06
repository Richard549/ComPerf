import os
import logging

def write_event_stats_to_file(results_folder_full,
		experiment_repeat_index,
		k_fold_idx,
		all_events,
		all_event_stats):

	filename = results_folder_full + "/event_stats.txt"

	if os.path.isfile(filename) == False:
		with open(filename,'w') as f:
			f.write("experiment,partition," + ",".join([str(event) + "_mean:" + str(event) + "_std" for event in all_events]) + "\n")
			f.write(str(experiment_repeat_index) + "," + str(k_fold_idx) + "," + ",".join([str(s[0]) + ":" + str(s[1])  for s in all_event_stats]) + "\n")
	else:
		with open(filename,'a') as f:
			f.write(str(experiment_repeat_index) + "," + str(k_fold_idx) + "," + ",".join([str(s[0]) + ":" + str(s[1])  for s in all_event_stats]) + "\n")

def write_model_performance_to_file(results_folder_full,
		experiment_repeat_index,
		k_fold_idx,
		ann_rmse_des,
		lin_rmse_des,
		ann_mae_des,
		lin_mae_des):

	filename = results_folder_full + "/model_performance.txt"

	if os.path.isfile(filename) == False:
		with open(filename,'w') as f:
			f.write("experiment,partition,ann_rmse,linear_regression_rmse,ann_mae,linear_regression_mae\n")
			f.write(str(experiment_repeat_index) + "," + str(k_fold_idx) + "," + str(ann_rmse_des) + "," + str(lin_rmse_des) + "," + str(ann_mae_des) + "," + str(lin_mae_des) + "\n")
	else:
		with open(filename,'a') as f:
			f.write(str(experiment_repeat_index) + "," + str(k_fold_idx) + "," + str(ann_rmse_des) + "," + str(lin_rmse_des) + "," + str(ann_mae_des) + "," + str(lin_mae_des) + "\n")

def write_linreg_coefficients_to_file(results_folder_full,
		experiment_repeat_index,
		k_fold_idx,
		model_input_events,
		lin_coeff_fs_des):

	filename = results_folder_full + "/linreg_coefficients.txt"

	if os.path.isfile(filename) == False:
		with open(filename,'w') as f:
			f.write("experiment,partition," + ",".join([event for event in model_input_events]) + "\n")
			f.write(str(experiment_repeat_index) + "," + str(k_fold_idx) + "," + ",".join([str(s) for s in lin_coeff_fs_des]) + "\n")
	else:
		with open(filename,'a') as f:
			f.write(str(experiment_repeat_index) + "," + str(k_fold_idx) + "," + ",".join([str(s) for s in lin_coeff_fs_des]) + "\n")

def write_performance_comparison_results_to_file(
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
		average_true_duration_variation):

	coefficients_results_filename = results_folder_full + "/computed_coefficients.txt"
	contributions_results_filename = results_folder_full + "/computed_contributions.txt"
	linreg_contributions_results_filename = results_folder_full + "/computed_linreg_contributions.txt"
	values_results_filename = results_folder_full + "/computed_values.txt"
	all_values_results_filename = results_folder_full + "/computed_all_event_values.txt"

	# Write headers if we need to
	for filename in [coefficients_results_filename, contributions_results_filename,values_results_filename,linreg_contributions_results_filename]:
		additional_header = ""
		if "values" in filename:
			additional_header = "predicted_duration,true_duration,"
		if os.path.isfile(filename) == False:
			with open(filename,'a') as f:
				f.write("experiment,configuration,partition," + additional_header + ",".join([str(event) for event in input_events]) + "\n")

	if os.path.isfile(all_values_results_filename) == False:
		with open(all_values_results_filename,'a') as f:
			f.write("experiment,configuration,partition," + ",".join([str(event) for event in all_events]) + "\n")

	# Now write the comparison results
	with open(linreg_contributions_results_filename,'a') as f:
		f.write(str(experiment_repeat_index) + "," + configuration_identifier + "," + str(k_fold_idx) + "," + ",".join([str(s) for s in average_linreg_contributions_per_event]) + "\n")
	with open(all_values_results_filename,'a') as f:
		f.write(str(experiment_repeat_index) + "," + configuration_identifier + "," + str(k_fold_idx) + "," + ",".join([str(s) for s in average_all_event_value_variations_per_event]) + "\n")

	with open(coefficients_results_filename,'a') as f:
		f.write(str(experiment_repeat_index) + "," + configuration_identifier + "," + str(k_fold_idx) + "," + ",".join([str(s) for s in average_coefficients_per_event]) + "\n")
	with open(contributions_results_filename,'a') as f:
		f.write(str(experiment_repeat_index) + "," + configuration_identifier + "," + str(k_fold_idx) + "," + ",".join([str(s) for s in average_contributions_per_event]) + "\n")
	with open(values_results_filename,'a') as f:
		f.write(str(experiment_repeat_index) + "," + configuration_identifier + "," + str(k_fold_idx) + "," + str(average_pred_duration_variation) + "," + str(average_true_duration_variation) + "," + ",".join([str(s) for s in average_input_event_value_variations_per_event]) + "\n")

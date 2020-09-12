class Config:
	
	datasets_prefix="/home/rneill/workspace/data/matmul_datasets/matmul_tiled_"
	datasets_suffix=None
	filename_suffix=".csv"

	# these are only the events that we want to use as input to the model
	input_events = [
		"CPU",
		"Symbol",
		"Label", # converted to integer rank
		"OFFCORE_RESPONSE_0:DMND_DATA_RD:LLC_MISS_REMOTE",
		"PAPI_BR_CN",
		"PAPI_BR_INS",
		"PAPI_BR_MSP",
		"PAPI_BR_NTK",
		"PAPI_FDV_INS",
		"PAPI_L1_DCM",
		"PAPI_L1_ICM",
		"PAPI_L1_LDM",
		"PAPI_L1_STM",
		"PAPI_L2_DCA",
		"PAPI_L2_DCR",
		"PAPI_L2_DCW",
		"PAPI_L2_ICA",
		"PAPI_L2_ICH",
		"PAPI_L2_ICM",
		"PAPI_L2_ICR",
		"PAPI_L2_STM",
		"PAPI_L2_TCM",
		"PAPI_L2_TCW",
		"PAPI_L3_DCR",
		"PAPI_L3_DCW",
		"PAPI_L3_ICA",
		"PAPI_L3_ICR",
		"PAPI_L3_TCA",
		"PAPI_L3_TCM",
		"PAPI_L3_TCW",
		"PAPI_LD_INS",
		"PAPI_SR_INS",
		"PAPI_TLB_IM",
		"PAPI_TOT_INS",
		"data_read_0_hops",
		"data_read_1_hops",
		"data_write_0_hops",
		"realised_parallelism",
		"serialized_subtasks"
	]

	rusage_events = [
		"inv_context_switches",
		"max_resident_size",
		"minor_page_faults",
		"major_page_faults",
	]

	syscall_events = [
		"syscall_9",
		"syscall_2",
		"syscall_3",
		"syscall_12",
		"syscall_41",
		"syscall_59",
		"syscall_11",
		"syscall_57"
	]

	runtime_events = [
		"wq_length",
		"wq_steals",
		"wq_pushes",
		"num_tcreate",
		"num_texec",
		"slab_refills",
		"reuse_addr",
		"reuse_copy",
	]

	state_events = [
		# OpenMP
		"cycles_barrier",
		"cycles_critical",
		"cycles_single",
		"cycles_taskwait",
		"cycles_master",
		"cycles_loop",
		"cycles_task",
		"cycles_sequential_execution",
		"cycles_parallel_region",
		# OpenStream
		"cycles_seeking",
		"cycles_taskexec",
		"cycles_tcreate",
		"cycles_resdep",
		"cycles_tdec",
		"cycles_bcast",
		"cycles_init",
		"cycles_estimate_costs",
		"cycles_reorder",
		"cycles_gpuexec",
		"cycles_gputransfer"
	]

	time_events = [
		"DURATION",
		"PAPI_TOT_CYC",
		"PAPI_REF_CYC",
		"PAPI_STL_ICY",
		"PERF_COUNT_HW_STALLED_CYCLES_BACKEND",
		"PERF_COUNT_HW_STALLED_CYCLES_FRONTEND",
		"kernelmode_us",
		"usermode_us"
	]

	#response_event="PAPI_TOT_CYC"
	response_event="DURATION"

	target = "work" # "work", "work_and_nonwork", or "nonwork"
	max_num_datasets=30
	dataset_delimiter=","

	should_preload = True
	should_sum = False
	should_standardise = True
	pca_transformation = True
	exclusive_cache_events = True
	regularisation = "l1" # "l1", "l2", or "l1l2"

	num_profiles_per_experiment = 6 # N in paper
	num_modeling_indexes = 5 # N-1
	num_validation_indexes = 1 # S in paper
	num_experiments = 10

	symbols=[
		"all"
	] # can be "largest","all",or a list of particular wfn strings

	pickle_file="dataset.pickle"
	models_folder="models"
	results_folder="results"
	hyperparams_filename="hyperparams.json"

	# Define the configurations that we want to study in this experiment
	@staticmethod
	def get_configurations(self):
		configurations = []
		ivals = [8]
		jvals = [8,16,24,32,48,64,96,128,192,256,512,768,1024,1280,1536,1792,2048]
		kvals = [8,16,24,32,48,64,96,128,192,256,512,768,1024,1280,1536,1792,2048]

		for i in ivals:
			for j in jvals:
				for k in kvals:
					tile_id = str(i) + "_" + str(j) + "_" + str(k)

					if i != 8 or (j > 8 and k > 8): # only looking at j and k traversals (not jk)
						continue

					configurations.append(tile_id)

		return configurations
	

import numpy as np
import sys, os, logging, pickle
from functools import cmp_to_key

from enum import Enum, auto

class Target(Enum):
	ALL = auto()
	WORK = auto()
	NONWORK = auto()

def label_compare(a,b):
	split_a_str = a.split("|")
	split_b_str = b.split("|")
	
	if(a == ""):
		return -1
	elif(b == ""):
		return 1
	
	split_a = list(map(int, map(float,split_a_str)))
	split_b = list(map(int, map(float,split_b_str)))
	for depth_idx, a_rank_at_depth in enumerate(split_a):
		if depth_idx >= len(split_b):
			# a is a child of b, therefore b is first
			return 1
		if a_rank_at_depth > split_b[depth_idx]:
			# a is a later sibling than b, therefore b is first
			return 1
		if a_rank_at_depth < split_b[depth_idx]:
			# a is before b
			return -1
		# else continue to next depth

	# at this point, if we haven't returned, then length of a is less than the length of b
	# and there are no differences in rank
	# therefore b is a child of a, therefore a is first
	return -1

class ComperfDataset:

	# Preloading means to parse all of the tasksets ahead of the actual load
	# This allows us to find (for example) the labels common to all tasksets and load only those from each
	def __init__(self,
			input_events,
			response_event,
			configurations,
			datasets_prefix,
			max_num_datasets,
			dataset_delimiter,
			filename_suffix,
			should_sum,
			target,
			pickle_filename,
			should_reload,
			should_preload):

		self.tasksets = [] # tasksets is of the format [benchmark_idx][trace_idx][task_idx][event_idx] == count
		self.events = []
		self.rank_to_label_str = [] # rank_to_label_str is a dict mapping integer rank to label string, one for each benchmark
		self.symbols = []
		self.benchmarks = []
		self.fastest_benchmark_idx = None
		self.fastest_benchmark_duration = None

		self.common_labels = set()
		self.common_events = set()
		self.all_events = set()

		self.fastest_mean_task_duration = None
		self.fastest_mean_benchmark = None

		loaded = False
		if not should_reload:
			loaded = self.load_from_pickle(pickle_filename)

		# If there was no pickle to load from, then do the load and save a new pickle
		if loaded == False:

			if should_preload:
				for idx, configuration_identifier in enumerate(configurations):
					self.preload(datasets_prefix + str(configuration_identifier),
						max_num_datasets,
						dataset_delimiter,
						None,
						filename_suffix=filename_suffix,
						summed=should_sum,
						target=target)

			# load
			for idx, configuration_identifier in enumerate(configurations):
				self.load(datasets_prefix + str(configuration_identifier),
					max_num_datasets,
					dataset_delimiter,
					None,
					filename_suffix=filename_suffix,
					summed=should_sum,
					target=target,
					should_preload=should_preload
					)

			# Not currently filtering what is saved in the pickle
			filter_to_events = None

			# These input events may be derived from multiple events that are loaded into self.events
			# Should be None to just use the raw events
			transformed_input_events = [event.lower() for event in input_events]
		
			self.transform_events_and_finalise_load(
				filter_to_events=filter_to_events,
				pickle_file=pickle_filename,
				transformed_input_events=transformed_input_events)

	def preload(self,
			filename_prefix,
			max_num_datasets,
			dataset_delimiter,
			dataset_suffix=None,
			filename_suffix=".csv",
			summed=False,
			target=Target.WORK
			):

		# This must be run in order to collect the common events, common labels, all events across the configurations
		# This function should be run for *every* configuration, before any load() is called

		if len(self.events) > 0:
			logging.error("Must not call any dataset.load(...) before calling all dataset.preload(...)")
			raise RuntimeError()

		preload_results = self.preload_label_intersection(
			filename_prefix,
			dataset_suffix,
			max_num_datasets,
			dataset_delimiter,
			filename_suffix,
			target)

		common_label_set = preload_results[0]
		common_event_set = preload_results[1]
		all_event_list = preload_results[3]

		for event in all_event_list:
			self.all_events.add(event)

		if len(self.common_labels) == 0:
			self.common_labels = common_label_set
		else:
			self.common_labels = self.common_labels.intersection(common_label_set)
		
		if len(self.common_events) == 0:
			self.common_events = common_event_set
		else:
			self.common_events = self.common_events.intersection(common_event_set)

	def preload_label_intersection(self,
			filename_prefix,
			dataset_suffix,
			max_num_datasets,
			dataset_delimiter,
			filename_suffix,
			target=Target.WORK):

		# get the common set of labels
		label_intersection = set()
		events_intersection = set() # incase a syscall happens in one that doesn't happen in another, for example
		events_total = set()
		
		file_idx = 0
		while True:

			if filename_prefix is None:
				filename = "/".join(dataset_suffix.split("/")[0:-1]) + "/" + str(file_idx) + dataset_suffix.split("/")[-1] + filename_suffix
			else:
				filename = filename_prefix + str(file_idx) + filename_suffix

			logging.debug("Loading " + filename)
			
			if file_idx >= max_num_datasets:
				logging.debug("Finished pre-loading " + str(max_num_datasets) + " tasksets")
				break
			
			if os.path.isfile(filename) == False:
				logging.debug("The file %s does not exist, so stopping the pre-load.", filename)
				break

			labels_in_file = set()
			header_in_file = set()

			with open(filename) as f:

				# find the header_line (in Fuse, for example, there is metadata before the task event counts)
				header_line = f.readline()
				header_line_idx = 0
				while header_line:
					if "papi" not in header_line.lower() and "x1" not in header_line.lower() and "label" not in header_line.lower(): # x1 for synthetic
						header_line = f.readline()
						header_line_idx += 1
					else:
						break

				header = [heading.lower().replace("\n","") for heading in header_line.split(dataset_delimiter)]

				# Adjust events with their synonyms
				if "DMND_DATA_RD".lower() in header: # assuming the event is OFFCORE_RESPONSE_0:DMND_DATA_RD:LLC_MISS_REMOTE
					numa_idx = header.index("DMND_DATA_RD".lower()) - 1
					header = header[:numa_idx] + ["OFFCORE_RESPONSE_0:DMND_DATA_RD:LLC_MISS_REMOTE".lower()] + header[numa_idx+3:]
				if "ANY_REQUEST".lower() in header: # assuming the event is OFFCORE_RESPONSE_0:ANY_REQUEST:LLC_MISS_REMOTE
					numa_idx = header.index("ANY_REQUEST".lower()) - 1
					header = header[:numa_idx] + ["OFFCORE_RESPONSE_0:ANY_REQUEST:LLC_MISS_REMOTE".lower()] + header[numa_idx+3:]

				line = f.readline()
				while line:
					
					if target == Target.NONWORK and "runtime" not in line and "non_work" not in line:
						line = f.readline()
						continue
					if target == Target.WORK and ("runtime" in line or "non_work" in line):
						line = f.readline()
						continue

					if "unknown" in line:
						logging.warning("Found a task with unknown values.")
						line = f.readline()
						continue
					
					split_line = line.split(dataset_delimiter)
					label = split_line[header.index("label")]
					labels_in_file.add(label)
					line = f.readline()

			for event in header:
				header_in_file.add(event)
				events_total.add(event)

			# now intersect with the running event set
			if len(label_intersection) == 0:
				label_intersection = labels_in_file
			else:
				label_intersection = label_intersection.intersection(labels_in_file)
			
			if len(events_intersection) == 0:
				events_intersection = header_in_file
			else:
				events_intersection = events_intersection.intersection(header_in_file)

			file_idx += 1
				
		return label_intersection, events_intersection, list(events_total)

	# TODO this needs refactored!
	def load(self,
			filename_prefix,
			max_num_datasets,
			dataset_delimiter,
			dataset_suffix=None,
			filename_suffix=".csv",
			summed=False,
			target=Target.WORK,
			should_preload=True
			):
		
		if should_preload == True:
			# Make sure we have done so
			if len(self.all_events) == 0:
				logging.fatal("Must not call any dataset.load(...) before calling all dataset.preload(...)")
				raise RuntimeError()

		else:
			# If we are not preloading across configurations, we still need to know common labels and events for this particular configuration
			# So in that case, we 'preload' one at a time as we load each one
			preload_results = self.preload_label_intersection(filename_prefix,
				dataset_suffix,
				max_num_datasets,
				dataset_delimiter,
				filename_suffix,
				target)
			self.common_labels = preload_results[0]
			self.common_events = preload_results[1]

		benchmark_id = None
		if filename_prefix not in self.benchmarks and dataset_suffix not in self.benchmarks:
			benchmark_id = len(self.benchmarks)
			self.tasksets.append([])
			self.rank_to_label_str.append({})
			if filename_prefix == None:
				self.benchmarks.append(dataset_suffix)
			else:
				self.benchmarks.append(filename_prefix)
		else:
			if filename_prefix in self.benchmarks:
				benchmark_id = self.benchmarks.index(filename_prefix)
			elif dataset_suffix in self.benchmarks:
				benchmark_id = self.benchmarks.index(dataset_suffix)
		
		if summed:
			self.events = list(self.all_events)
		else:
			self.events = list(self.common_events)
				
		derived_events = ["depth","benchmark","kernel_time"]
		derived_event_indexes = [len(self.events), len(self.events)+1, len(self.events)+2]

		# add some derived events
		if "depth" not in self.events:
			self.events.append("depth")
		if "benchmark" not in self.events:
			self.events.append("benchmark")
		if "kernel_time" not in self.events:
			self.events.append("kernel_time")

		file_idx = 0
		while True:

			if filename_prefix == None:
				print(dataset_suffix)
				filename = "/".join(dataset_suffix.split("/")[0:-1]) + "/" + str(file_idx) + dataset_suffix.split("/")[-1] + filename_suffix
			else:
				filename = filename_prefix + str(file_idx) + filename_suffix
			logging.info("Loading %s", filename)
			
			if file_idx >= max_num_datasets:
				logging.info("Finished loading %d tasksets.", max_num_datasets)
				break
			
			if os.path.isfile(filename) == False:
				logging.info("The file %s does not exist, so stopping the load.", filename)
				break
			
			with open(filename) as f:

				ordered_labels = []

				# find the header_line (in Fuse, for example, there is metadata before the task event counts)
				header_line = f.readline()
				header_line_idx = 0
				while header_line:
					if "papi" not in header_line.lower() and "x1" not in header_line.lower() and "label" not in header_line.lower(): # x1 for synthetic
						header_line = f.readline()
						header_line_idx += 1
					else:
						break

				# Adjust events with their synonyms
				header = [heading.lower().replace("\n","") for heading in header_line.split(dataset_delimiter)]
				if "DMND_DATA_RD".lower() in header: # assuming the event is OFFCORE_RESPONSE_0:DMND_DATA_RD:LLC_MISS_REMOTE
					numa_idx = header.index("DMND_DATA_RD".lower()) - 1
					header = header[:numa_idx] + ["OFFCORE_RESPONSE_0:DMND_DATA_RD:LLC_MISS_REMOTE".lower()] + header[numa_idx+3:]
				if "ANY_REQUEST".lower() in header: # assuming the event is OFFCORE_RESPONSE_0:ANY_REQUEST:LLC_MISS_REMOTE
					numa_idx = header.index("ANY_REQUEST".lower()) - 1
					header = header[:numa_idx] + ["OFFCORE_RESPONSE_0:ANY_REQUEST:LLC_MISS_REMOTE".lower()] + header[numa_idx+3:]

				# these are be the indexes of the taskset, for each event in self.events
				# it is possible for an event in self.events to not be in the header (if summed)
				# it is possible for there to be more events in the header than in self.events (if not summed)
				indexes = []
				for event in self.events:
					if event in header:
						# where in the taskset is this event?
						header_index = header.index(event)
						indexes.append(header_index)
					elif event in derived_events:
						pass
					else:
						# the taskset does not include this event
						# so add a placeholder 
						indexes.append(-1)

				if summed:
					ordered_labels = list(range(int(1e6))) # because some summed might have more total labels than common labels
				else:
					ordered_labels = self._parse_label_ranks(f,header.index("label"),dataset_delimiter,header_line_idx,self.common_labels,target=target)

				new_taskset = [[] for event in self.events]

				num_runtime_tasks = 0

				line = f.readline()
				task_idx=0
				while line:

					if target == Target.NONWORK and "runtime" not in line and "non_work" not in line:
						line = f.readline()
						continue
					if target == Target.WORK and ("runtime" in line or "non_work" in line):
						line = f.readline()
						continue
					
					if "unknown" in line:
						logging.warning("Found a task with unknown values.")
						line = f.readline()
						continue

					if target == Target.WORK or target == Target.ALL:
						if "runtime" in line or "non_work" in line:
							num_runtime_tasks += 1

					split_line = line.split(dataset_delimiter)
					if summed == False and split_line[header.index("label")] not in self.common_labels:
						logging.warning("Found a task not in the common set (this task has label " + str(split_line[header.index("label")]) + ".")
						line = f.readline()
						continue

					# header_idx is the correct index in the header that corresponds to self.events[idx]
					for idx, header_idx in enumerate(indexes):

						if header_idx == -1:
							# we don't have this event in the current taskset
							new_taskset[idx].append(0)
							continue

						if header[header_idx] == "label":
							new_taskset[idx].append(ordered_labels[task_idx])
							if file_idx == 0:
								self.rank_to_label_str[benchmark_id][ordered_labels[task_idx]] = split_line[header_idx]
							depth_of_task = len(split_line[header_idx].split("-"))
			
						elif header[header_idx] == "symbol":
							if split_line[header_idx] in self.symbols:
								new_taskset[idx].append(self.symbols.index(split_line[header_idx]))
							else:
								self.symbols.append(split_line[header_idx])
								new_taskset[idx].append(len(self.symbols)-1)
						else:
							new_taskset[idx].append(int(split_line[header_idx]))

						if header[header_idx] != self.events[idx]:
							logging.error("Logic error in loading a line of event values from %s", filename)
							logging.error("The line was: %s", line)
							raise RuntimeError()

						if self.events[idx] == "cpu":
							label_index = self.events.index("cpu")
							index_in_header = indexes[label_index]

					# after all the normal events, append the derived events in correct order
					new_taskset[derived_event_indexes[0]].append(depth_of_task)		
					new_taskset[derived_event_indexes[1]].append(benchmark_id)

					if "papi_tot_cyc" in self.events:
						# TODO is this accurate?
						kernel_time = new_taskset[self.events.index("duration")][task_idx] - new_taskset[self.events.index("papi_tot_cyc")][task_idx]
						new_taskset[derived_event_indexes[2]].append(kernel_time)
					else:
						new_taskset[derived_event_indexes[2]].append(0)

					line = f.readline()
					task_idx += 1

				logging.info("There were %d non-work instances.", num_runtime_tasks)
				# now fill any remaining runtime tasks
				if target == Target.NONWORK or target == Target.ALL:

					# TODO should make this more robust (preload should find how many CPUs are expected)
					# This is added because non-work instances are changing across configurations (e.g. thread counts), and we need the same numbers to build the matrix
					num_required_runtime_tasks = 16
					for runtime_task_idx in range(num_required_runtime_tasks - num_runtime_tasks):
						for idx, header_idx in enumerate(indexes):
							if header[header_idx] == "label":
								new_taskset[idx].append(ordered_labels[0])
							elif header[header_idx] == "symbol":
								new_taskset[idx].append(0)
							else:
								new_taskset[idx].append(0)
						new_taskset[derived_event_indexes[0]].append(0)
						new_taskset[derived_event_indexes[1]].append(benchmark_id)
						new_taskset[derived_event_indexes[2]].append(0)

				if summed:
					compare_duration = np.sum(np.array(new_taskset,dtype='float32')[self.events.index("duration")])
				else:
					# find closest to mean task
					compare_duration = np.mean(np.array(new_taskset,dtype='float32')[self.events.index("duration")])
					
					duration_of_mean = 0
					difference_from_mean = 999999999
					for task_idx, duration in enumerate(np.array(new_taskset,dtype='float32')[self.events.index("duration")]):
						if abs(compare_duration - duration) < difference_from_mean:
							difference_from_mean = abs(compare_duration - duration)
							duration_of_mean = duration

					if self.fastest_mean_task_duration == None or duration_of_mean < self.fastest_mean_task_duration:
						self.fastest_mean_benchmark = benchmark_id
						self.fastest_mean_task_duration = duration_of_mean

				if self.fastest_benchmark_duration == None or compare_duration < self.fastest_benchmark_duration:
					self.fastest_benchmark_duration = compare_duration
					self.fastest_benchmark_idx = benchmark_id

				new_taskset = np.array(new_taskset,dtype='float32')
				self.tasksets[benchmark_id].append(new_taskset.transpose())

				logging.debug("Loaded taskset with shape:" + str(new_taskset.shape))
				
				symbol_idx = self.events.index("symbol")
				unique_symbols, num_tasks_per_symbol = np.unique(new_taskset[symbol_idx],return_counts=True)

				for unique_symbol, num_tasks in zip(unique_symbols,num_tasks_per_symbol):
					logging.debug(str(self.symbols[int(unique_symbol)]) + ":" + str(num_tasks))

				file_idx += 1

		return

	def transform_events_and_finalise_load(self,
			filter_to_events=None,
			pickle_file=None,
			transformed_input_events=None):

		# Change the derived event values
		for benchmark_idx in range(len(self.benchmarks)):

			self.tasksets[benchmark_idx] = np.array(self.tasksets[benchmark_idx],dtype='float32')

			if transformed_input_events is not None:
				if "PAPI_BR_UCN".lower() in transformed_input_events and "papi_br_cn" in self.events and "papi_br_ins" in self.events:
					self.tasksets[benchmark_idx].transpose()[self.events.index("papi_br_ins")] -= self.tasksets[benchmark_idx].transpose()[self.events.index("papi_br_cn")]
					self.tasksets[benchmark_idx].transpose()[self.events.index("papi_br_ins")] = self.tasksets[benchmark_idx].transpose()[self.events.index("papi_br_ins")].clip(min=0.0)
				
				if "L2_DATA_PF_RD".lower() in transformed_input_events and "papi_l1_dcm" in self.events and "papi_l1_stm" in self.events and "papi_l1_ldm" in self.events:
					self.tasksets[benchmark_idx].transpose()[self.events.index("papi_l1_dcm")] -= (self.tasksets[benchmark_idx].transpose()[self.events.index("papi_l1_ldm")] + self.tasksets[benchmark_idx].transpose()[self.events.index("papi_l1_stm")])

				if "L2_INS_PF_RD".lower() in transformed_input_events and "papi_l2_ica" in self.events and "papi_l1_icm" in self.events:
					self.tasksets[benchmark_idx].transpose()[self.events.index("papi_l2_ica")] -= self.tasksets[benchmark_idx].transpose()[self.events.index("papi_l1_icm")]

				if "L3_DATA_PF_RD".lower() in transformed_input_events and "papi_l2_tcm" in self.events and "papi_l2_stm" in self.events and "papi_l3_dcr" in self.events and "papi_l2_icm" in self.events:
					self.tasksets[benchmark_idx].transpose()[self.events.index("papi_l2_tcm")] -= (self.tasksets[benchmark_idx].transpose()[self.events.index("papi_l2_stm")] + self.tasksets[benchmark_idx].transpose()[self.events.index("papi_l3_dcr")] + self.tasksets[benchmark_idx].transpose()[self.events.index("papi_l2_icm")])
					self.tasksets[benchmark_idx].transpose()[self.events.index("papi_l2_tcm")] = self.tasksets[benchmark_idx].transpose()[self.events.index("papi_l2_tcm")].clip(min=0.0)

				if "local_mem_hits".lower() in transformed_input_events and "papi_l3_tcm" in self.events and "OFFCORE_RESPONSE_0:DMND_DATA_RD:LLC_MISS_REMOTE".lower() in self.events:
					self.tasksets[benchmark_idx].transpose()[self.events.index("papi_l3_tcm")] -= self.tasksets[benchmark_idx].transpose()[self.events.index("OFFCORE_RESPONSE_0:DMND_DATA_RD:LLC_MISS_REMOTE".lower())]
					self.tasksets[benchmark_idx].transpose()[self.events.index("papi_l3_tcm")] = self.tasksets[benchmark_idx].transpose()[self.events.index("papi_l3_tcm")].clip(min=0.0)
	
			if filter_to_events != None:
				event_indexes = [i for i,event in enumerate(self.events) if event.lower() in list(map(str.lower,filter_to_events))]
				events_filtered = [event for i,event in enumerate(self.events) if event.lower() in list(map(str.lower,filter_to_events))]

				logging.debug("Filtering from " + str(len(self.events)) + " to " + str(len(filter_to_events)) + " gave " + str(len(event_indexes)) + " events") 
				self.tasksets[benchmark_idx] = self.tasksets[benchmark_idx].transpose()[event_indexes].transpose()

			logging.debug("Benchmark %s has shape %s.", str(benchmark_idx), str(self.tasksets[benchmark_idx].shape))
			
		# Now change the event names
		if transformed_input_events is not None:
			if "PAPI_BR_UCN".lower() in transformed_input_events and "papi_br_cn" in self.events and "papi_br_ins" in self.events:
				self.events[self.events.index("papi_br_ins")] = "papi_br_ucn"
			if "L2_DATA_PF_RD".lower() in transformed_input_events and "papi_l1_dcm" in self.events and "papi_l1_stm" in self.events and "papi_l1_ldm" in self.events:
				self.events[self.events.index("papi_l1_dcm")] = "l2_data_pf_rd"
			if "L2_INS_PF_RD".lower() in transformed_input_events and "papi_l2_ica" in self.events and "papi_l1_icm" in self.events:
				self.events[self.events.index("papi_l2_ica")] = "l2_ins_pf_rd"
			if "L3_DATA_PF_RD".lower() in transformed_input_events and "papi_l2_tcm" in self.events and "papi_l2_stm" in self.events and "papi_l3_dcr" in self.events and "papi_l2_icm" in self.events:
				self.events[self.events.index("papi_l2_tcm")] = "l3_data_pf_rd"
			if "local_mem_hits".lower() in transformed_input_events and "papi_l3_tcm" in self.events and "OFFCORE_RESPONSE_0:DMND_DATA_RD:LLC_MISS_REMOTE".lower() in self.events:
				self.events[self.events.index("papi_l3_tcm")] = "local_mem_hits"
			if "remote_mem_hits".lower() in transformed_input_events and "OFFCORE_RESPONSE_0:DMND_DATA_RD:LLC_MISS_REMOTE".lower() in self.events:
				self.events[self.events.index("OFFCORE_RESPONSE_0:DMND_DATA_RD:LLC_MISS_REMOTE".lower())] = "remote_mem_hits"

		# Always change this event's name
		if "remote_mem_hits".lower() in transformed_input_events and "OFFCORE_RESPONSE_0:DMND_DATA_RD:LLC_MISS_REMOTE".lower() in self.events:
			self.events[self.events.index("OFFCORE_RESPONSE_0:DMND_DATA_RD:LLC_MISS_REMOTE".lower())] = "remote_mem_hits"
		
		if filter_to_events is not None:
			self.events = [event.lower() for event in self.events if event.lower() in list(map(str.lower,filter_to_events))]

		if pickle_file is not None:
			data = []
			data.append(self.tasksets)
			data.append(self.events)
			data.append(self.rank_to_label_str)
			data.append(self.symbols)
			data.append(self.benchmarks)
			data.append(self.fastest_benchmark_idx)
			data.append(self.fastest_benchmark_duration)

			with open(pickle_file, 'wb') as f:
				pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)

		return
	
	def load_from_pickle(self, pickle_file):

		if os.path.isfile(pickle_file) == False:
			return False

		with open(pickle_file, 'rb') as f:
			data = pickle.load(f)
			self.tasksets = data[0]
			self.events = data[1]
			self.rank_to_label_str = data[2]
			self.symbols = data[3]
			self.benchmarks = data[4]
			self.fastest_benchmark_idx = data[5]
			self.fastest_benchmark_duration = data[6]

		logging.debug("Shape of loaded tasksets from pickle file: (%s, %s)", str(len(self.tasksets)), str(self.tasksets[0].shape))
		return True

	def get_num_tasksets(self):
		return self.tasksets[0].shape[0]
	
	def get_num_benchmarks(self):
		return len(self.benchmarks)

	def parse_label_ranks(self,
			file_handle,
			label_idx,
			delimiter,
			header_line_idx,
			common_label_set,
			target=Target.WORK):

		# iterate through the file to collect all of the labels for ranking
		# then reset the file_handle to original position

		ordered_labels = []

		line = file_handle.readline()
		while line:
			split_line = line.split(delimiter)
			if split_line[label_idx] not in common_label_set:
				line = file_handle.readline()
				continue

			if target == Target.NONWORK and "runtime" not in line and "non_work" not in line:
				line = f.readline()
				continue
			if target == Target.WORK and ("runtime" in line or "non_work" in line):
				line = f.readline()
				continue

			label = split_line[label_idx].replace("[","").replace("]","")
			ordered_labels.append(label)
			line = file_handle.readline()

		ordered_labels = list(enumerate(ordered_labels))

		key_func = cmp_to_key(lambda a, b: label_compare(a[1], b[1]))

		ordered_labels.sort(key=key_func)
		ordered_labels = [x[0] for x in ordered_labels]
		ordered_labels = np.array(ordered_labels)
		ordered_labels = np.argsort(ordered_labels)

		# ordered_labels[idx] now gives the rank of task idx's label

		# Now reset line position such that the first readline() will give the first task
		# Returns to the first byte:
		file_handle.seek(0)
		for skip_line in range(header_line_idx+1):
			line = file_handle.readline()

		return ordered_labels

	# Each set should be a triple of (list_taskset_indexes, standardised examples, responses, label_ranks)
	def get_taskset(self,
			input_events,
			output_event,
			set_indexes,
			specific_symbols,
			constant_events,
			is_training_set=False,
			training_event_stats=None,
			event_rates=False,
			standardised=True,
			normalised_to_ins=False,
			benchmark_id=-1,
			summed=False,
			get_mean_only=False,
			exclusive_cache_events=False):

		if training_event_stats == None:
			training_event_stats = []

		input_events = [event.lower() for event in input_events]
		logging.debug("Getting taskset for input_events: %s", str(list(enumerate(input_events))))
		logging.debug("From the dataset with self.events: %s", str(list(enumerate(self.events))))
		
		combined_taskset = np.array(self.tasksets) # this gives shape (num_confs, num_repeats, num_tasks, num_events)
		
		# Filter by symbols as given in the experiment config
		if specific_symbols[0].lower() == "all":
			# Experiment includes all symbols, so do nothing
			pass
		else:
			symbols_to_filter_to = []
			symbol_idx = self.events.index("symbol")

			if specific_symbols[0] == "largest":
				unique_symbols, num_tasks_per_symbol = np.unique(combined_taskset.transpose()[symbol_idx],return_counts=True)
				largest_symbol = unique_symbols[np.argmax(num_tasks_per_symbol)]
				symbols_to_filter_to.append(largest_symbol)
				logging.debug("Filtering tasks to largest symbol, index:" + str(largest_symbol))
			else:
				symbols_to_filter_to = [self.symbols.index(symbol_string) for symbol_string in specific_symbols]
				logging.debug("Filtering tasks to:" + str(specific_symbols))

			num_confs = combined_taskset.shape[0]
			num_repeats = combined_taskset.shape[1]
			num_events = combined_taskset.shape[3]
			combined_taskset = combined_taskset[np.isin(combined_taskset[:,:,:,symbol_idx], symbols_to_filter_to)]
			num_filtered_tasks = int(combined_taskset.size / (num_confs*num_repeats*num_events))
			combined_taskset = combined_taskset.reshape([num_confs,num_repeats,num_filtered_tasks,num_events])

		if benchmark_id == -1:

			if summed:
				combined_taskset = np.sum(combined_taskset, axis=2) # this gives shape (num_confs, num_repeats, num_events)
				combined_taskset = np.expand_dims(combined_taskset, axis=2) # this gives shape (num_confs, num_repeats, 1, num_events)
				# therefore we have collapsed all the tasks into 1
			
			combined_taskset = np.copy(np.concatenate(combined_taskset,axis=1))
			combined_taskset = np.concatenate(combined_taskset[set_indexes])

		else:
			
			combined_taskset = np.array(self.tasksets) # this gives shape (num_confs, num_repeats, num_tasks, num_events)
			if summed:
				combined_taskset = np.sum(combined_taskset, axis=2) # this gives shape (num_confs, num_repeats, num_events)
				combined_taskset = np.expand_dims(combined_taskset, axis=2) # this gives shape (num_confs, num_repeats, 1, num_events)

			combined_taskset = np.copy(np.concatenate(combined_taskset[benchmark_id][set_indexes]))

		logging.info("Loading taskset with shape: %s", str(combined_taskset.shape))

		predictor_indexes = []
		for input_event in input_events:

			# What is the index of this input_event in self.events?
			predictor_index = self.events.index(input_event.lower())

			if input_event.lower() == "label" and summed:
				constant_events.append(input_event.lower())
				continue
			
			if is_training_set == True:
				first_value = combined_taskset[0][predictor_index]
				if (np.any(combined_taskset.transpose()[predictor_index] != first_value)): # is it non-constant
					predictor_indexes.append(predictor_index)
				else:
					if input_event.lower() not in constant_events:
						constant_events.append(input_event.lower())
					logging.debug("No counts for event " + input_event + " so disregarding it from the modelling.")
			else:
				# If it's not a training set, then just include all requested events without checking
				if input_event not in constant_events:
					predictor_indexes.append(predictor_index)

		logging.info("Constant events are: %s", str(constant_events))
		input_events = [event.lower() for event in input_events if event.lower() not in constant_events]

		response_indexes = [self.events.index(output_event.lower())]
		combined_responses = combined_taskset.transpose()[response_indexes].transpose()

		combined_examples = combined_taskset.transpose()[predictor_indexes].transpose()

		# the combined_taskset is now in order of input_events
		# i.e. combined_taskset.transpose()[4] are the counts for input_events[4]

		if summed == True:
			combined_label_ranks = np.zeros(len(combined_examples))
		else:
			combined_label_ranks = combined_taskset.transpose()[self.events.index("label")].transpose()

		# If we just want to get the mean value
		if get_mean_only == True:

			difference_from_mean = 999999999
			mean_example_idx = 0
			mean = np.mean(combined_responses)
			for task_idx, duration in enumerate(combined_responses):
				if abs(mean - duration) < difference_from_mean:
					difference_from_mean = abs(mean - duration)
					mean_example_idx = task_idx
			
			# filter to only mean_example_idx
			combined_examples = np.array([combined_examples[mean_example_idx]])
			combined_responses = np.array([combined_responses[mean_example_idx]])
			combined_label_ranks = np.array([combined_label_ranks[mean_example_idx]])
		
		if exclusive_cache_events == True:

			combined_examples = combined_examples.transpose()

			if "papi_l1_tcm" in input_events and "papi_l2_tcm" in input_events and "papi_l3_tcm" in input_events:
				combined_examples[input_events.index("papi_l1_tcm")] = combined_examples[input_events.index("papi_l1_tcm")] - combined_examples[input_events.index("papi_l2_tcm")]
				combined_examples[input_events.index("papi_l2_tcm")] = combined_examples[input_events.index("papi_l2_tcm")] - combined_examples[input_events.index("papi_l3_tcm")]
			
			if "papi_l1_tcm" in input_events and "papi_l2_tcm" in input_events and "papi_l3_tcm" in input_events and "remote_mem_access" in input_events:
				combined_examples[input_events.index("papi_l3_tcm")] = combined_examples[input_events.index("papi_l3_tcm")] - combined_examples[input_events.index("remote_mem_access")]

			combined_examples = combined_examples.transpose()

		# save the training event stats, even if we don't standardise
		if is_training_set == True:
			combined_examples = combined_examples.transpose()
			for predictor_idx, predictor_event in enumerate(input_events):
				mean = np.mean(combined_examples[predictor_idx])
				std_dev = np.std(combined_examples[predictor_idx])
				training_event_stats.append((mean,std_dev))
			combined_examples = combined_examples.transpose()

			mean = np.mean(combined_responses)
			std_dev = np.std(combined_responses)
			training_event_stats.append((mean,std_dev))
		
		if standardised == True:
			combined_examples = combined_examples.transpose()

			# standardise
			logging.debug("Standardising the %d input events", len(input_events))
			for predictor_idx, predictor_event in enumerate(input_events):
				if is_training_set == True:
					mean = np.mean(combined_examples[predictor_idx])
					std_dev = np.std(combined_examples[predictor_idx])
				else:
					mean = training_event_stats[predictor_idx][0]
					std_dev = training_event_stats[predictor_idx][1]

				logging.debug("Event %s has mean %f and std %f.", predictor_event, mean, std_dev)

				combined_examples[predictor_idx] = ((combined_examples[predictor_idx] - mean)/std_dev) # standardise
				#combined_examples[predictor_idx] = ((combined_examples[predictor_idx])/std_dev) # only normalise
				#combined_examples[predictor_idx] = ((combined_examples[predictor_idx] - mean)) # only mean center

			combined_examples = combined_examples.transpose()

			if is_training_set == True:
				mean = np.mean(combined_responses)
				std_dev = np.std(combined_responses)
			else:
				mean = training_event_stats[-1][0]
				std_dev = training_event_stats[-1][1]

			combined_responses = ((combined_responses - mean)/std_dev) # standardise the response
			#combined_responses = (combined_responses/std_dev) # only normalize
			#combined_responses = (combined_responses - mean) # only mean center

import os, random, logging, json, math
import numpy as np
from enum import Enum, auto

import tensorflow as tf
import keras
from keras import backend
from keras.backend.tensorflow_backend import set_session
from keras import metrics, regularizers, initializers
from keras.layers import Input, Dense, BatchNormalization, Activation, Dropout, LeakyReLU
from keras.models import Model, Sequential, load_model
from keras.callbacks import ModelCheckpoint, CSVLogger, EarlyStopping
from keras.constraints import NonNeg, Constraint
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score
import deeplift
from deeplift.conversion import kerasapi_conversion as kc

from ComperfUtil import acquire_lock, release_lock, ensure_exists

class Regularisation(Enum):
	L1 = auto()
	L2 = auto()
	L1L2 = auto()

def convert_regularisation_str_to_enum(reg_str):
	if reg_str == "l1":
		return Regularisation.L1
	elif reg_str == "l2":
		return Regularisation.L2
	elif reg_str == "l1l2":
		return Regularisation.L1L2
	else:
		logging.error("Do not recognise regularisation: %s", reg_str)
		raise ValueError()

def convert_regularisation_enum_to_str(reg):
	if reg == Regularisation.L1:
		return "l1"
	elif reg == Regularisation.L2:
		return "l2"
	elif reg == Regularisation.L1L2:
		return "l1l2"
	else:
		logging.error("Do not recognise regularisation: %s", reg_str)
		raise ValueError()

"""
	This returns the experiment indexes (repeat indexes across all configurations) for the training/validation/testing splits
	If the splits haven't already been created in partitions.json, this function will create them
"""
def get_experiment_indexes_fixed_partitioning(
		requested_experiment_index,
		requested_partition_index,
		num_total_modeling_indexes=500,
		experiment_modeling_size=100,
		num_validation_indexes=20,
		num_experiments=10,
		folder=None):

	if folder is None:
		raise ValueError()

	experiment_spec_file = folder + "/partitions.json"

	if os.path.isfile(experiment_spec_file) == False:
		
		# For each experiment, randomly sample one to use as a testing index
		testing_indexes = [list(map(int,random.sample(range(num_total_modeling_indexes), 1)))[0] for i in range(num_experiments)]

		# For each testing_index, I need a selection of (experiment_modeling_size) modeling indexes, that I will then apply k-fold CV on
		modeling_indexes_per_experiment = [
			list(map(int,
				random.sample(
					[modeling_index for modeling_index in range(num_total_modeling_indexes) if modeling_index != testing_indexes[experiment_idx]], experiment_modeling_size
				)
			)) for experiment_idx in range(num_experiments)]

		# For each set of modeling indexes, I need to select validation indexes without replacement to form the hold-out set for k-fold CV
		validation_indexes = [
			np.random.choice(
				modeling_indexes_per_experiment[experiment_idx],
				size=(int(experiment_modeling_size/num_validation_indexes),num_validation_indexes),replace=False
			).tolist()
			for experiment_idx in range(num_experiments)]

		# For each experiment, the experiment's modelling indexes that I didn't use as a validation index, are used as training indexes
		training_indexes = [[
				[training_index for training_index in modeling_indexes_per_experiment[experiment_idx] if training_index not in validation_indexes[experiment_idx][partition_idx]]
				for partition_idx in range(int(experiment_modeling_size/num_validation_indexes))
			]
			for experiment_idx in range(num_experiments)]

		# Check that the above logic was good, and we don't have non-zero intersections between training and testing sets etc
		for exp_id in range(num_experiments):
			for part_id in range(int(experiment_modeling_size/num_validation_indexes)):
				test_id = testing_indexes[exp_id]
				validation_ids = validation_indexes[exp_id][part_id]
				training_ids = training_indexes[exp_id][part_id]
				
				if test_id in validation_ids or test_id in training_ids:
					logging.error("A testing taskset was part of the validation/training taskset")
					raise RuntimeError()
					exit(1)
				for val_id in validation_ids:
					if val_id in training_ids:
						logging.error("A validation taskset was part of the training taskset")
						raise RuntimeError()

		json_dict = {}
		json_dict["testing_indexes"] = testing_indexes
		json_dict["modeling_indexes_per_experiment"] = modeling_indexes_per_experiment
		json_dict["validation_indexes"] = validation_indexes
		json_dict["training_indexes"] = training_indexes

		with open(experiment_spec_file,'w') as f:
			json.dump(json_dict,f)
	
	try:
		with open(experiment_spec_file, "r") as read_file:
			json_data = json.load(read_file)

			testing_index = json_data["testing_indexes"][requested_experiment_index]
			training_indexes = list(sorted(json_data["training_indexes"][requested_experiment_index][requested_partition_index]))
			validation_indexes = list(sorted(json_data["validation_indexes"][requested_experiment_index][requested_partition_index]))
	
	except IOError:
		logging.error("Could not read %s as JSON.", experiment_spec_file)
		raise ValueError()

	return [testing_index], training_indexes, validation_indexes

class pca_struct:
	components_ = None

class TrainingMonitor(keras.callbacks.Callback):

	def __init__(self, timeout_seconds=None):
		self.epochs = 0
		self.timeout_seconds = timeout_seconds
		self.timeout_reached = False

	def on_train_begin(self, logs={}):
		self.start_time = datetime.now()
		self.saved_models = np.zeros(self.model.patience).tolist()

	def on_epoch_end(self, epoch, logs={}):
		self.epochs += 1

		if self.timeout_seconds is not None:
			if (datetime.now() - self.start_time).total_seconds() > self.timeout_seconds:
				logging.warning("Model exceeded timeout, stopping training")
				self.timeout_reached = True
				self.model.stop_training = True

class ComperfModel:

	"""
		A model is serialised to disk with all of the parameters that were used to build and train it
		Models may then be loaded on demand for analysis
	"""

	def __init__(self,
			model_file,
			model_metadata_file,
			training_set_indexes,
			validation_set_indexes,
			architecture=None,
			learning_rate=None,
			patience=None,
			batch_size=None,
			final_epoch=None,
			max_training_epochs=None,
			regularizer_str=None,
			weighted=False,
			timeout_reached=False):

		self.model_file = model_file
		self.model_metadata_file = model_metadata_file
		self.training_set_indexes = training_set_indexes
		self.validation_set_indexes = validation_set_indexes
		self.architecture = architecture
		self.learning_rate = learning_rate
		self.patience = patience
		self.batch_size = batch_size
		self.final_epoch = final_epoch
		self.max_training_epochs = max_training_epochs
		self.regularizer_str = regularizer_str
		self.weighted = weighted
		self.timeout_reached = timeout_reached

		self.errors = {}
		
	def add_error(self, testing_set_indexes, test_mse):
		if type(testing_set_indexes) is str:
			self.errors[testing_set_indexes] = test_mse
		else:
			self.errors[str(sorted(testing_set_indexes))] = test_mse
			
	def save(self):
		# save the model's metadata file including errors
		json_dict = {}
		json_dict["architecture"] = self.architecture
		json_dict["learning_rate"] = self.learning_rate
		json_dict["patience"] = self.patience
		json_dict["final_epoch"] = self.final_epoch
		json_dict["batch_size"] = self.batch_size
		json_dict["training_indexes"] = self.training_set_indexes
		json_dict["validation_indexes"] = self.validation_set_indexes
		json_dict["max_training_epochs"] = self.max_training_epochs
		json_dict["errors"] = self.errors
		json_dict["regularizer_str"] = self.regularizer_str
		json_dict["weighted"] = self.weighted
		json_dict["timeout_reached"] = self.timeout_reached
	
		with open(self.model_metadata_file,'w') as f:
			json.dump(json_dict,f)	

class ComperfModeller:

	def __init__(self,
			model_folder,
			architecture,
			learning_rate,
			batch_size,
			patience,
			max_training_epochs=5000,
			regulariser=None,
			weighted=False):

		self.model_folder = model_folder
		self.architecture = architecture
		self.learning_rate = learning_rate
		self.batch_size = batch_size
		self.patience = patience
		self.max_training_epochs = max_training_epochs
		self.regulariser = regulariser
		self.weighted = weighted

		if regulariser == Regularisation.L1:
			self.regulariser_str = "l1"
		elif regulariser == Regularisation.L2:
			self.regulariser_str = "l2"
		elif regulariser == Regularisation.L1L2:
			self.regulariser_str = "l1l2"
		else:
			logging.error("Do not recognise regulariser: %s", str(regulariser))
			raise ValueError()

		logging.info("Initialised new modeller for architecture: %s", str(architecture))

		self.next_model_idx = self.get_next_model_indexes(1)[0]
	
	def get_next_model_indexes(self, count):
		
		lock_filename = self.model_folder + "/mutex.lock"
		mutex = open(lock_filename, 'w+')

		# We may be running multiple modellers simultaneously on the same experiment folder
		# Therefore I need to make this a critical section
		acquire_lock(mutex)

		model_indexes = []

		# check current number of models and get the next available one
		next_model_idx = 0
		for repeat in range(count):
			for subdir, dirs, files in os.walk(self.model_folder):
				while True:
					if str(next_model_idx) not in dirs:
						ensure_exists(self.model_folder + "/" + str(next_model_idx))
						break
					next_model_idx += 1
			model_indexes.append(next_model_idx)

		release_lock(mutex)
		mutex.close()

		return model_indexes

	"""
		Go to the models folder and check each of the models (subdirectories) for a
		model that corresponds to the correct combination of training set
		indexes, validation set indexes architecture, learning_rate, patience,
		batch_size
	"""
	def get_trained_models(self,
			training_set,
			validation_set,
			testing_set):
		
		training_indexes = training_set[0]
		validation_indexes = validation_set[0]

		if testing_set != None:
			testing_indexes = testing_set[0]

		models = []
		
		lock_filename = self.model_folder + "/mutex.lock"
		mutex = open(lock_filename, 'w+')
		acquire_lock(mutex)
			
		current_num_models = 0
		for subdir, dirs, files in os.walk(self.model_folder):
			while True:
				if str(current_num_models) not in dirs:
					break
				current_num_models += 1

		logging.trace("Getting all trained models in %s", self.model_folder)
		for subdir, dirs, files in os.walk(self.model_folder):

			for model_idx in range(current_num_models):

				if str(model_idx) not in dirs:
					break

				model_metadata_json = self.model_folder + "/" + str(model_idx) + "/metadata.json"
				failure = False

				logging.trace("Reading %s", model_metadata_json)
				try:
					with open(model_metadata_json, "r") as read_file:
						metadata = json.load(read_file)

						# check architecture
						if len(metadata["architecture"]) != len(self.architecture):
							failure = True
						else:
							for hidden_layer_idx, layer_width_str in enumerate(metadata["architecture"]): # this should be a list of integers, each being the width of a hidden layer
								if int(layer_width_str) != self.architecture[hidden_layer_idx]:
									failure = True
									break
						if failure:
							logging.trace(model_metadata_json + ":incorrect architecture")
							continue 
						
						if metadata["regularizer_str"] != self.regularizer_str:
							logging.trace(model_metadata_json + ":incorrect regularizer")
							continue

						# check learning_rate, patience, batch_size
						if (float(metadata["learning_rate"]) != self.learning_rate or
								int(metadata["patience"]) != self.patience or
								int(metadata["batch_size"]) != self.batch_size):
							logging.trace(model_metadata_json + ":incorrect learning_rate, patience, or batch_size")
							continue

						# if early stopping was disabled, then specification includes max_training_epochs
						if (self.patience == 0) and (int(metadata["max_training_epochs"]) != self.max_training_epochs):
							logging.trace(model_metadata_json + ":early stopping disabled, and different max_training_epochs")
							continue
												
						# check training_indexes
						for training_index_str in metadata["training_indexes"]:
							if int(training_index_str) not in training_indexes:
								failure = True
								break
						if failure:
							logging.trace(model_metadata_json + ":incorrect training indexes")
							continue 
					
						# check validation_indexes
						for validation_index_str in metadata["validation_indexes"]:
							if int(validation_index_str) not in validation_indexes:
								failure = True
								break
						if failure:
							logging.trace(model_metadata_json + ":incorrect validation indexes")
							continue

						if "weighted" in metadata:
							if metadata["weighted"] != self.weighted:
								logging.trace(model_metadata_json + ":incorrect sample weight configuration")
								continue
						else:
							if self.weighted == True:
								logging.trace(model_metadata_json + ":sample weight configuration not found, and looking for weighted model")
								continue

						# if we get here, the model is at least one of the models that we want!
						model_file = self.model_folder + "/" + str(model_idx) + "/model.h5"

						if "timeout_reached" in metadata and metadata["timeout_reached"] == True:
							logging.debug("A model %s has the correct specification, but reached its timeout. Loading anyway.", model_metadata_json)
						else:
							logging.debug("A model %s has the correct specification", model_metadata_json)

						model = ComperfModel(
							model_file,
							model_metadata_json,
							training_indexes,
							validation_indexes,
							max_training_epochs=self.max_training_epochs)

						for testing_indexes, error in metadata["errors"].items():
							model.add_error(testing_indexes,float(error))
						
						models.append(model)

				except IOError as e:
					logging.error("The folder for model index %d exists, but has no metadata JSON file.", model_idx)
					continue
		
		release_lock(mutex)
		mutex.close()

		return models
	
	def train_models(self,
			training_set,
			validation_set,
			testing_set,
			repeats,
			mem_alloc=0.1,
			training_sample_weights=None,
			validation_sample_weights=None,
			testing_sample_weights=None,
			timeout_seconds=None):

		logging.info("Training %d new %s", repeat, ("model" if repeats == 1 else "models"))

		logging.debug("Number of training examples:" + str(len(training_set[1])))
		logging.debug("Number of validation examples:" + str(len(validation_set[1])))

		sample_weights=None
		if self.weighted == True:
			if training_sample_weights is None:
				# I want each sample to weigh its distance from the mean
				mean_training_response = np.mean(training_set[2])
				training_sample_weights = np.array([abs(response-mean_training_response) for response in training_set[2]],dtype='float32').flatten()
				validation_sample_weights = np.array([abs(response-mean_training_response) for response in validation_set[2]],dtype='float32').flatten()

		models = []

		model_indexes = [self.next_model_idx]
		test_mse = None

		if repeats > 1:
			additional_indexes = self.get_next_model_indexes(repeats-1)
			model_indexes = model_indexes + additional_indexes

		for repeat_idx in range(repeats):

			model_index = model_indexes[repeat_idx]

			logging.info("Training new model (index: %d).", model_index)
			ensure_exists(self.model_folder + "/" + str(model_index))

			save_file = self.model_folder + "/" + str(model_index) + "/model.h5"
			metadata_file = self.model_folder + "/" + str(model_index) + "/metadata.json"

			keras_model = self.build_model_keras(self.architecture, len(training_set[1][0]), self.regulariser, mem_alloc=mem_alloc)
			keras_model.patience = self.patience

			opt = keras.optimizers.Adam(lr=self.learning_rate)
			keras_model.compile(optimizer=opt,loss='mse')

			checkpoint_saver = ModelCheckpoint(save_file,
				monitor='val_loss',
				verbose=1,
				save_best_only=True,
				mode='min')
			monitor = TrainingMonitor(timeout_seconds)
			callbacks = [checkpoint_saver,monitor]

			if self.patience > 0:
				stopper = EarlyStopping(monitor='val_loss', 
					#min_delta=0.005,
					patience=self.patience,
					verbose=1,
					mode='min')
				callbacks.append(stopper)

			try:
				keras_model.fit(training_set[1].astype(np.float32),
					training_set[2].astype(np.float32).flatten(),
					batch_size=self.batch_size,
					epochs=self.max_training_epochs,
					shuffle=True,
					callbacks=callbacks,
					sample_weight=training_sample_weights.astype(np.float32).flatten(),
					validation_data=(validation_set[1].astype(np.float32),validation_set[2].astype(np.float32).flatten(),validation_sample_weights.astype(np.float32).flatten()),
					verbose=2)
			except KeyboardInterrupt:
				logging.warning("Stopping the model %s training due to keyboard interrupt", save_file)

			keras_model.load_weights(save_file) # load best model parameters (i.e. 'patience' epochs ago)
			
			logging.info("Model training completed at epoch %s (patience=%s).", str(monitor.epochs), str(self.patience))

			timed_out = False
			num_epochs = monitor.epochs - self.patience
			if timeout_seconds is not None:
				if monitor.timeout_reached:
					logging.warning("Model training timed out.")
					num_epochs = monitor.epochs

			if testing_set is not None:

				if testing_sample_weights is None:
					sample_weights = None
				else:
					sample_weights = testing_sample_weights.astype(np.float32).flatten()

				test_mse = keras_model.evaluate(
					testing_set[1].astype(np.float32),
					testing_set[2].astype(np.float32).flatten(),
					sample_weight=sample_weights, verbose=0)

				logging.info("Model errors on testing set %s had MSE %f and RMSE %f", str(testing_set[0]), test_mse, math.sqrt(test_mse))

			# Create model and save its metadata file
			saved_model = Model(
				save_file,
				metadata_file,
				training_set[0],
				validation_set[0],
				self.architecture,
				self.learning_rate,
				self.patience,
				self.batch_size,
				num_epochs,
				self.max_training_epochs,
				self.regularizer_str,
				self.weighted,
				monitor.timeout_reached)

			if testing_set is not None:
				saved_model.add_error(testing_set[0],test_mse)

			saved_model.save()

			models.append(saved_model)
			logging.info("Saved completed model to %s", metadata_file)

			backend.clear_session()

		return models
	
	def build_model_keras(self, architecture, input_num_events, regulariser=None, mem_alloc=0.049):
		
		logging.debug("Building keras architecture with %d input neurons.", input_num_events)

		backend.set_floatx('float32')

		config = tf.ConfigProto()
		#config.gpu_options.per_process_gpu_memory_fraction = mem_alloc
		config.gpu_options.allow_growth = True
		set_session(tf.Session(config=config))

		model = Sequential()
	
		keras_regulariser = None
		if regulariser == Regularisation.L1:
			keras_regulariser = regularizers.l1(0.01)
		elif regulariser == Regularisation.L2:
			keras_regulariser = regularizers.l2(0.01)
		elif regulariser == Regularisation.L1L2:
			keras_regulariser = regularizers.l1_l2(0.01)
		else:
			logging.error("Do not recognise regulariser: %s", str(regulariser))
			raise ValueError()
    
		if architecture == [0]:
			# linear model
			model.add(Dense(1,
					activation='linear',
					input_dim=input_num_events))
		else:
			model.add(Dense(architecture[0],
				input_dim=input_num_events,
				activation='relu',
				kernel_regularizer=keras_regulariser,
				use_bias=False))
			model.add(BatchNormalization())

			for layer in architecture[1:]:
				model.add(Dense(layer,
					activation='relu',
					kernel_regularizer=keras_regulariser,
					activity_regularizer=keras_regulariser,
					use_bias=False))
				model.add(BatchNormalization())

			model.add(Dense(1,
				use_bias=False,
				activation='linear'))

		return model

def run_linear_regression(
		training_taskset,
		testing_taskset,
		training_pca_matrix,
		training_event_stats
		):

	# Construct, train, then predict on testing set
	linear_regression = linear_model.LinearRegression()
	linear_regression.fit(training_taskset[1], training_taskset[2])
	linear_predictions = linear_regression.predict(testing_taskset[1])[:,0]
	
	# Get average testing errors
	mae = np.sum(np.absolute(np.array(linear_predictions) - testing_taskset[2][:,0])) / float(len(testing_taskset[2]))
	mse = mean_squared_error(testing_taskset[2][:,0], linear_predictions)
	rmse = math.sqrt(mse)

	linreg_coeff = linear_regression.coef_
	linreg_coeff_fs = np.matmul(linreg_coeff, training_pca_matrix.components_)
	linreg_coeff_fs_destandardised = ((linreg_coeff_fs * np.array(training_event_stats)[-1][1])) / np.array(training_event_stats).transpose()[1].transpose()[:-1][0]

	return mae, mse, rmse, linreg_coeff_fs_destandardised

def predict_from_model(model_file, examples):

	backend.clear_session()
	keras_model = load_model(model_file)
	predictions = keras_model.predict(examples)[:,0]

	return predictions

def run_deeplift_comparison(
		model_file,
		target_example,
		reference_example):
			
	backend.clear_session()

	deeplift_model = kc.convert_model_from_saved_files(
		model_file,
		nonlinear_mxts_mode=deeplift.layers.NonlinearMxtsMode.RevealCancel)

	deeplift_contribs_func = deeplift_model.get_target_contribs_func(
		find_scores_layer_idx=0,
		target_layer_idx=-1)

	deeplift_results = np.array(deeplift_contribs_func(task_idx=0,
		input_data_list=[[target]],
		input_references_list=[reference_example],
		batch_size=1,
		progress_update=1))

	return deeplift_results

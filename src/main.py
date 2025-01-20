import os, json, copy, sys, gc, setproctitle

from multiprocessing import set_start_method

from tensorflow.keras.models import model_from_json

from .shared import diagnostic, plot_utils as pu
from .shared.dataset_utils import quantize_dataset, save_datasets
from .shared.enums import common as cm, validation_algos as va, aggregation_algos as ag
from .shared.constants import DEFAULT_CONFIG

from .committee.main import start_committee_simulation_main
from .pos.main import start_pos_simulation_main
from .pow.main import start_pow_simulation_main

MAIN_IDENTIFIER = "Main"

# Required on WINDOWS: This line is necessary due to a dependency conflinct on Intel-OpenMP
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

# Try to improve GPU memory management
os.environ["TF_GPU_ALLOCATOR"] = "cuda_malloc_async"

def get_model():
	from tensorflow import keras as tensorflow_keras

	model = tensorflow_keras.models.Sequential()
	model.add(tensorflow_keras.layers.Conv2D(16, (5, 5), padding='same', activation='relu', input_shape=(32, 32, 3)))
	model.add(tensorflow_keras.layers.Flatten())
	model.add(tensorflow_keras.layers.Dense(10, activation='softmax'))
	return model

def get_model_from_existing_model_architecture(model_architecture_path: str):
	if type(model_architecture_path) != str:
		raise TypeError("define_random_weights_from_existing_model_architecture")
	elif os.path.exists(model_architecture_path) is False:
		raise FileNotFoundError("define_random_weights_from_existing_model_architecture")

	with open(model_architecture_path, "r") as file:
		model_architecture_str = file.read()

	return model_from_json(model_architecture_str)

def get_model_from_existing_architecture_and_weights(model_architecture_path: str, model_weights_path: str):
	import numpy as np

	if type(model_architecture_path) != str or type(model_weights_path) != str:
		raise TypeError("get_model_from_existing_architecture_and_weights method")

	# Load the model architecture
	model = get_model_from_existing_model_architecture(model_architecture_path)

	# Load the model weights
	with open(model_weights_path, "r") as file:
		weights = json.load(file)
	
	weights_list = [np.array(arr) for arr in weights]
	model.set_weights(weights_list)

	return model

def get_random_optimizer_state(model_instance):
	from tensorflow.keras.optimizers import SGD
	
	optimizer_variables = dict()

	sgd = SGD(learning_rate=0.001, momentum=0.9, weight_decay=0.001)
	model_instance.compile(optimizer=sgd, loss="sparse_categorical_crossentropy", metrics=["accuracy"])
	
	model_instance.optimizer.save_own_variables(optimizer_variables)
	optimizer_variables = {key: arr.tolist() for key, arr in optimizer_variables.items()}

	return optimizer_variables

def load_config_from_file(config_file: str) -> dict:
	'''
	Merge the default configuration with the one stored in the file. The file must be a JSON file.
	'''
   
	if type(config_file) != str:
		raise TypeError("load_config_from_file")
	
	if os.path.exists(config_file) is False:
		raise FileNotFoundError("load_config_from_file")

	if list(DEFAULT_CONFIG.keys()).sort() != cm.ConfigParams.list().sort():
		raise KeyError("load_config_from_file. Keys in the expected configuration enum do not match the keys in the default configuration")

	with open(config_file, "r") as file:
		stored_config = json.load(file)

	base_loaded_config = copy.deepcopy(DEFAULT_CONFIG)

	for key in stored_config.keys():	
		if key not in base_loaded_config:
			raise KeyError(f"load_config_from_file. Key {key} not recognized")
		
		elif key in cm.ConfigSubCategories.list():
			
			if key in [cm.ConfigSubCategories.ARCHIVE_PARAMS, cm.ConfigSubCategories.NODES_PARAMS, cm.ConfigSubCategories.DATASET_PARAMS, cm.ConfigSubCategories.MALICIOUS_NODES_PARAMS]:
				for subkey in stored_config[key]:
					if subkey not in base_loaded_config[key]:
						raise KeyError(f"load_config_from_file. Key {subkey} not recognized")
					else:
						base_loaded_config[key][subkey] = stored_config[key][subkey]
			
			elif key in [cm.ConfigSubCategories.AGGREGATION_PARAMS, cm.ConfigSubCategories.VALIDATION_PARAMS, cm.ConfigSubCategories.CONSENSUS_PARAMS]:
				# At the moment, we do not have any specific parameter for these subcategories. Consequently, we store the whole dictionary
				base_loaded_config[key] = stored_config[key]

			else:
				raise KeyError(f"load_config_from_file. Sub-category key {key} not recognized")

		else:
			base_loaded_config[key] = stored_config[key]

	return base_loaded_config

def validate_validation_and_aggregation_algorithms_params(validation_algorithm_params: dict, aggregation_algorithm_params: dict):
	if type(validation_algorithm_params) != dict or type(aggregation_algorithm_params) != dict:
		raise TypeError("validate_validation_and_aggregation_algorithms_params")
	
	if any(key not in validation_algorithm_params for key in cm.ConfigGenericValidationAlgorithmParams.list()):
		raise KeyError("validate_validation_and_aggregation_algorithms_params. Key not available in the validation algorithm parameters")
	elif validation_algorithm_params[cm.ConfigGenericValidationAlgorithmParams.TYPE] not in cm.ValidationAlgorithmType.list():
		raise ValueError("validate_validation_and_aggregation_algorithms_params. Validation algorithm type is not a valid value")

	if validation_algorithm_params[cm.ConfigGenericValidationAlgorithmParams.TYPE] == cm.ValidationAlgorithmType.LOCAL_DATASET_VALIDATION:
		if any(key not in validation_algorithm_params for key in va.LocalDatasetUsedForValidationParams.list()):
			raise KeyError("validate_validation_and_aggregation_algorithms_params. Key not available in the validation algorithm parameters")
		
		# Min update validation score first round
		if type(validation_algorithm_params[va.LocalDatasetUsedForValidationParams.MIN_UPDATE_VALIDATION_SCORE_FIRST_ROUND]) not in [float, int]:
			raise TypeError("validate_validation_and_aggregation_algorithms_params. Min update validation score first round is not a float or int")
		elif validation_algorithm_params[va.LocalDatasetUsedForValidationParams.MIN_UPDATE_VALIDATION_SCORE_FIRST_ROUND] < 0.0 or validation_algorithm_params[va.LocalDatasetUsedForValidationParams.MIN_UPDATE_VALIDATION_SCORE_FIRST_ROUND] > 1.0:
			raise ValueError("validate_validation_and_aggregation_algorithms_params. Min update validation score first round is not between 0 and 1")
		
		# Min number of updates between aggregations
		if type(validation_algorithm_params[va.LocalDatasetUsedForValidationParams.MINIMUM_NUM_OF_UPDATES_BETWEEN_AGGREGATIONS]) != int:
			raise TypeError("validate_validation_and_aggregation_algorithms_params. Minimum number of updates between aggregations is not an integer")
		elif validation_algorithm_params[va.LocalDatasetUsedForValidationParams.MINIMUM_NUM_OF_UPDATES_BETWEEN_AGGREGATIONS] < 1:
			raise ValueError("validate_validation_and_aggregation_algorithms_params. Minimum number of updates between aggregations is less than 1")
		
		# Max number of updates between aggregations
		if type(validation_algorithm_params[va.LocalDatasetUsedForValidationParams.MAXIMUM_NUM_OF_UPDATES_BETWEEN_AGGREGATIONS]) != int:
			raise TypeError("validate_validation_and_aggregation_algorithms_params. Maximum number of updates between aggregations is not an integer")
		elif validation_algorithm_params[va.LocalDatasetUsedForValidationParams.MAXIMUM_NUM_OF_UPDATES_BETWEEN_AGGREGATIONS] < validation_algorithm_params[va.LocalDatasetUsedForValidationParams.MINIMUM_NUM_OF_UPDATES_BETWEEN_AGGREGATIONS]:
			raise ValueError("validate_validation_and_aggregation_algorithms_params. Maximum number of updates between aggregations is less than the minimum number of updates between aggregations")
		
		# Count down timer to start aggregation
		if type(validation_algorithm_params[va.LocalDatasetUsedForValidationParams.COUNT_DOWN_TIMER_TO_START_AGGREGATION]) not in [int, float]:
			raise TypeError("validate_validation_and_aggregation_algorithms_params. Count down timer to start aggregation is not an int or float")
		elif validation_algorithm_params[va.LocalDatasetUsedForValidationParams.COUNT_DOWN_TIMER_TO_START_AGGREGATION] < 0:
			raise ValueError("validate_validation_and_aggregation_algorithms_params. Count down timer to start aggregation is less than 0")

	elif validation_algorithm_params[cm.ConfigGenericValidationAlgorithmParams.TYPE] == cm.ValidationAlgorithmType.KRUM:
		if any(key not in validation_algorithm_params for key in va.KrumValidationParams.list()):
			raise KeyError("validate_validation_and_aggregation_algorithms_params. Key not available in the validation algorithm parameters")

		elif type(validation_algorithm_params[va.KrumValidationParams.NUM_OF_UPDATES_TO_VALIDATE_NEGATIVELY]) != int:
			raise TypeError("validate_validation_and_aggregation_algorithms_params. Number of updates to validate negatively is not an integer")
		elif validation_algorithm_params[va.KrumValidationParams.NUM_OF_UPDATES_TO_VALIDATE_NEGATIVELY] < 0:
			raise ValueError("validate_validation_and_aggregation_algorithms_params. Number of updates to validate negatively is less than 0")

		elif type(validation_algorithm_params[va.KrumValidationParams.MINIMUM_NUM_OF_UPDATES_NEEDED_TO_START_VALIDATION]) != int:
			raise TypeError("validate_validation_and_aggregation_algorithms_params. Minimum number of updates needed to start validation is not an integer")
		elif validation_algorithm_params[va.KrumValidationParams.MINIMUM_NUM_OF_UPDATES_NEEDED_TO_START_VALIDATION] < 1:
			raise ValueError("validate_validation_and_aggregation_algorithms_params. Minimum number of updates needed to start validation is less than 1")
		elif validation_algorithm_params[va.KrumValidationParams.MINIMUM_NUM_OF_UPDATES_NEEDED_TO_START_VALIDATION] < validation_algorithm_params[va.KrumValidationParams.NUM_OF_UPDATES_TO_VALIDATE_NEGATIVELY]:
			raise ValueError("validate_validation_and_aggregation_algorithms_params. Minimum number of updates needed to start validation is less than the number of updates to validate negatively")
		
		elif type(validation_algorithm_params[va.KrumValidationParams.MAXIMUM_NUM_OF_UPDATES_NEEDED_TO_START_VALIDATION]) != int:
			raise TypeError("validate_validation_and_aggregation_algorithms_params. Maximum number of updates needed to start validation is not an integer")
		elif validation_algorithm_params[va.KrumValidationParams.MAXIMUM_NUM_OF_UPDATES_NEEDED_TO_START_VALIDATION] < validation_algorithm_params[va.KrumValidationParams.MINIMUM_NUM_OF_UPDATES_NEEDED_TO_START_VALIDATION]:
			raise ValueError("validate_validation_and_aggregation_algorithms_params. Maximum number of updates needed to start validation is less than the minimum number of updates needed to start validation")
		
		elif type(validation_algorithm_params[va.KrumValidationParams.COUNT_DOWN_TIMER_TO_START_VALIDATION]) not in [int, float]:
			raise TypeError("validate_validation_and_aggregation_algorithms_params. Count down timer to start validation is not an int or float")
		elif validation_algorithm_params[va.KrumValidationParams.COUNT_DOWN_TIMER_TO_START_VALIDATION] < 0:
			raise ValueError("validate_validation_and_aggregation_algorithms_params. Count down timer to start validation is less than 0")

		elif type(validation_algorithm_params[va.KrumValidationParams.DISTANCE_FUNCTION]) != str:
			raise TypeError("validate_validation_and_aggregation_algorithms_params. Distance function is not a string")
		elif validation_algorithm_params[va.KrumValidationParams.DISTANCE_FUNCTION] not in cm.DistanceFunctionType.list():
			raise ValueError("validate_validation_and_aggregation_algorithms_params. Distance function is not a valid value")

	elif validation_algorithm_params[cm.ConfigGenericValidationAlgorithmParams.TYPE] == cm.ValidationAlgorithmType.TRIMMED_MEAN:
		if any(key not in validation_algorithm_params for key in va.TrimmedMeanValidationParams.list()):
			raise KeyError("validate_validation_and_aggregation_algorithms_params. Key not available in the validation algorithm parameters")
		
		elif type(validation_algorithm_params[va.TrimmedMeanValidationParams.MINIMUM_NUM_OF_UPDATES_NEEDED_TO_START_VALIDATION]) != int:
			raise TypeError("validate_validation_and_aggregation_algorithms_params. Minimum number of updates needed to start validation is not an integer")
		elif validation_algorithm_params[va.TrimmedMeanValidationParams.MINIMUM_NUM_OF_UPDATES_NEEDED_TO_START_VALIDATION] < 1:
			raise ValueError("validate_validation_and_aggregation_algorithms_params. Minimum number of updates needed to start validation is less than 1")
		
		elif type(validation_algorithm_params[va.TrimmedMeanValidationParams.MAXIMUM_NUM_OF_UPDATES_NEEDED_TO_START_VALIDATION]) != int:
			raise TypeError("validate_validation_and_aggregation_algorithms_params. Maximum number of updates needed to start validation is not an integer")
		elif validation_algorithm_params[va.TrimmedMeanValidationParams.MAXIMUM_NUM_OF_UPDATES_NEEDED_TO_START_VALIDATION] < validation_algorithm_params[va.TrimmedMeanValidationParams.MINIMUM_NUM_OF_UPDATES_NEEDED_TO_START_VALIDATION]:
			raise ValueError("validate_validation_and_aggregation_algorithms_params. Maximum number of updates needed to start validation is less than the minimum number of updates needed to start validation")
		
		elif type(validation_algorithm_params[va.TrimmedMeanValidationParams.TRIMMING_PERCENTAGE]) not in [float, int]:
			raise TypeError("validate_validation_and_aggregation_algorithms_params. Trimming percentage is not a float or int")
		elif validation_algorithm_params[va.TrimmedMeanValidationParams.TRIMMING_PERCENTAGE] < 0.0 or validation_algorithm_params[va.TrimmedMeanValidationParams.TRIMMING_PERCENTAGE] >= 0.5:
			raise ValueError("validate_validation_and_aggregation_algorithms_params. Trimming percentage must be equal or greater than 0 and lower than 0.5")
		
		elif type(validation_algorithm_params[va.TrimmedMeanValidationParams.COUNT_DOWN_TIMER_TO_START_VALIDATION]) not in [int, float]:
			raise TypeError("validate_validation_and_aggregation_algorithms_params. Count down timer to start validation is not an int or float")
		elif validation_algorithm_params[va.TrimmedMeanValidationParams.COUNT_DOWN_TIMER_TO_START_VALIDATION] < 0:
			raise ValueError("validate_validation_and_aggregation_algorithms_params. Count down timer to start validation is less than 0")
		
		elif type(validation_algorithm_params[va.TrimmedMeanValidationParams.DISTANCE_FUNCTION]) != str:
			raise TypeError("validate_validation_and_aggregation_algorithms_params. Distance function is not a string")
		elif validation_algorithm_params[va.TrimmedMeanValidationParams.DISTANCE_FUNCTION] not in cm.DistanceFunctionType.list():
			raise ValueError("validate_validation_and_aggregation_algorithms_params. Distance function is not a valid value")
	
	elif validation_algorithm_params[cm.ConfigGenericValidationAlgorithmParams.TYPE] == cm.ValidationAlgorithmType.PASS_WEIGHTS:
		if any(key not in validation_algorithm_params for key in va.PassWeightsValidationParams.list()):
			raise KeyError("validate_validation_and_aggregation_algorithms_params. Key not available in the validation algorithm parameters")
		
		elif type(validation_algorithm_params[va.PassWeightsValidationParams.MINIMUM_NUM_OF_UPDATES_BETWEEN_AGGREGATIONS]) != int:
			raise TypeError("validate_validation_and_aggregation_algorithms_params. Min number of updates between aggregations is not an integer")
		elif validation_algorithm_params[va.PassWeightsValidationParams.MINIMUM_NUM_OF_UPDATES_BETWEEN_AGGREGATIONS] < 1:
			raise ValueError("validate_validation_and_aggregation_algorithms_params. Min number of updates between aggregations is less than 1")
		
		elif type(validation_algorithm_params[va.PassWeightsValidationParams.MAXIMUM_NUM_OF_UPDATES_BETWEEN_AGGREGATIONS]) != int:
			raise TypeError("validate_validation_and_aggregation_algorithms_params. Max number of updates between aggregations is not an integer")
		elif validation_algorithm_params[va.PassWeightsValidationParams.MAXIMUM_NUM_OF_UPDATES_BETWEEN_AGGREGATIONS] < validation_algorithm_params[va.PassWeightsValidationParams.MINIMUM_NUM_OF_UPDATES_BETWEEN_AGGREGATIONS]:
			raise ValueError("validate_validation_and_aggregation_algorithms_params. Max number of updates between aggregations is less than 1")
		
		elif type(validation_algorithm_params[va.PassWeightsValidationParams.COUNT_DOWN_TIMER_TO_START_AGGREGATION]) not in [int, float]:
			raise TypeError("validate_validation_and_aggregation_algorithms_params. Count down timer to start aggregation is not an int or float")
		elif validation_algorithm_params[va.PassWeightsValidationParams.COUNT_DOWN_TIMER_TO_START_AGGREGATION] < 0:
			raise ValueError("validate_validation_and_aggregation_algorithms_params. Count down timer to start aggregation is less than 0")
	
	elif validation_algorithm_params[cm.ConfigGenericValidationAlgorithmParams.TYPE] == cm.ValidationAlgorithmType.PASS_GRADIENTS:
		if any(key not in validation_algorithm_params for key in va.PassGradientsValidationParams.list()):
			raise KeyError("validate_validation_and_aggregation_algorithms_params. Key not available in the validation algorithm parameters")
		
		elif type(validation_algorithm_params[va.PassGradientsValidationParams.MINIMUM_NUM_OF_UPDATES_NEEDED_TO_START_VALIDATION]) != int:
			raise TypeError("validate_validation_and_aggregation_algorithms_params. Minimum number of updates needed to start validation is not an integer")
		elif validation_algorithm_params[va.PassGradientsValidationParams.MINIMUM_NUM_OF_UPDATES_NEEDED_TO_START_VALIDATION] < 1:
			raise ValueError("validate_validation_and_aggregation_algorithms_params. Minimum number of updates needed to start validation is less than 1")
		
		elif type(validation_algorithm_params[va.PassGradientsValidationParams.MAXIMUM_NUM_OF_UPDATES_NEEDED_TO_START_VALIDATION]) != int:
			raise TypeError("validate_validation_and_aggregation_algorithms_params. Maximum number of updates needed to start validation is not an integer")
		elif validation_algorithm_params[va.PassGradientsValidationParams.MAXIMUM_NUM_OF_UPDATES_NEEDED_TO_START_VALIDATION] < validation_algorithm_params[va.PassGradientsValidationParams.MINIMUM_NUM_OF_UPDATES_NEEDED_TO_START_VALIDATION]:
			raise ValueError("validate_validation_and_aggregation_algorithms_params. Maximum number of updates needed to start validation is less than the minimum number of updates needed to start validation")

		elif type(validation_algorithm_params[va.PassGradientsValidationParams.COUNT_DOWN_TIMER_TO_START_VALIDATION]) not in [int, float]:
			raise TypeError("validate_validation_and_aggregation_algorithms_params. Count down timer to start validation is not an int or float")
		elif validation_algorithm_params[va.PassGradientsValidationParams.COUNT_DOWN_TIMER_TO_START_VALIDATION] < 0:
			raise ValueError("validate_validation_and_aggregation_algorithms_params. Count down timer to start validation is less than 0")

	elif validation_algorithm_params[cm.ConfigGenericValidationAlgorithmParams.TYPE] == cm.ValidationAlgorithmType.GLOBAL_DATASET_VALIDATION:
		if any(key not in validation_algorithm_params for key in va.GlobalDatasetUsedForValidationParams.list()):
			raise KeyError("validate_validation_and_aggregation_algorithms_params. Key not available in the validation algorithm parameters")
		
		# Min update validation score first round
		if type(validation_algorithm_params[va.GlobalDatasetUsedForValidationParams.MIN_UPDATE_VALIDATION_SCORE_FIRST_ROUND]) not in [float, int]:
			raise TypeError("validate_validation_and_aggregation_algorithms_params. Min update validation score first round is not a float or int")
		elif validation_algorithm_params[va.GlobalDatasetUsedForValidationParams.MIN_UPDATE_VALIDATION_SCORE_FIRST_ROUND] < 0.0 or validation_algorithm_params[va.GlobalDatasetUsedForValidationParams.MIN_UPDATE_VALIDATION_SCORE_FIRST_ROUND] > 1.0:
			raise ValueError("validate_validation_and_aggregation_algorithms_params. Min update validation score first round is not between 0 and 1")
	
		# Min number of updates between aggregations
		if type(validation_algorithm_params[va.GlobalDatasetUsedForValidationParams.MINIMUM_NUM_OF_UPDATES_BETWEEN_AGGREGATIONS]) != int:
			raise TypeError("validate_validation_and_aggregation_algorithms_params. Min number of updates between aggregations is not an integer")
		elif validation_algorithm_params[va.GlobalDatasetUsedForValidationParams.MINIMUM_NUM_OF_UPDATES_BETWEEN_AGGREGATIONS] < 1:
			raise ValueError("validate_validation_and_aggregation_algorithms_params. Min number of updates between aggregations is less than 1")
		
		# Max number of updates between aggregations
		if type(validation_algorithm_params[va.GlobalDatasetUsedForValidationParams.MAXIMUM_NUM_OF_UPDATES_BETWEEN_AGGREGATIONS]) != int:
			raise TypeError("validate_validation_and_aggregation_algorithms_params. Max number of updates between aggregations is not an integer")
		elif validation_algorithm_params[va.GlobalDatasetUsedForValidationParams.MAXIMUM_NUM_OF_UPDATES_BETWEEN_AGGREGATIONS] < validation_algorithm_params[va.GlobalDatasetUsedForValidationParams.MINIMUM_NUM_OF_UPDATES_BETWEEN_AGGREGATIONS]:
			raise ValueError("validate_validation_and_aggregation_algorithms_params. Max number of updates between aggregations is less than 1")
		
		# Count down timer to start aggregation
		if type(validation_algorithm_params[va.GlobalDatasetUsedForValidationParams.COUNT_DOWN_TIMER_TO_START_AGGREGATION]) not in [int, float]:
			raise TypeError("validate_validation_and_aggregation_algorithms_params. Count down timer to start aggregation is not an int or float")
		elif validation_algorithm_params[va.GlobalDatasetUsedForValidationParams.COUNT_DOWN_TIMER_TO_START_AGGREGATION] < 0:
			raise ValueError("validate_validation_and_aggregation_algorithms_params. Count down timer to start aggregation is less than 0")

	else:
		raise ValueError("validate_validation_and_aggregation_algorithms_params. Validation algorithm type is not a valid value")

	if any(key not in aggregation_algorithm_params for key in cm.ConfigGenericAggregationAlgorithmParams.list()):
		raise KeyError("validate_validation_and_aggregation_algorithms_params. Key not available in the aggregation algorithm parameters")
	elif aggregation_algorithm_params[cm.ConfigGenericAggregationAlgorithmParams.TYPE] not in cm.AggregationAlgorithmType.list():
		raise ValueError("validate_validation_and_aggregation_algorithms_params. Aggregation algorithm type is not a valid value")

	if aggregation_algorithm_params[cm.ConfigGenericAggregationAlgorithmParams.TYPE] == cm.AggregationAlgorithmType.TRIMMED_MEAN:
		if any(key not in aggregation_algorithm_params for key in ag.TrimmedMeanAggParams.list()):
			raise KeyError("validate_validation_and_aggregation_algorithms_params. Key not available in the aggregation algorithm parameters")
		
		if type(aggregation_algorithm_params[ag.TrimmedMeanAggParams.TRIMMING_PERCENTAGE]) not in [float, int]:
			raise TypeError("validate_validation_and_aggregation_algorithms_params. Trimming percentage is not a float or int")
		elif aggregation_algorithm_params[ag.TrimmedMeanAggParams.TRIMMING_PERCENTAGE] < 0.0 or aggregation_algorithm_params[ag.TrimmedMeanAggParams.TRIMMING_PERCENTAGE] >= 0.5:
			raise ValueError("validate_validation_and_aggregation_algorithms_params. Trimming percentage must be equal or greater than 0 and lower than 0.5")

	#
	#	Constraints on the combinations of validation and aggregation algorithms
	#

	if validation_algorithm_params[cm.ConfigGenericValidationAlgorithmParams.TYPE] in cm.WeightsBasedValidationAlgorithmType.list():
		if aggregation_algorithm_params[cm.ConfigGenericAggregationAlgorithmParams.TYPE] not in cm.WeightsBasedAggregationAlgorithmType.list():
			raise ValueError("validate_validation_and_aggregation_algorithms_params. The aggregation algorithm is not compatible with the validation algorithm because the validation algorithm is weights-based while the aggregation algorithm is not weights-based")
	elif validation_algorithm_params[cm.ConfigGenericValidationAlgorithmParams.TYPE] in cm.GradientsBasedValidationAlgorithmType.list():
		if aggregation_algorithm_params[cm.ConfigGenericAggregationAlgorithmParams.TYPE] not in cm.GradientsBasedAggregationAlgorithmType.list():
			raise ValueError("validate_validation_and_aggregation_algorithms_params. The aggregation algorithm is not compatible with the validation algorithm because the validation algorithm is gradients-based while the aggregation algorithm is not gradients-based")
	else:
		raise ValueError("validate_validation_and_aggregation_algorithms_params. The validation algorithm type is not recognized")

def validate_config(stored_config: dict) -> dict:
	'''
	Validate the configuration stored in the file and return the dictionary with the run-time only parameters.
	'''

	if type(stored_config) != dict:
		raise TypeError("validate_config")
	
	for key in cm.ConfigParams.list():
		if key not in stored_config:
			raise KeyError(f"validate_config. Key {key} not available in the stored configuration")

	run_time_only_params = {
		cm.RunTimeOnlyParams.MODEL_ARCHITECTURE: None,
		cm.RunTimeOnlyParams.STARTING_WEIGHTS: None,
		cm.RunTimeOnlyParams.NODES_COMPOSITE_DATASETS: None,
		cm.RunTimeOnlyParams.NODES: None
	}

	#
	# Base params
	#

	# Is main simulation
	if stored_config[cm.ConfigParams.IS_MAIN_SIMULATION] is None:
		raise ValueError("validate_config. Is main simulation is None")
	elif type(stored_config[cm.ConfigParams.IS_MAIN_SIMULATION]) != bool:
		raise TypeError("validate_config. Is main simulation is not a boolean")
	
	is_support_machine = not stored_config[cm.ConfigParams.IS_MAIN_SIMULATION]

	# Logger path
	if stored_config[cm.ConfigParams.LOGGER_PATH] is None:
		raise ValueError("validate_config. Logger path is None")
	elif type(stored_config[cm.ConfigParams.LOGGER_PATH]) != str:
		raise TypeError("validate_config. Logger path is not a string")
	
	# Logger level
	if stored_config[cm.ConfigParams.LOGGER_LEVEL] is None:
		raise ValueError("validate_config. Logger level is None")
	elif type(stored_config[cm.ConfigParams.LOGGER_LEVEL]) != int:
		raise TypeError("validate_config. Logger level is not an integer")
	elif stored_config[cm.ConfigParams.LOGGER_LEVEL] not in [diagnostic.DEBUG, diagnostic.INFO, diagnostic.WARNING, diagnostic.ERROR, diagnostic.CRITICAL]:
		raise ValueError("validate_config. Logger level is not a valid value")
	
	# RAM usage log path
	if stored_config[cm.ConfigParams.RAM_USAGE_LOG_PATH] is None:
		raise ValueError("validate_config. RAM usage log path is None")
	elif type(stored_config[cm.ConfigParams.RAM_USAGE_LOG_PATH]) != str:
		raise TypeError("validate_config. RAM usage log path is not a string")

	# Model architecture path
	model_architecture_found = False
	model_instance = None
	model_architecture_str = None
	
	if is_support_machine:
		if stored_config[cm.ConfigParams.MODEL_ARCHITECTURE_PATH] is not None:
			raise ValueError("validate_config. Model architecture path is not None but the support machine does not need to load a model architecture")

	else:
		if stored_config[cm.ConfigParams.MODEL_ARCHITECTURE_PATH] is None:
			raise ValueError("validate_config. Model architecture path is None")
		elif type(stored_config[cm.ConfigParams.MODEL_ARCHITECTURE_PATH]) != str:
			raise TypeError("validate_config. Model architecture path is not a string")
		
		if os.path.exists(stored_config[cm.ConfigParams.MODEL_ARCHITECTURE_PATH]):
			with open(stored_config[cm.ConfigParams.MODEL_ARCHITECTURE_PATH], "r") as file:			
				run_time_only_params[cm.RunTimeOnlyParams.MODEL_ARCHITECTURE] = json.load(file)
			
			model_architecture_found = True
		
		else:
			print(f"Model architecture path {stored_config[cm.ConfigParams.MODEL_ARCHITECTURE_PATH]} does not exist. The model will be created from scratch.")

			model_instance = get_model()
			model_architecture_str = model_instance.to_json()
			run_time_only_params[cm.RunTimeOnlyParams.MODEL_ARCHITECTURE] = json.loads(model_architecture_str)

			with open(stored_config[cm.ConfigParams.MODEL_ARCHITECTURE_PATH], "w") as file:
				file.write(model_architecture_str)
		
	del model_architecture_str

	# Starting weights path
	weights = None

	if is_support_machine:
		if stored_config[cm.ConfigParams.STARTING_WEIGHTS_PATH] is not None:
			raise ValueError("validate_config. Starting weights path is not None but the support machine does not need to load starting weights")
		
	else:
		if stored_config[cm.ConfigParams.STARTING_WEIGHTS_PATH] is None:
			raise ValueError("validate_config. Starting weights path is None")
		
		elif type(stored_config[cm.ConfigParams.STARTING_WEIGHTS_PATH]) != str:
			raise TypeError("validate_config. Starting weights path is not a string")
		
		if os.path.exists(stored_config[cm.ConfigParams.STARTING_WEIGHTS_PATH]):
			with open(stored_config[cm.ConfigParams.STARTING_WEIGHTS_PATH], "r") as file:
				run_time_only_params[cm.RunTimeOnlyParams.STARTING_WEIGHTS] = json.load(file)
		
		else:
			print(f"Starting weights path {stored_config[cm.ConfigParams.STARTING_WEIGHTS_PATH]} does not exist. The weights will be randomly initialized.")
			
			if model_instance is None:
				if model_architecture_found:
					print("Model architecture found but weights not found... Creating random weights...")
					model_instance = get_model_from_existing_model_architecture(stored_config[cm.ConfigParams.MODEL_ARCHITECTURE_PATH])

				else:
					raise ValueError("validate_config. Model architecture not found and model instance is None")
		
			weights = model_instance.get_weights()
			run_time_only_params[cm.RunTimeOnlyParams.STARTING_WEIGHTS] = [weights_elem.tolist() for weights_elem in weights]

			with open(stored_config[cm.ConfigParams.STARTING_WEIGHTS_PATH], "w") as file:
				json.dump(run_time_only_params[cm.RunTimeOnlyParams.STARTING_WEIGHTS], file)

	del model_instance, weights

	# Starting optimizer state path
	if is_support_machine:
		if stored_config[cm.ConfigParams.STARTING_OPTIMIZER_STATE_PATH] is not None:
			raise ValueError("validate_config. Optimizer state path is not None but the support machine does not need to load optimizer state")
	
	else:
		if stored_config[cm.ConfigParams.STARTING_OPTIMIZER_STATE_PATH] is None:
			run_time_only_params[cm.RunTimeOnlyParams.STARTING_OPTIMIZER_STATE] = None
			print(f"Starting optimizer state path is None. If the starting optimizer state is needed, then it will be randomly initialized during the simulation and will not be stored in a file.")

		elif type(stored_config[cm.ConfigParams.STARTING_OPTIMIZER_STATE_PATH]) != str:
			raise TypeError("validate_config. Optimizer state path is neither string or None")

		else:		
			if os.path.exists(stored_config[cm.ConfigParams.STARTING_OPTIMIZER_STATE_PATH]) is True:
				with open(stored_config[cm.ConfigParams.STARTING_OPTIMIZER_STATE_PATH], "r") as file:
					run_time_only_params[cm.RunTimeOnlyParams.STARTING_OPTIMIZER_STATE] = json.load(file)
			
			else:
				model_instance = get_model_from_existing_architecture_and_weights(stored_config[cm.ConfigParams.MODEL_ARCHITECTURE_PATH], stored_config[cm.ConfigParams.STARTING_WEIGHTS_PATH])
				optimizer_variables = get_random_optimizer_state(model_instance)

				with open(stored_config[cm.ConfigParams.STARTING_OPTIMIZER_STATE_PATH], "w") as file:
					json.dump(optimizer_variables, file)
				
				run_time_only_params[cm.RunTimeOnlyParams.STARTING_OPTIMIZER_STATE] = optimizer_variables
				print(f"Starting optimizer state path {stored_config[cm.ConfigParams.STARTING_OPTIMIZER_STATE_PATH]} does not exist. The optimizer state is going to be randomly initialized and stored in a file...")

	# Number of processes to use to manage nodes
	if stored_config[cm.ConfigParams.NUM_OF_PROCESSES_TO_USE_TO_MANAGE_NODES] is None:
		raise ValueError("validate_config. Number of processes to use to manage nodes is None")
	elif type(stored_config[cm.ConfigParams.NUM_OF_PROCESSES_TO_USE_TO_MANAGE_NODES]) != int:
		raise TypeError("validate_config. Number of processes to use to manage nodes is not an integer")
	elif stored_config[cm.ConfigParams.NUM_OF_PROCESSES_TO_USE_TO_MANAGE_NODES] < 1:
		raise ValueError("validate_config. Number of processes to use to manage nodes is less than 1")

	# Maximum number of parallel malicious trainings
	if stored_config[cm.ConfigParams.MAXIMUM_NUMBER_OF_PARALLEL_HONEST_TRAININGS] is None:
		raise ValueError("validate_config. Maximum number of parallel honest trainings is None")
	elif type(stored_config[cm.ConfigParams.MAXIMUM_NUMBER_OF_PARALLEL_HONEST_TRAININGS]) != int:
		raise TypeError("validate_config. Maximum number of parallel honest trainings is not an integer")
	elif stored_config[cm.ConfigParams.MAXIMUM_NUMBER_OF_PARALLEL_HONEST_TRAININGS] < 1:
		raise ValueError("validate_config. Maximum number of parallel honest trainings is less than 1")
	
	# Maximum number of parallel honest trainings
	if stored_config[cm.ConfigParams.MAXIMUM_NUMBER_OF_PARALLEL_HONEST_TRAININGS] is None:
		raise ValueError("validate_config. Maximum number of parallel honest trainings is None")
	elif type(stored_config[cm.ConfigParams.MAXIMUM_NUMBER_OF_PARALLEL_HONEST_TRAININGS]) != int:
		raise TypeError("validate_config. Maximum number of parallel honest trainings is not an integer")
	elif stored_config[cm.ConfigParams.MAXIMUM_NUMBER_OF_PARALLEL_HONEST_TRAININGS] < 1:
		raise ValueError("validate_config. Maximum number of parallel honest trainings is less than 1")
	
	# Maximum number of parallel validations
	if stored_config[cm.ConfigParams.MAXIMUM_NUMBER_OF_PARALLEL_VALIDATIONS] is None:
		raise ValueError("validate_config. Maximum number of parallel validations is None")
	elif type(stored_config[cm.ConfigParams.MAXIMUM_NUMBER_OF_PARALLEL_VALIDATIONS]) != int:
		raise TypeError("validate_config. Maximum number of parallel validations is not an integer")
	elif stored_config[cm.ConfigParams.MAXIMUM_NUMBER_OF_PARALLEL_VALIDATIONS] < 1:
		raise ValueError("validate_config. Maximum number of parallel validations is less than 1")

	# Fit epochs
	if stored_config[cm.ConfigParams.FIT_EPOCHS] is None:
		raise ValueError("validate_config. Fit epochs is None")
	elif type(stored_config[cm.ConfigParams.FIT_EPOCHS]) != int:
		raise TypeError("validate_config. Fit epochs is not an integer")
	elif stored_config[cm.ConfigParams.FIT_EPOCHS] < 1:
		raise ValueError("validate_config. Fit epochs is less than 1")
	
	# Batch size
	if stored_config[cm.ConfigParams.BATCH_SIZE] is None:
		raise ValueError("validate_config. Batch size is None")
	elif type(stored_config[cm.ConfigParams.BATCH_SIZE]) != int:
		raise TypeError("validate_config. Batch size is not an integer")
	elif stored_config[cm.ConfigParams.BATCH_SIZE] < 1:
		raise ValueError("validate_config. Batch size is less than 1")

	# Validation with test set after model aggregation
	if stored_config[cm.ConfigParams.VALIDATION_WITH_TEST_SET_AFTER_MODEL_AGGREGATION] is None:
		raise ValueError("validate_config. Validation with test set after model aggregation is None")
	elif type(stored_config[cm.ConfigParams.VALIDATION_WITH_TEST_SET_AFTER_MODEL_AGGREGATION]) != bool:
		raise TypeError("validate_config. Validation with test set after model aggregation is not a boolean")

	# Nodes can store weights directly in the archive tmp dir
	if stored_config[cm.ConfigParams.STORE_WEIGHTS_DIRECTLY_IN_ARCHIVE_TMP_DIR] is None:
		raise ValueError("validate_config. Store weights directly in archive tmp dir is None")
	elif type(stored_config[cm.ConfigParams.STORE_WEIGHTS_DIRECTLY_IN_ARCHIVE_TMP_DIR]) != bool:
		raise TypeError("validate_config. Store weights directly in archive tmp dir is not a boolean")
	
	# Max number of rounds
	if is_support_machine is False:
		if stored_config[cm.ConfigParams.MAX_NUM_OF_ROUNDS] is None:
			raise ValueError("validate_config. Max number of rounds is None")
		elif type(stored_config[cm.ConfigParams.MAX_NUM_OF_ROUNDS]) != int:
			raise TypeError("validate_config. Max number of rounds is not an integer")
		elif stored_config[cm.ConfigParams.MAX_NUM_OF_ROUNDS] < 1:
			raise ValueError("validate_config. Max number of rounds is less than 1")

	# Need to join an existing network
	if stored_config[cm.ConfigParams.NEED_TO_JOIN_AN_EXISTING_NETWORK] is None:
		raise ValueError("validate_config. Need to join an existing network is None")
	elif type(stored_config[cm.ConfigParams.NEED_TO_JOIN_AN_EXISTING_NETWORK]) != bool:
		raise TypeError("validate_config. Need to join an existing network is not a boolean")

	if is_support_machine:
		if stored_config[cm.ConfigParams.NEED_TO_JOIN_AN_EXISTING_NETWORK] is False:
			raise ValueError("validate_config. Need to join an existing network is False but the support machine needs to join an existing network")
	else:
		if type(stored_config[cm.ConfigParams.NEED_TO_JOIN_AN_EXISTING_NETWORK]) is True:
			raise ValueError("validate_config. Need to join an existing network is True but the main machine does not need to join an existing network")
	
	# Entry point nodes
	if stored_config[cm.ConfigParams.ENTRY_POINT_NODES] is None:
		raise ValueError("validate_config. Entry point nodes is None")
	elif type(stored_config[cm.ConfigParams.ENTRY_POINT_NODES]) != list:
		raise TypeError("validate_config. Entry point nodes is not a list")
	
	if is_support_machine:
		if stored_config[cm.ConfigParams.ENTRY_POINT_NODES] == []:
			raise ValueError("validate_config. Entry point nodes is empty but the support machine needs to join an existing network")
		elif any(type(elem) != dict or any(key not in elem for key in cm.ConfigEntryPointNodesParams.list()) for elem in stored_config[cm.ConfigParams.ENTRY_POINT_NODES]):
			raise TypeError("validate_config. Entry point nodes contains an element that is not a dictionary or that does not have all the required keys")
		
		for elem in stored_config[cm.ConfigParams.ENTRY_POINT_NODES]:
			if type(elem[cm.ConfigEntryPointNodesParams.HOST]) != str:
				raise TypeError("validate_config. Entry point node host is not a string")
			elif type(elem[cm.ConfigEntryPointNodesParams.PORT]) != int:
				raise TypeError("validate_config. Entry point node port is not an integer")
			elif elem[cm.ConfigEntryPointNodesParams.PORT] < 0:
				raise ValueError("validate_config. Entry point node port is less than 0")
			elif elem[cm.ConfigEntryPointNodesParams.PORT] > 65535:
				raise ValueError("validate_config. Entry point node port is greater than 65535")
	else:
		if stored_config[cm.ConfigParams.ENTRY_POINT_NODES] != []:
			raise ValueError("validate_config. Entry point nodes is not empty but the main machine does not need to join an existing network")

	#
	# Nodes params
	#

	if stored_config[cm.ConfigParams.NODES_PARAMS] is None:
		raise ValueError("validate_config. Nodes params is None")
	elif type(stored_config[cm.ConfigParams.NODES_PARAMS]) != dict:
		raise TypeError("validate_config. Nodes params is not a dictionary")
	
	for key in cm.ConfigNodesParams.list():
		if key not in stored_config[cm.ConfigParams.NODES_PARAMS]:
			raise KeyError(f"validate_config. Key {key} not available in the stored configuration")
	
	first_node_id = None
	nodes_params = stored_config[cm.ConfigParams.NODES_PARAMS]

	# Overall number of nodes
	if nodes_params[cm.ConfigNodesParams.OVERALL_NUM_OF_NODES] is None:
		raise ValueError("validate_config. Overall number of nodes is None")
	elif type(nodes_params[cm.ConfigNodesParams.OVERALL_NUM_OF_NODES]) != int:
		raise TypeError("validate_config. Overall number of nodes is not an integer")
	elif nodes_params[cm.ConfigNodesParams.OVERALL_NUM_OF_NODES] < 1:
		raise ValueError("validate_config. Overall number of nodes is less than 1")
	elif nodes_params[cm.ConfigNodesParams.OVERALL_NUM_OF_NODES] < stored_config[cm.ConfigParams.NUM_OF_PROCESSES_TO_USE_TO_MANAGE_NODES]:
		raise ValueError("validate_config. Overall number of nodes is lower than the number of processes to use to manage nodes")

	# First node ID
	if nodes_params[cm.ConfigNodesParams.FIRST_NODE_ID] is None:
		raise ValueError("validate_config. First node ID is None")
	elif type(nodes_params[cm.ConfigNodesParams.FIRST_NODE_ID]) != int:
		raise TypeError("validate_config. First node ID is not an integer")
	elif nodes_params[cm.ConfigNodesParams.FIRST_NODE_ID] < 0:
		raise ValueError("validate_config. First node ID is less than 0")
	
	first_node_id = nodes_params[cm.ConfigNodesParams.FIRST_NODE_ID]

	# Host
	if nodes_params[cm.ConfigNodesParams.HOST] is None:
		raise ValueError("validate_config. Host is None")
	elif type(nodes_params[cm.ConfigNodesParams.HOST]) != str:
		raise TypeError("validate_config. Host is not a string")
	
	# First port
	if nodes_params[cm.ConfigNodesParams.FIRST_PORT] is None:
		raise ValueError("validate_config. First port is None")
	elif type(nodes_params[cm.ConfigNodesParams.FIRST_PORT]) != int:
		raise TypeError("validate_config. First port is not an integer")
	elif nodes_params[cm.ConfigNodesParams.FIRST_PORT] < 0 or nodes_params[cm.ConfigNodesParams.FIRST_PORT] > 65535:
		raise ValueError("validate_config. First port is less than 0 or greater than 65535")
	
	# List of nodes allowed to produce debug log messages
	if nodes_params[cm.ConfigNodesParams.LIST_OF_NODES_ALLOWED_TO_PRODUCE_DEBUG_LOG_MESSAGES] is None:
		raise ValueError("validate_config. List of nodes allowed to produce debug log messages is None")
	elif type(nodes_params[cm.ConfigNodesParams.LIST_OF_NODES_ALLOWED_TO_PRODUCE_DEBUG_LOG_MESSAGES]) != list:
		raise TypeError("validate_config. List of nodes allowed to produce debug log messages is not a list")
	elif any(type(elem) != int for elem in nodes_params[cm.ConfigNodesParams.LIST_OF_NODES_ALLOWED_TO_PRODUCE_DEBUG_LOG_MESSAGES]):
		raise TypeError("validate_config. List of nodes allowed to produce debug log messages contains an element that is not an integer")

	# Define the nodes

	run_time_only_params[cm.RunTimeOnlyParams.NODES] = []

	for i in range(nodes_params[cm.ConfigNodesParams.OVERALL_NUM_OF_NODES]):
		node_id = nodes_params[cm.ConfigNodesParams.FIRST_NODE_ID] + i

		if node_id in nodes_params[cm.ConfigNodesParams.LIST_OF_NODES_ALLOWED_TO_PRODUCE_DEBUG_LOG_MESSAGES]:
			allowed_to_produce_debug_log_messages = True
		else:
			allowed_to_produce_debug_log_messages = False

		run_time_only_params[cm.RunTimeOnlyParams.NODES].append({
			cm.RunTimeNodeParams.NODE_ID: node_id,
			cm.RunTimeNodeParams.HOST: nodes_params[cm.ConfigNodesParams.HOST],
			cm.RunTimeNodeParams.PORT: nodes_params[cm.ConfigNodesParams.FIRST_PORT] + i,
			cm.RunTimeNodeParams.ALLOWED_TO_PRODUCE_DEBUG_LOG_MESSAGES: allowed_to_produce_debug_log_messages
		})

	del nodes_params
	
	#
	# Dataset params
	#

	if stored_config[cm.ConfigParams.DATASET_PARAMS] is None:
		raise ValueError("validate_config. Dataset params is None")
	elif type(stored_config[cm.ConfigParams.DATASET_PARAMS]) != dict:
		raise TypeError("validate_config. Dataset params is not a dictionary")
	
	for key in cm.ConfigDatasetParams.list():
		if key not in stored_config[cm.ConfigParams.DATASET_PARAMS]:
			raise KeyError(f"validate_config. Key {key} not available in the stored configuration")

	dataset_params = stored_config[cm.ConfigParams.DATASET_PARAMS]

	# Dataset
	if dataset_params[cm.ConfigDatasetParams.DATASET] is None:
		raise ValueError("validate_config. Dataset is None")
	elif type(dataset_params[cm.ConfigDatasetParams.DATASET]) != str:
		raise TypeError("validate_config. Dataset is not a string")
	elif dataset_params[cm.ConfigDatasetParams.DATASET] not in cm.AvailableDataset.list():
		raise ValueError("validate_config. Dataset is unavailable")
		
	# Temperature
	if dataset_params[cm.ConfigDatasetParams.TEMPERATURE] is None:
		raise ValueError("validate_config. Temperature is None")
	elif type(dataset_params[cm.ConfigDatasetParams.TEMPERATURE]) not in [float, int]:
		raise TypeError("validate_config. Temperature is not a float or int")
	elif dataset_params[cm.ConfigDatasetParams.TEMPERATURE] < 0.0 or dataset_params[cm.ConfigDatasetParams.TEMPERATURE] > 1.0:
		raise ValueError("validate_config. Temperature is not between 0 and 1")
	
	# Number of quanta
	if dataset_params[cm.ConfigDatasetParams.NUM_OF_QUANTA] is None:
		raise ValueError("validate_config. Number of quanta is None")
	elif type(dataset_params[cm.ConfigDatasetParams.NUM_OF_QUANTA]) != int:
		raise TypeError("validate_config. Number of quanta is not an integer")
	elif dataset_params[cm.ConfigDatasetParams.NUM_OF_QUANTA] < 0:
		raise ValueError("validate_config. Number of quanta is less than 0")
	
	# Percentage of IID quanta
	if dataset_params[cm.ConfigDatasetParams.PERC_OF_IID_QUANTA] is None:
		raise ValueError("validate_config. Percentage of IID quanta is None")
	elif type(dataset_params[cm.ConfigDatasetParams.PERC_OF_IID_QUANTA]) not in [float, int]:
		raise TypeError("validate_config. Percentage of IID quanta is not a float or int")
	elif dataset_params[cm.ConfigDatasetParams.PERC_OF_IID_QUANTA] < 0.0 or dataset_params[cm.ConfigDatasetParams.PERC_OF_IID_QUANTA] > 1.0:
		raise ValueError("validate_config. Percentage of IID quanta is not between 0 and 1")
	
	# dataset lazy loading
	if dataset_params[cm.ConfigDatasetParams.LAZY_LOADING] is None:
		raise ValueError("validate_config. Lazy loading is None")
	elif type(dataset_params[cm.ConfigDatasetParams.LAZY_LOADING]) != bool:
		raise TypeError("validate_config. Lazy loading is not a boolean")
	
	# Dataset files path
	if dataset_params[cm.ConfigDatasetParams.DATASET_FILES_DIR_PATH] is None:
		raise ValueError("validate_config. Dataset files path is None")
	elif type(dataset_params[cm.ConfigDatasetParams.DATASET_FILES_DIR_PATH]) != str:
		raise TypeError("validate_config. Dataset files path is not a string")
	# We don't need to check if the path exists, because the dataset files will be created if they don't exist

	# Nodes composite datasets
	if dataset_params[cm.ConfigDatasetParams.NODES_COMPOSITE_DATASETS] is None:
		raise ValueError("validate_config. Nodes composite datasets is None")
	elif type(dataset_params[cm.ConfigDatasetParams.NODES_COMPOSITE_DATASETS]) != list:
		raise TypeError("validate_config. Nodes composite datasets is not a list")
	elif any(type(elem) != dict or any(key not in elem for key in cm.ConfigNodesCompositeDatasetsParams.list()) for elem in dataset_params[cm.ConfigDatasetParams.NODES_COMPOSITE_DATASETS]):
		raise TypeError("validate_config. Nodes composite datasets contains an element that is not a dictionary or that does not have all the required keys")

	run_time_only_params[cm.RunTimeOnlyParams.NODES_COMPOSITE_DATASETS] = []
	sum_of_handled_nodes = 0

	for elem in dataset_params[cm.ConfigDatasetParams.NODES_COMPOSITE_DATASETS]:
		if type(elem[cm.ConfigNodesCompositeDatasetsParams.ALIAS]) != str:
			raise TypeError("load_config_from_file method. Nodes composite dataset alias is not a string")
		elif type(elem[cm.ConfigNodesCompositeDatasetsParams.NUM_OF_NODES]) != int:
			raise TypeError("load_config_from_file method. Nodes composite dataset number of nodes is not an integer")
		elif elem[cm.ConfigNodesCompositeDatasetsParams.NUM_OF_NODES] < 0:
			raise ValueError("load_config_from_file method. Nodes composite dataset number of nodes is less than 0")
		elif type(elem[cm.ConfigNodesCompositeDatasetsParams.NUM_OF_QUANTA_TO_USE]) != int:
			raise TypeError("load_config_from_file method. Nodes composite dataset number of quanta to use is not an integer")
		elif elem[cm.ConfigNodesCompositeDatasetsParams.NUM_OF_QUANTA_TO_USE] < 0:
			raise ValueError("load_config_from_file method. Nodes composite dataset number of quanta to use is less than 0")
		elif elem[cm.ConfigNodesCompositeDatasetsParams.NUM_OF_QUANTA_TO_USE] > dataset_params[cm.ConfigDatasetParams.NUM_OF_QUANTA]:
			raise ValueError("load_config_from_file method. Nodes composite dataset number of quanta to use is greater than the total number of quanta")
		elif type(elem[cm.ConfigNodesCompositeDatasetsParams.IID_QUANTA_TO_USE]) != int:
			raise TypeError("load_config_from_file method. Nodes composite dataset number of IID quanta to use is not an integer")
		elif elem[cm.ConfigNodesCompositeDatasetsParams.IID_QUANTA_TO_USE] < 0 or elem[cm.ConfigNodesCompositeDatasetsParams.IID_QUANTA_TO_USE] > elem[cm.ConfigNodesCompositeDatasetsParams.NUM_OF_QUANTA_TO_USE]:
			raise ValueError("load_config_from_file method. Nodes composite dataset number of IID quanta to use is less than 0 or greater than the total number of quanta")
		
		run_time_only_params[cm.RunTimeOnlyParams.NODES_COMPOSITE_DATASETS].append({
			cm.RunTimeNodeCompositeDatasetParams.ALIAS: elem[cm.ConfigNodesCompositeDatasetsParams.ALIAS],
			cm.RunTimeNodeCompositeDatasetParams.RELATED_NODES_IDS: [first_node_id + node_id for node_id in range(sum_of_handled_nodes, sum_of_handled_nodes + elem[cm.ConfigNodesCompositeDatasetsParams.NUM_OF_NODES])],
			cm.RunTimeNodeCompositeDatasetParams.NUM_OF_QUANTA_TO_USE: elem[cm.ConfigNodesCompositeDatasetsParams.NUM_OF_QUANTA_TO_USE],
			cm.RunTimeNodeCompositeDatasetParams.IID_QUANTA_TO_USE: elem[cm.ConfigNodesCompositeDatasetsParams.IID_QUANTA_TO_USE]
		})

		sum_of_handled_nodes += elem[cm.ConfigNodesCompositeDatasetsParams.NUM_OF_NODES]
	
	if sum_of_handled_nodes < stored_config[cm.ConfigParams.NODES_PARAMS][cm.ConfigNodesParams.OVERALL_NUM_OF_NODES]:
		raise ValueError(f"load_config_from_file method. The sum of the number of nodes in the nodes composite datasets is less than the total number of nodes. Total number of nodes: {stored_config[cm.ConfigParams.NODES_PARAMS][cm.ConfigNodesParams.OVERALL_NUM_OF_NODES]}, sum of the number of nodes in the nodes composite datasets: {sum_of_handled_nodes}")
	
	del sum_of_handled_nodes

	#
	# Archive params
	#

	if stored_config[cm.ConfigParams.ARCHIVE_PARAMS] is None:
		raise ValueError("validate_config. Archive params is None")
	elif type(stored_config[cm.ConfigParams.ARCHIVE_PARAMS]) != dict:
		raise TypeError("validate_config. Archive params is not a dictionary")
	
	for key in cm.ConfigArchiveParams.list():
		if key not in stored_config[cm.ConfigParams.ARCHIVE_PARAMS]:
			raise KeyError(f"validate_config. Archive Key {key} not available in the stored configuration")
	
	archive_params = stored_config[cm.ConfigParams.ARCHIVE_PARAMS]

	# Host
	if archive_params[cm.ConfigArchiveParams.HOST] is None:
		raise ValueError("validate_config. Archive host is None")
	elif type(archive_params[cm.ConfigArchiveParams.HOST]) != str:
		raise TypeError("validate_config. Archive host is not a string")
	
	# Port
	if archive_params[cm.ConfigArchiveParams.PORT] is None:
		raise ValueError("validate_config. Archive port is None")
	elif type(archive_params[cm.ConfigArchiveParams.PORT]) != int:
		raise TypeError("validate_config. Archive port is not an integer")
	elif archive_params[cm.ConfigArchiveParams.PORT] < 0 or archive_params[cm.ConfigArchiveParams.PORT] > 65535:
		raise ValueError("validate_config. Archive port is less than 0 or greater than 65535")
	
	# Archive must be created
	if archive_params[cm.ConfigArchiveParams.ARCHIVE_MUST_BE_CREATED] is None:
		raise ValueError("validate_config. Create archive is None")
	elif type(archive_params[cm.ConfigArchiveParams.ARCHIVE_MUST_BE_CREATED]) != bool:
		raise TypeError("validate_config. Create archive is not a boolean")
	
	if is_support_machine:
		if archive_params[cm.ConfigArchiveParams.ARCHIVE_MUST_BE_CREATED] is True:
			raise ValueError("validate_config. Create archive is True but the support machine cannot create an archive")

	# Temporary directory
	if stored_config[cm.ConfigParams.STORE_WEIGHTS_DIRECTLY_IN_ARCHIVE_TMP_DIR] is True or archive_params[cm.ConfigArchiveParams.ARCHIVE_MUST_BE_CREATED] is True:
		if archive_params[cm.ConfigArchiveParams.TMP_DIR] is None:
			raise ValueError("validate_config. Archive temporary directory is None")
		elif type(archive_params[cm.ConfigArchiveParams.TMP_DIR]) != str:
			raise TypeError("validate_config. Archive temporary directory is not a string")

	if archive_params[cm.ConfigArchiveParams.ARCHIVE_MUST_BE_CREATED] is True:
		# Persistent mode
		if archive_params[cm.ConfigArchiveParams.PERSISTENT_MODE] is None:
			raise ValueError("validate_config. Persist mode is None")
		elif type(archive_params[cm.ConfigArchiveParams.PERSISTENT_MODE]) != bool:
			raise TypeError("validate_config. Persist mode is not a boolean")

		# Archive logger path
		if archive_params[cm.ConfigArchiveParams.LOGGER_PATH] is None:
			raise ValueError("validate_config. Archive logger path is None")
		elif type(archive_params[cm.ConfigArchiveParams.LOGGER_PATH]) != str:
			raise TypeError("validate_config. Archive logger path is not a string")
	
		# Archive logger level
		if archive_params[cm.ConfigArchiveParams.LOGGER_LEVEL] is None:
			raise ValueError("validate_config. Archive logger level is None")
		elif type(archive_params[cm.ConfigArchiveParams.LOGGER_LEVEL]) != int:
			raise TypeError("validate_config. Archive logger level is not an integer")
		elif archive_params[cm.ConfigArchiveParams.LOGGER_LEVEL] not in [diagnostic.DEBUG, diagnostic.INFO, diagnostic.WARNING, diagnostic.ERROR, diagnostic.CRITICAL]:
			raise ValueError("validate_config. Archive logger level is not a valid value")

	del archive_params

	#
	# Malicious nodes params
	#

	if stored_config[cm.ConfigParams.MALICIOUS_NODES_PARAMS] is None:
		raise ValueError("validate_config. Malicious nodes params is None")
	elif type(stored_config[cm.ConfigParams.MALICIOUS_NODES_PARAMS]) != dict:
		raise TypeError("validate_config. Malicious nodes params is not a dictionary")
	
	for key in cm.ConfigMaliciousNodesParams.list():
		if key not in stored_config[cm.ConfigParams.MALICIOUS_NODES_PARAMS]:
			raise KeyError(f"validate_config. Malicious nodes Key {key} not available in the stored configuration")

	malicious_nodes_params = stored_config[cm.ConfigParams.MALICIOUS_NODES_PARAMS]

	# Number of malicious nodes
	if malicious_nodes_params[cm.ConfigMaliciousNodesParams.NUM_OF_MALICIOUS_NODES] is None:
		raise ValueError("validate_config. Number of malicious nodes is None")
	elif type(malicious_nodes_params[cm.ConfigMaliciousNodesParams.NUM_OF_MALICIOUS_NODES]) != int:
		raise TypeError("validate_config. Number of malicious nodes is not an integer")
	elif malicious_nodes_params[cm.ConfigMaliciousNodesParams.NUM_OF_MALICIOUS_NODES] < 0:
		raise ValueError("validate_config. Number of malicious nodes is less than 0")
	elif malicious_nodes_params[cm.ConfigMaliciousNodesParams.NUM_OF_MALICIOUS_NODES] > stored_config[cm.ConfigParams.NODES_PARAMS][cm.ConfigNodesParams.OVERALL_NUM_OF_NODES]:
		raise ValueError("validate_config. Number of malicious nodes is greater than the total number of nodes")
	
	# List of collusion peer ids
	if malicious_nodes_params[cm.ConfigMaliciousNodesParams.LIST_OF_COLLUSION_PEER_IDS] is None:
		raise ValueError("validate_config. List of collusion peer IDs is None")
	elif type(malicious_nodes_params[cm.ConfigMaliciousNodesParams.LIST_OF_COLLUSION_PEER_IDS]) != list:
		raise TypeError("validate_config. List of collusion peer IDs is not a list")
	elif any(type(elem) != int for elem in malicious_nodes_params[cm.ConfigMaliciousNodesParams.LIST_OF_COLLUSION_PEER_IDS]):
		raise TypeError("validate_config. List of collusion peer IDs contains an element that is not an integer")

	tmp_list_of_collusion_peers = list()

	# Get the list of nodes that are part of this configuration (so that is going to be created) and that are inside the list of collusion peer ids
	for node in run_time_only_params[cm.RunTimeOnlyParams.NODES]:
		if node[cm.RunTimeNodeParams.NODE_ID] in malicious_nodes_params[cm.ConfigMaliciousNodesParams.LIST_OF_COLLUSION_PEER_IDS]:
			tmp_list_of_collusion_peers.append(node[cm.RunTimeNodeParams.NODE_ID])

	# Malicious nodes
	if malicious_nodes_params[cm.ConfigMaliciousNodesParams.NODE_BEHAVIOURS] is None:
		raise ValueError("validate_config. Malicious nodes is None")
	elif type(malicious_nodes_params[cm.ConfigMaliciousNodesParams.NODE_BEHAVIOURS]) != list:
		raise TypeError("validate_config. Malicious nodes is not a list")
	
	num_of_malicious_nodes = 0
	malicious_node_ids = []

	for elem in malicious_nodes_params[cm.ConfigMaliciousNodesParams.NODE_BEHAVIOURS]:
		if type(elem) != dict:
			raise TypeError("validate_config. Malicious node is not a dictionary")
		
		# Mandatory keys
		
		# Type
		if cm.ConfigNodeBehaviorParams.TYPE not in elem:
			raise KeyError("validate_config. Malicious node does not have the type key")
		elif elem[cm.ConfigNodeBehaviorParams.TYPE] not in cm.MaliciousNodeBehaviourType.list():
			raise ValueError("validate_config. Malicious node type is not a valid value")
		
		# Nodes ID
		if cm.ConfigNodeBehaviorParams.NODES_ID not in elem:
			raise KeyError("validate_config. Malicious node does not have the nodes ID key")
		elif type(elem[cm.ConfigNodeBehaviorParams.NODES_ID]) != list:
			raise TypeError("validate_config. Malicious node label flipping nodes ID is not a list")
		elif type(node_id) != int:
			raise TypeError("validate_config. Malicious node label flipping nodes ID contains an element that is not an integer")
		
		# check if the malicious node ids are part of the configuration
		for node_id in elem[cm.ConfigNodeBehaviorParams.NODES_ID]:
			if any(node_id == node[cm.RunTimeNodeParams.NODE_ID] for node in run_time_only_params[cm.RunTimeOnlyParams.NODES]) is False:
				raise ValueError(f"validate_config. Malicious node ID {node_id} is not part of the configuration")

		# remove malicious nodes from collusion peers and add them to the list of malicious nodes to check for duplicates
		for node_id in elem[cm.ConfigNodeBehaviorParams.NODES_ID]:
			if node_id in tmp_list_of_collusion_peers:
				tmp_list_of_collusion_peers.remove(node_id)
			
			if node_id in malicious_node_ids:
				raise ValueError(f"validate_config. Malicious node ID is duplicated. Malicious node ID: {node_id}")
			
			malicious_node_ids.append(node_id)

		# Number of samples
		if cm.ConfigNodeBehaviorParams.NUM_OF_SAMPLES not in elem:
			raise KeyError("validate_config. Malicious node does not have the number of samples key")
		elif type(elem[cm.ConfigNodeBehaviorParams.NUM_OF_SAMPLES]) != int:
			raise TypeError("validate_config. Malicious node number of samples is not an integer")
		elif elem[cm.ConfigNodeBehaviorParams.NUM_OF_SAMPLES] < 0:
			raise ValueError("validate_config. Malicious node number of samples is less than 0")

		# Starting round for the malicious behaviour
		if cm.ConfigNodeBehaviorParams.STARTING_ROUND_FOR_MALICIOUS_BEHAVIOUR not in elem:
			raise KeyError("validate_config. Malicious node does not have the starting round for malicious behaviour key")
		elif type(elem[cm.ConfigNodeBehaviorParams.STARTING_ROUND_FOR_MALICIOUS_BEHAVIOUR]) != int:
			raise TypeError("validate_config. Malicious node starting round for malicious behaviour is not an integer")
		elif elem[cm.ConfigNodeBehaviorParams.STARTING_ROUND_FOR_MALICIOUS_BEHAVIOUR] < 1:
			raise ValueError("validate_config. Malicious node starting round for malicious behaviour is less than 1")

		# Optional keys
		optional_keys = list()

		if elem[cm.ConfigNodeBehaviorParams.TYPE] == cm.MaliciousNodeBehaviourType.LABEL_FLIPPING:
			optional_keys = [cm.ConfigNodeBehaviorParams.SELECTED_CLASSES, cm.ConfigNodeBehaviorParams.TARGET_CLASSES]
		
		elif elem[cm.ConfigNodeBehaviorParams.TYPE] == cm.MaliciousNodeBehaviourType.TARGETED_POISONING:
			optional_keys = [cm.ConfigNodeBehaviorParams.TARGET_CLASS, cm.ConfigNodeBehaviorParams.SQUARE_SIZE]
			
		elif elem[cm.ConfigNodeBehaviorParams.TYPE] == cm.MaliciousNodeBehaviourType.RANDOM_LABEL:
			optional_keys = []

		elif elem[cm.ConfigNodeBehaviorParams.TYPE] == cm.MaliciousNodeBehaviourType.ADDITIVE_NOISE:
			optional_keys = [cm.ConfigNodeBehaviorParams.SIGMA]
		
		elif elem[cm.ConfigNodeBehaviorParams.TYPE] == cm.MaliciousNodeBehaviourType.RANDOM_NOISE:
			optional_keys = []

		else:
			raise ValueError("validate_config. Malicious node type is not expected")
		
		for key in optional_keys:
			if key == cm.ConfigNodeBehaviorParams.SELECTED_CLASSES:
				if type(elem[cm.ConfigNodeBehaviorParams.SELECTED_CLASSES]) != list:
					raise TypeError("validate_config. Malicious node selected classes is not a list")
				elif any(type(class_id) != int for class_id in elem[cm.ConfigNodeBehaviorParams.SELECTED_CLASSES]):
					raise TypeError("validate_config. Malicious node selected classes contains an element that is not an integer")

				if dataset_params[cm.ConfigDatasetParams.DATASET] in [cm.AvailableDataset.CIFAR10, cm.AvailableDataset.MNIST]:
					if any(class_id < 0 or class_id > 9 for class_id in elem[cm.ConfigNodeBehaviorParams.SELECTED_CLASSES]):
						raise ValueError("validate_config. Malicious node selected classes contains an element that is not between 0 and 9")
				elif dataset_params[cm.ConfigDatasetParams.DATASET] in [cm.AvailableDataset.CIFAR100]:
					if any(class_id < 0 or class_id > 99 for class_id in elem[cm.ConfigNodeBehaviorParams.SELECTED_CLASSES]):
						raise ValueError("validate_config. Malicious node selected classes contains an element that is not between 0 and 99")
				else:
					raise ValueError("validate_config. Dataset not recognized")
				
			elif key == cm.ConfigNodeBehaviorParams.TARGET_CLASSES:
				if type(elem[cm.ConfigNodeBehaviorParams.TARGET_CLASSES]) != list:
					raise TypeError("validate_config. Malicious node target classes is not a list")
				elif any(type(class_id) != int for class_id in elem[cm.ConfigNodeBehaviorParams.TARGET_CLASSES]):
					raise TypeError("validate_config. Malicious node target classes contains an element that is not an integer")
				
				if dataset_params[cm.ConfigDatasetParams.DATASET] in [cm.AvailableDataset.CIFAR10, cm.AvailableDataset.MNIST]:
					if any(class_id < 0 or class_id > 9 for class_id in elem[cm.ConfigNodeBehaviorParams.TARGET_CLASSES]):
						raise ValueError("validate_config. Malicious node target classes contains an element that is not between 0 and 9")
				elif dataset_params[cm.ConfigDatasetParams.DATASET] in [cm.AvailableDataset.CIFAR100]:
					if any(class_id < 0 or class_id > 99 for class_id in elem[cm.ConfigNodeBehaviorParams.TARGET_CLASSES]):
						raise ValueError("validate_config. Malicious node target classes contains an element that is not between 0 and 99")
				else:
					raise ValueError("validate_config. Dataset not recognized")
				
			elif key == cm.ConfigNodeBehaviorParams.TARGET_CLASS:
				if type(elem[cm.ConfigNodeBehaviorParams.TARGET_CLASS]) != int:
					raise TypeError("validate_config. Malicious node target class is not an integer")
				elif elem[cm.ConfigNodeBehaviorParams.TARGET_CLASS] < 0:
					raise ValueError("validate_config. Malicious node target class is less than 0")
				
				if dataset_params[cm.ConfigDatasetParams.DATASET] in [cm.AvailableDataset.CIFAR10, cm.AvailableDataset.MNIST]:
					if elem[cm.ConfigNodeBehaviorParams.TARGET_CLASS] < 0 or elem[cm.ConfigNodeBehaviorParams.TARGET_CLASS] > 9:
						raise ValueError("validate_config. Malicious node target class is not between 0 and 9")
				elif dataset_params[cm.ConfigDatasetParams.DATASET] in [cm.AvailableDataset.CIFAR100]:
					if elem[cm.ConfigNodeBehaviorParams.TARGET_CLASS] < 0 or elem[cm.ConfigNodeBehaviorParams.TARGET_CLASS] > 99:
						raise ValueError("validate_config. Malicious node target class is not between 0 and 99")
				else:
					raise ValueError("validate_config. Dataset not recognized")
				
			elif key == cm.ConfigNodeBehaviorParams.SQUARE_SIZE:
				if type(elem[cm.ConfigNodeBehaviorParams.SQUARE_SIZE]) != int:
					raise TypeError("validate_config. Malicious node square size is not an integer")
				elif elem[cm.ConfigNodeBehaviorParams.SQUARE_SIZE] < 0:
					raise ValueError("validate_config. Malicious node square size is less than 0")
				
			elif key == cm.ConfigNodeBehaviorParams.SIGMA:
				if type(elem[cm.ConfigNodeBehaviorParams.SIGMA]) not in [float, int]:
					raise TypeError("validate_config. Malicious node sigma is not a float or int")
				elif elem[cm.ConfigNodeBehaviorParams.SIGMA] <= 0.0:
					raise ValueError("validate_config. Malicious node sigma is less than 0.0")
				
		num_of_malicious_nodes += len(elem[cm.ConfigNodeBehaviorParams.NODES_ID])

	if num_of_malicious_nodes != malicious_nodes_params[cm.ConfigMaliciousNodesParams.NUM_OF_MALICIOUS_NODES]:
		raise ValueError("validate_config. The number of malicious nodes is different from the expected number of malicious nodes")

	if len(tmp_list_of_collusion_peers) > 0:
		raise ValueError(f"validate_config. The list of collusion peers contains honest nodes. Honest node ids: {tmp_list_of_collusion_peers}")

	del num_of_malicious_nodes, malicious_nodes_params

	#
	# Validation and aggregation algorithms params
	#

	if stored_config[cm.ConfigParams.VALIDATION_PARAMS] is None:
		raise ValueError("validate_config. Validation params is None")
	elif stored_config[cm.ConfigParams.AGGREGATION_PARAMS] is None:
		raise ValueError("validate_config. Aggregation params is None")
	
	validate_validation_and_aggregation_algorithms_params(stored_config[cm.ConfigParams.VALIDATION_PARAMS], stored_config[cm.ConfigParams.AGGREGATION_PARAMS])
	
	#
	# Consensus params
	#

	if stored_config[cm.ConfigParams.CONSENSUS_PARAMS] is None:
		raise ValueError("validate_config. Consensus params is None")
	elif type(stored_config[cm.ConfigParams.CONSENSUS_PARAMS]) != dict:
		raise TypeError("validate_config. Consensus params is not a dictionary")
	
	for key in cm.ConfigGenericConsensusAlgorithmParams.list():
		if key not in stored_config[cm.ConfigParams.CONSENSUS_PARAMS]:
			raise KeyError(f"validate_config. Consensus Key {key} not available in the stored configuration")
		
	consensus_params = stored_config[cm.ConfigParams.CONSENSUS_PARAMS]

	# Consensus algorithm
	if consensus_params[cm.ConfigGenericConsensusAlgorithmParams.TYPE] is None:
		raise ValueError("validate_config. Consensus algorithm is None")
	elif consensus_params[cm.ConfigGenericConsensusAlgorithmParams.TYPE] not in cm.ConsensusAlgorithmType.list():
		raise ValueError("validate_config. Consensus algorithm is not a valid value")
	
	del consensus_params

	return run_time_only_params

def start_main(config_file_path: str) -> None:
	'''Start the main machine'''

	try:
		stored_config = load_config_from_file(config_file_path)
		run_time_only_params = validate_config(stored_config)
	except Exception as e:
		print(f"Exception while loading config file. Exception: {e}")
		sys.exit(1)

	logger = diagnostic.Diagnostic(stored_config[cm.ConfigParams.LOGGER_PATH], stored_config[cm.ConfigParams.LOGGER_LEVEL], logger_name= "Simulator")
	
	print("Initializing the simulation...")
	logger.record("Initializing the simulation...", diagnostic.INFO, MAIN_IDENTIFIER)

	# Start handling the ram usage	
	pu.ram_usage(stored_config[cm.ConfigParams.RAM_USAGE_LOG_PATH], True)

	print("Creating the dataset...")
	logger.record("Creating the dataset...", diagnostic.DEBUG, MAIN_IDENTIFIER)

	if os.path.isdir(stored_config[cm.ConfigParams.DATASET_PARAMS][cm.ConfigDatasetParams.DATASET_FILES_DIR_PATH]) is False:
		# Creation of the various quanta and test set - quantization of the global dataset
		quanta, testset, valset = quantize_dataset(
			dataset_path = stored_config[cm.ConfigParams.DATASET_PARAMS][cm.ConfigDatasetParams.DATASET],
			num_quant = stored_config[cm.ConfigParams.DATASET_PARAMS][cm.ConfigDatasetParams.NUM_OF_QUANTA],
			temperature = stored_config[cm.ConfigParams.DATASET_PARAMS][cm.ConfigDatasetParams.TEMPERATURE],
			iid_percentage = stored_config[cm.ConfigParams.DATASET_PARAMS][cm.ConfigDatasetParams.PERC_OF_IID_QUANTA],
		)

		quanta_paths, testset_path, valset_path = save_datasets(quanta, testset, valset, stored_config[cm.ConfigParams.DATASET_PARAMS][cm.ConfigDatasetParams.DATASET_FILES_DIR_PATH])

		del quanta
		del testset

	# Get list of paths of files in the dataset directory
	else:
		quanta_paths = [os.path.join(stored_config[cm.ConfigParams.DATASET_PARAMS][cm.ConfigDatasetParams.DATASET_FILES_DIR_PATH], f) for f in os.listdir(stored_config[cm.ConfigParams.DATASET_PARAMS][cm.ConfigDatasetParams.DATASET_FILES_DIR_PATH]) if os.path.isfile(os.path.join(stored_config[cm.ConfigParams.DATASET_PARAMS][cm.ConfigDatasetParams.DATASET_FILES_DIR_PATH], f)) and f.startswith("trainset_")]
		testset_path = os.path.join(stored_config[cm.ConfigParams.DATASET_PARAMS][cm.ConfigDatasetParams.DATASET_FILES_DIR_PATH], "testset.npz")
		valset_path = os.path.join(stored_config[cm.ConfigParams.DATASET_PARAMS][cm.ConfigDatasetParams.DATASET_FILES_DIR_PATH], "valset.npz")

	gc.collect()

	try:
		if stored_config[cm.ConfigParams.CONSENSUS_PARAMS][cm.ConfigGenericConsensusAlgorithmParams.TYPE] == cm.ConsensusAlgorithmType.COMMITTEE:
			start_committee_simulation_main(logger, quanta_paths, testset_path, valset_path, config_file_path, stored_config, run_time_only_params)
		elif stored_config[cm.ConfigParams.CONSENSUS_PARAMS][cm.ConfigGenericConsensusAlgorithmParams.TYPE] == cm.ConsensusAlgorithmType.POS:
			start_pos_simulation_main(logger, quanta_paths, testset_path, valset_path, config_file_path, stored_config, run_time_only_params)
		elif stored_config[cm.ConfigParams.CONSENSUS_PARAMS][cm.ConfigGenericConsensusAlgorithmParams.TYPE] == cm.ConsensusAlgorithmType.POW:
			start_pow_simulation_main(logger, quanta_paths, testset_path, valset_path, config_file_path, stored_config, run_time_only_params)
		else:
			raise ValueError("start_main method. Invalid simulation type")
	except Exception as e:
		print(f"Exception while running the simulation. Exception: {e}")
		logger.record(f"Exception while running the simulation. Exception: {e}", diagnostic.CRITICAL, MAIN_IDENTIFIER)
		sys.exit(1)
	else:
		logger.record("Simulation ended", diagnostic.INFO, MAIN_IDENTIFIER)
		print("Simulation ended")
	finally:
		logger.shutdown()

if __name__ == "__main__":

	set_start_method("spawn", force=True)

	setproctitle.setproctitle("FedBlockSimulator - Main")

	if len(sys.argv) != 2:
		print("Usage: python main.py <config_file_path>")
		sys.exit(1)
		
	config_file_path = sys.argv[1]
	
	if os.path.exists(config_file_path) is False:
		print("Config file not found")
		sys.exit(1)

	start_main(config_file_path)
	sys.exit(0)
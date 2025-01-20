class AbstractEnum:
	'''Abstract Enum Class to be inherited by all Enum Classes'''

	def __init__(self) -> None:
		raise Exception("Enum Classes can't be concrete!")

	@classmethod
	def list(cls):
		return [
            getattr(cls, key) for key in dir(cls)
            if not key.startswith('__')  # Exclude special attributes
            and not callable(getattr(cls, key))  # Exclude methods (callable objects)
            and not isinstance(getattr(cls, key), (classmethod, staticmethod))  # Exclude class/static methods
        ]

class ConfigBaseParams(AbstractEnum):
	'''Enum for Base Parameters in Config File'''

	IS_MAIN_SIMULATION = "is_main_simulation"
	LOGGER_PATH = "logger_path"
	LOGGER_LEVEL = "logger_level"
	RAM_USAGE_LOG_PATH = "ram_usage_log_path"
	MODEL_ARCHITECTURE_PATH = "model_architecture_path"
	STARTING_WEIGHTS_PATH = "starting_weights_path"
	STARTING_OPTIMIZER_STATE_PATH = "starting_optimizer_state_path"
	NUM_OF_PROCESSES_TO_USE_TO_MANAGE_NODES = "num_of_processes_to_use_to_manage_nodes"
	MAXIMUM_NUMBER_OF_PARALLEL_MALICIOUS_TRAININGS = "max_number_of_parallel_malicious_trainings"
	MAXIMUM_NUMBER_OF_PARALLEL_HONEST_TRAININGS = "max_number_of_parallel_honest_trainings"
	MAXIMUM_NUMBER_OF_PARALLEL_VALIDATIONS = "max_number_of_parallel_validations"
	MAX_NUM_OF_ROUNDS = "max_num_of_rounds"
	FIT_EPOCHS = "fit_epochs"
	BATCH_SIZE = "batch_size"
	VALIDATION_WITH_TEST_SET_AFTER_MODEL_AGGREGATION = "validation_with_test_set_after_model_aggregation"
	STORE_WEIGHTS_DIRECTLY_IN_ARCHIVE_TMP_DIR = "store_weights_directly_in_archive_tmp_dir"
	NEED_TO_JOIN_AN_EXISTING_NETWORK = "need_to_join_an_existing_network"
	ENTRY_POINT_NODES = "entry_point_nodes"

class ConfigSubCategories(AbstractEnum):
	# Sub-categories
	NODES_PARAMS = "nodes_params"
	CONSENSUS_PARAMS = "consensus_algorithm_params"
	VALIDATION_PARAMS = "validation_algorithm_params"
	AGGREGATION_PARAMS = "aggregation_algorithm_params"
	MALICIOUS_NODES_PARAMS = "malicious_nodes_params"
	DATASET_PARAMS = "dataset_params"
	ARCHIVE_PARAMS = "archive_params"

class ConfigParams(ConfigBaseParams, ConfigSubCategories):
	'''Enum for Required Parameters in Config File'''
	pass

class ConfigEntryPointNodesParams(AbstractEnum):
	'''Enum for Entry Point Nodes Parameters in Config File'''
	HOST = "host"
	PORT = "port"

class ConfigNodesParams(AbstractEnum):
	'''Enum for Nodes Parameters in Config File'''
	OVERALL_NUM_OF_NODES = "overall_num_of_nodes"
	FIRST_NODE_ID = "first_node_id"
	HOST = "host"
	FIRST_PORT = "first_port"
	LIST_OF_NODES_ALLOWED_TO_PRODUCE_DEBUG_LOG_MESSAGES = "list_of_node_ids_allowed_to_produce_debug_log_messages"

class ConfigGenericConsensusAlgorithmParams(AbstractEnum):
	'''Enum for Consensus Algorithm Parameters in Config File'''
	TYPE = "type"

class ConfigGenericValidationAlgorithmParams(AbstractEnum):
	'''Enum for Validation Algorithm Parameters in Config File'''
	TYPE = "type"

class ConfigGenericAggregationAlgorithmParams(AbstractEnum):
	'''Enum for Aggregation Algorithm Parameters in Config File'''
	TYPE = "type"

class ConfigMaliciousNodesParams(AbstractEnum):
	'''Enum for Malicious Nodes Parameters in Config File'''
	NUM_OF_MALICIOUS_NODES = "num_of_malicious_nodes"
	LIST_OF_COLLUSION_PEER_IDS = "list_of_collusion_peer_ids"
	NODE_BEHAVIOURS = "node_behaviours"

class ConfigNodeBehaviorParams(AbstractEnum):
	'''Enum for Node Behavior Parameters in Config File'''
	TYPE = "type"
	NODES_ID = "nodes_id"
	SELECTED_CLASSES = "selected_classes"
	TARGET_CLASSES = "target_classes"
	TARGET_CLASS = "target_class"
	SQUARE_SIZE = "square_size"
	SIGMA = "sigma"
	NUM_OF_SAMPLES = "num_of_samples"
	STARTING_ROUND_FOR_MALICIOUS_BEHAVIOUR = "starting_round_for_malicious_behaviour"

class ConfigArchiveParams(AbstractEnum):
	'''Enum for Archive Parameters in Config File'''
	ARCHIVE_MUST_BE_CREATED = "archive_must_be_created"
	HOST = "host"
	PORT = "port"
	TMP_DIR = "tmp_dir"
	PERSISTENT_MODE = "persistent_mode"
	LOGGER_PATH = "logger_path"
	LOGGER_LEVEL = "logger_level"

class ConfigDatasetParams(AbstractEnum):
	'''Enum for Dataset Parameters in Config File'''
	DATASET = "dataset"
	TEMPERATURE = "temperature"
	NUM_OF_QUANTA = "num_of_quanta"
	PERC_OF_IID_QUANTA = "perc_of_iid_quanta"
	LAZY_LOADING = "lazy_loading"
	NODES_COMPOSITE_DATASETS = "nodes_composite_datasets"
	DATASET_FILES_DIR_PATH = "dataset_files_dir_path"

class ConfigNodesCompositeDatasetsParams(AbstractEnum):
	'''Enum for Nodes Composite Datasets Parameters in Config File'''
	ALIAS = "alias"
	NUM_OF_NODES = "num_of_nodes"
	NUM_OF_QUANTA_TO_USE = "num_of_quanta_to_use"
	IID_QUANTA_TO_USE = "iid_quanta_to_use"

class ConsensusAlgorithmType(AbstractEnum):
	'''Enum for Consensus Algorithm Types'''
	COMMITTEE = "committee"
	POW = "pow"
	POS = "pos"

class WeightsBasedValidationAlgorithmType(AbstractEnum):
	'''Enum for Validation Algorithm Types based on Weights'''
	PASS_WEIGHTS = "pass_weights"
	LOCAL_DATASET_VALIDATION = "local_dataset_validation"
	GLOBAL_DATASET_VALIDATION = "global_dataset_validation"

class GradientsBasedValidationAlgorithmType(AbstractEnum):
	'''Enum for Validation Algorithm Types based on Gradients'''
	PASS_GRADIENTS = "pass_gradients"
	KRUM = "krum"
	TRIMMED_MEAN = "trimmed_mean"

class ValidationAlgorithmType(WeightsBasedValidationAlgorithmType, GradientsBasedValidationAlgorithmType):
	'''Enum for Validation Algorithm Types'''
	pass

class WeightsBasedAggregationAlgorithmType(AbstractEnum):
	'''Enum for Aggregation Algorithm Types based on Weights'''
	FEDAVG = "fedavg"
	MEAN = "mean"
	MEDIAN = "median"
	TRIMMED_MEAN = "trimmed_mean"

class GradientsBasedAggregationAlgorithmType(AbstractEnum):
	'''Enum for Aggregation Algorithm Types based on Gradients'''
	MEAN = "mean"
	MEDIAN = "median"
	TRIMMED_MEAN = "trimmed_mean"

class AggregationAlgorithmType(WeightsBasedAggregationAlgorithmType, GradientsBasedAggregationAlgorithmType):
	'''Enum for Aggregation Algorithm Types'''
	pass

class MaliciousNodeBehaviourType(AbstractEnum):
	'''Enum for Node Behaviour Types'''
	LABEL_FLIPPING = "label_flipping"
	TARGETED_POISONING = "targeted_poisoning"
	RANDOM_LABEL = "random_label"
	ADDITIVE_NOISE = "additive_noise"
	RANDOM_NOISE = "random_noise"

class AvailableDataset(AbstractEnum):
	'''Enum for Available Datasets'''
	CIFAR10 = "cifar10"
	CIFAR100 = "cifar100"
	MNIST = "mnist"

class RunTimeOnlyParams(AbstractEnum):
	'''Enum for Run-time Only Parameters. These parameters are not stored in the config file, but are derived from the config file'''
	MODEL_ARCHITECTURE = "model_architecture"
	STARTING_WEIGHTS = "starting_weights"
	STARTING_OPTIMIZER_STATE = "starting_optimizer_state"
	NODES_COMPOSITE_DATASETS = "nodes_composite_datasets"
	NODES = "nodes"

class RunTimeNodeParams(AbstractEnum):
	'''Enum for Run-time Node Parameters. These parameters are not stored in the config file, but are derived from the config file'''
	NODE_ID = "node_id"
	HOST = "host"
	PORT = "port"
	ALLOWED_TO_PRODUCE_DEBUG_LOG_MESSAGES = "allowed_to_produce_debug_log_messages"

class RunTimeNodeCompositeDatasetParams(AbstractEnum):
	'''Enum for Run-time Node Composite Dataset Parameters. These parameters are not stored in the config file, but are derived from the config file'''
	ALIAS = "alias"
	NUM_OF_NODES = "num_of_nodes"
	NUM_OF_QUANTA_TO_USE = "num_of_quanta_to_use"
	IID_QUANTA_TO_USE = "iid_quanta_to_use"
	RELATED_NODES_IDS = "related_nodes_ids"

class NodeSpecificConfigParams(AbstractEnum):
	NODE_ID = "node_id"
	HOST = "host"
	PORT = "port"
	NUM_OF_QUANTA_TO_USE = "num_of_quanta_to_use"
	NUM_OF_IID_QUANTA_TO_USE = "num_of_iid_quanta_to_use"
	ACTIVE_TRAINER_IN_FIRST_ROUND = "is_active_trainer_in_first_round"
	ALLOWED_TO_PRODUCE_DEBUG_LOG_MESSAGES = "allowed_to_produce_debug_log_messages"
	PAUSE_RESUME_EVENT = "pause_resume_event"
	MALICIOUS = "malicious"

class AttackerSpecificConfigParams(NodeSpecificConfigParams):
	TYPE = "type"
	SELECTED_CLASSES = "selected_classes"
	TARGET_CLASSES = "target_classes"
	TARGET_CLASS = "target_class"
	SQUARE_SIZE = "square_size"
	SIGMA = "sigma"
	NUM_OF_SAMPLES = "num_of_samples"
	COLLUSION_PEERS = "collusion_peers"
	STARTING_ROUND_FOR_MALICIOUS_BEHAVIOUR = "starting_round_for_malicious_behaviour"

class DistanceFunctionType(AbstractEnum):
	EUCLIDEAN = "euclidean" # minkowski with p=2
	CITYBLOCK = "cityblock" # minkowski with p=1
	SEUCLIDEAN = "seuclidean" # compute the standardized Euclidean distance using the variance of each coordinate
	COSINE = "cosine" # compute the cosine distance
	CORRELATION = "correlation" 
	CHEBYSHEV = "chebyshev" # compute the Chebyshev distance which is the maximum absolute difference between any coordinate (minkowski with p=inf)
	CANBERRA = "canberra" # compute the Canberra distance between two n-vectors u and v
	BRAYCURTIS = "braycurtis" # compute the Bray-Curtis distance between two n-vectors u and v
	MAHALANOBIS = "mahalanobis" # compute the Mahalanobis distance between two n-vectors u and v
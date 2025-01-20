from .enums.common import ConfigParams, ConfigArchiveParams, ConfigDatasetParams, ConfigNodesParams, ConfigGenericConsensusAlgorithmParams, ConfigMaliciousNodesParams, ConfigGenericValidationAlgorithmParams, ConfigGenericAggregationAlgorithmParams
from . import diagnostic

SEED = 42

TESTING_OPS = False

PREPROCESS_MOBILENET = False

PREPROCESS_EFFICIENTNETB1 = False

VERBOSE = 0

PLOT_GRAPHS = False

DELETE_TMP_DIR = False

FILE_WHERE_TO_STORE_FINAL_BLOCKCHAIN = "./blockchain.json"

DIRECTORY_WHERE_TO_STORE_TMP_FILES = "/tmp"

DEFAULT_CONFIG = {
    ConfigParams.IS_MAIN_SIMULATION: True,
	ConfigParams.LOGGER_PATH: "./logger.log",
	ConfigParams.LOGGER_LEVEL: diagnostic.INFO,
	ConfigParams.RAM_USAGE_LOG_PATH: "./ram_usage.txt",
	ConfigParams.MODEL_ARCHITECTURE_PATH: None,
	ConfigParams.STARTING_WEIGHTS_PATH: None,
	ConfigParams.STARTING_OPTIMIZER_STATE_PATH: None,
    ConfigParams.NUM_OF_PROCESSES_TO_USE_TO_MANAGE_NODES: 1,
	ConfigParams.MAXIMUM_NUMBER_OF_PARALLEL_MALICIOUS_TRAININGS: 1,
	ConfigParams.MAXIMUM_NUMBER_OF_PARALLEL_HONEST_TRAININGS: 1,
	ConfigParams.MAXIMUM_NUMBER_OF_PARALLEL_VALIDATIONS: 1,
	ConfigParams.MAX_NUM_OF_ROUNDS: 10,
	ConfigParams.FIT_EPOCHS: 10,
	ConfigParams.BATCH_SIZE: 32,
	ConfigParams.NEED_TO_JOIN_AN_EXISTING_NETWORK: False,
	ConfigParams.ENTRY_POINT_NODES: [],
	ConfigParams.VALIDATION_WITH_TEST_SET_AFTER_MODEL_AGGREGATION: False,
    ConfigParams.STORE_WEIGHTS_DIRECTLY_IN_ARCHIVE_TMP_DIR: False,
    
	ConfigParams.NODES_PARAMS: {
		ConfigNodesParams.OVERALL_NUM_OF_NODES: 0,
		ConfigNodesParams.FIRST_NODE_ID: 0,
		ConfigNodesParams.HOST: "localhost",
		ConfigNodesParams.FIRST_PORT: 8000,
		ConfigNodesParams.LIST_OF_NODES_ALLOWED_TO_PRODUCE_DEBUG_LOG_MESSAGES: []
	},
	ConfigParams.ARCHIVE_PARAMS: {
		ConfigArchiveParams.ARCHIVE_MUST_BE_CREATED: True,
		ConfigArchiveParams.HOST: "localhost",
		ConfigArchiveParams.PORT: 8001,
		ConfigArchiveParams.TMP_DIR: "./tmp",
		ConfigArchiveParams.PERSISTENT_MODE: False,
		ConfigArchiveParams.LOGGER_PATH: "./logger.log",
		ConfigArchiveParams.LOGGER_LEVEL: diagnostic.INFO
	},
	ConfigParams.DATASET_PARAMS: {
		ConfigDatasetParams.DATASET: "cifar10",
		ConfigDatasetParams.TEMPERATURE: 0.3,
		ConfigDatasetParams.NUM_OF_QUANTA: 20,
		ConfigDatasetParams.PERC_OF_IID_QUANTA: 0.5,
        ConfigDatasetParams.LAZY_LOADING: False,
		ConfigDatasetParams.NODES_COMPOSITE_DATASETS: [],
        ConfigDatasetParams.DATASET_FILES_DIR_PATH: "./tmp/datasets"
	},
    ConfigParams.CONSENSUS_PARAMS: {
        ConfigGenericConsensusAlgorithmParams.TYPE: None
	},
    ConfigParams.VALIDATION_PARAMS: {
		ConfigGenericValidationAlgorithmParams.TYPE: None
	},
	ConfigParams.AGGREGATION_PARAMS: {
		ConfigGenericAggregationAlgorithmParams.TYPE: None
	},
	ConfigParams.MALICIOUS_NODES_PARAMS: {
		ConfigMaliciousNodesParams.NUM_OF_MALICIOUS_NODES: 0,
		ConfigMaliciousNodesParams.NODE_BEHAVIOURS: [],
		ConfigMaliciousNodesParams.LIST_OF_COLLUSION_PEER_IDS: []
	}
}
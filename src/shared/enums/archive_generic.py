from .common import AbstractEnum

class FileStored(AbstractEnum):
    TYPE = "type"
    ROUND = "round"

class FileStoredType(AbstractEnum):
    UPDATE = "update"
    MODEL = "model"

class FileWrittenByNode(AbstractEnum):
    FILE_NAME = "file_name"
    TIMESTAMP_REQUEST = "timestamp_request"

class InvalidRoundNumber(Exception):
    pass

class PeerUpdate(AbstractEnum):
    PEER_ID = "peer_id"
    NAME = "update_name"
    WEIGHTS = "update_weights"
    NUM_SAMPLES = "num_samples"

class AggregatedModel(AbstractEnum):
    NAME = "model_name"
    WEIGHTS = "model_weights"
    OPTIMIZER = "model_optimizer"
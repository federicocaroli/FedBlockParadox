import json

from ..shared.enums.common import AbstractEnum
from ..shared.block import Block, BlockType

class ModelBlockInvolvedTrainerFields(AbstractEnum):
    NODE_ID = "node_id"
    UPDATE_NAME = "update_name"
    UPDATE_SCORE = "update_score"

class ModelBlock(Block):
    def __init__(self, previous_hash: str, timestamp: float, global_model_name: str, aggregation_round: int, involved_trainers: list, list_of_current_validators: list, available_nodes: list, block_hash: (str | None) = None):

        if type(previous_hash) != str or type(timestamp) not in [float, int] or type(global_model_name) != str or type(aggregation_round) != int or type(involved_trainers) != list or type(list_of_current_validators) != list or type(available_nodes) != list or (block_hash is not None and type(block_hash) != str):
            raise TypeError("ModelBlock")
        elif any(key not in elem for key in ModelBlockInvolvedTrainerFields.list() for elem in involved_trainers):
            raise ValueError("ModelBlock. Involved trainers")
        
        for elem in involved_trainers:
            if type(elem[ModelBlockInvolvedTrainerFields.NODE_ID]) != int or type(elem[ModelBlockInvolvedTrainerFields.UPDATE_NAME]) != str or type(elem[ModelBlockInvolvedTrainerFields.UPDATE_SCORE]) not in [int, float]:
                raise TypeError(f"ModelBlock. Involved trainers elem: {elem}")
        
        # block header
        self._block_type = BlockType.MODEL
        self._previous_hash = previous_hash
        self._timestamp = timestamp
        # body
        self._global_model_name = global_model_name
        self._aggregation_round = aggregation_round
        self._involved_trainers = involved_trainers
        self._list_of_current_validators = list_of_current_validators
        self._available_nodes = available_nodes

        if block_hash is None:
            self._block_hash = self.calculate_hash()
        else:
            self._block_hash = block_hash
    
    def get_block_hash(self):
        return self._block_hash
    
    def get_previous_hash(self):
        return self._previous_hash

    def get_aggregation_round(self):
        return self._aggregation_round
    
    def get_global_model_name(self):
        return self._global_model_name
    
    def get_block_type(self):
        return self._block_type

    def get_involved_trainers(self):
        return self._involved_trainers
    
    def get_list_of_current_validators(self):
        return self._list_of_current_validators
    
    def get_available_nodes(self):
        return self._available_nodes

    def to_json(self):
        return json.dumps({
            "previous_hash": self._previous_hash,
            "timestamp": self._timestamp,
            "global_model_name": self._global_model_name,
            "aggregation_round": self._aggregation_round,
            "involved_trainers": self._involved_trainers,
            "list_of_current_validators": self._list_of_current_validators,
            "available_nodes": self._available_nodes,
            "block_hash": self._block_hash
        })
    
    @classmethod
    def from_json(cls, json_string: str):
        if type(json_string) != str:
            raise TypeError("ModelBlock.from_json")
        
        data = json.loads(json_string)

        if any([elem not in data for elem in ["previous_hash", "timestamp", "global_model_name", "aggregation_round", "involved_trainers", "list_of_current_validators", "available_nodes", "block_hash"]]):
            raise ValueError("ModelBlock.from_json")

        return cls(data["previous_hash"], data["timestamp"], data["global_model_name"], data["aggregation_round"], data["involved_trainers"], data["list_of_current_validators"], data["available_nodes"], data["block_hash"])

class GenesisBlock(Block):

    def __init__(self, model_architecture: dict, model_starting_weights: list, model_starting_optimizer_state: dict | None, max_num_of_aggregation_rounds: int, num_of_validators: int, perc_of_nodes_active_in_a_round: float, threshold_to_pass_validation: int, list_of_first_validators: list, validation_algorithm_params: dict, aggregation_algorithm_params: dict, block_hash: (str | None) = None):
        
        if type(model_architecture) != dict or type(model_starting_weights) != list or (model_starting_optimizer_state is not None and type(model_starting_optimizer_state) != dict) or type(max_num_of_aggregation_rounds) != int or type(num_of_validators) != int or type(perc_of_nodes_active_in_a_round) not in [int, float] or type(threshold_to_pass_validation) != int or type(list_of_first_validators) != list or type(validation_algorithm_params) != dict or type(aggregation_algorithm_params) != dict or (block_hash is not None and type(block_hash) != str):
            raise TypeError("GenesisBlock")
        
        # block header
        self._block_type = BlockType.GENESIS
        self._previous_hash = None
        # body
        self._model_architecture = model_architecture
        self._model_starting_weights = model_starting_weights
        self._model_starting_optimizer_state = model_starting_optimizer_state
        self._perc_of_nodes_active_in_a_round = perc_of_nodes_active_in_a_round
        self._max_num_of_aggregation_rounds = max_num_of_aggregation_rounds
        self._validation_algorithm_params = validation_algorithm_params
        self._aggregation_algorithm_params = aggregation_algorithm_params
        self._num_of_validators = num_of_validators
        self._threshold_to_pass_validation = threshold_to_pass_validation
        self._list_of_first_validators = list_of_first_validators

        if block_hash is None:
            self._block_hash = self.calculate_hash()
        else:
            self._block_hash = block_hash
        
    def get_block_hash(self):
        return self._block_hash
        
    def get_previous_hash(self):
        return self._previous_hash
    
    def get_block_type(self):
        return self._block_type
    
    def get_model_architecture(self):
        return self._model_architecture
    
    def get_model_starting_weights(self):
        return self._model_starting_weights
    
    def get_model_starting_optimizer_state(self):
        return self._model_starting_optimizer_state
    
    def get_validation_algorithm_params(self):
        return self._validation_algorithm_params
    
    def get_aggregation_algorithm_params(self):
        return self._aggregation_algorithm_params
    
    def get_threshold_to_pass_validation(self):
        return self._threshold_to_pass_validation
    
    def get_perc_of_nodes_active_in_a_round(self):
        return self._perc_of_nodes_active_in_a_round

    def get_max_num_of_aggregation_rounds(self):
        return self._max_num_of_aggregation_rounds
    
    def get_num_of_validators(self):
        return self._num_of_validators
    
    def get_list_of_first_validators(self):
        return self._list_of_first_validators

    def to_json(self):
        return json.dumps({
            "model_architecture": self._model_architecture,
            "model_starting_weights": self._model_starting_weights,
            "model_starting_optimizer_state": self._model_starting_optimizer_state,
            "max_num_of_aggregation_rounds": self._max_num_of_aggregation_rounds,
            "number_of_validators": self._num_of_validators,
            "threshold_to_pass_validation": self._threshold_to_pass_validation,
            "list_of_first_validators": self._list_of_first_validators,
            "perc_of_nodes_active_in_a_round": self._perc_of_nodes_active_in_a_round,
            "validation_algorithm_params": self._validation_algorithm_params,
            "aggregation_algorithm_params": self._aggregation_algorithm_params,
            "block_hash": self._block_hash
        })
    
    @classmethod
    def from_json(cls, json_string: str):
        if type(json_string) != str:
            raise TypeError("GenesisBlock.from_json")
        
        data = json.loads(json_string)

        if any([elem not in data for elem in ["model_architecture", "model_starting_weights", "model_starting_optimizer_state", "max_num_of_aggregation_rounds", "number_of_validators", "threshold_to_pass_validation", "list_of_first_validators", "perc_of_nodes_active_in_a_round", "validation_algorithm_params", "aggregation_algorithm_params", "block_hash"]]):
            raise ValueError("GenesisBlock.from_json")
        
        return cls(data["model_architecture"], data["model_starting_weights"], data["model_starting_optimizer_state"], data["max_num_of_aggregation_rounds"], data["number_of_validators"], data["perc_of_nodes_active_in_a_round"], data["threshold_to_pass_validation"], data["list_of_first_validators"], data["validation_algorithm_params"], data["aggregation_algorithm_params"], data["block_hash"])


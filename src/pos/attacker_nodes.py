from multiprocessing.synchronize import BoundedSemaphore as BoundedSemaphoreClass

from .node import PosBasedNode

from ..shared import attacker_nodes as shared_an, diagnostic
from ..shared.constants import SEED

class PosBasedLabelFlippingNode(PosBasedNode):
	def __init__(self, logger: diagnostic.Diagnostic, node_id: int, host: str, port: int, dataset_quanta_paths: list, archive_host: str, archive_port: int, lazy_loading: bool, test_set_path: str, val_set_path: str, malicious_training_semaphore: BoundedSemaphoreClass, honest_training_semaphore: BoundedSemaphoreClass, validation_semaphore: BoundedSemaphoreClass, fit_epochs: int = 3, batch_size: int = 32, already_available_peers: dict = {}, allowed_to_write_redudant_log_messages: bool = False, test_set_validation_of_global_model_after_each_round: bool = False, genesis_block = None, is_active_trainer_in_first_round: (bool | None) = None, store_weights_directly_in_archive_tmp_dir: bool = False, archive_tmp_dir: (str | None) = None, starting_round_for_malicious_behaviour: int = 1, selected_classes: list=[], target_classes: list=[], num_of_samples: int = 0):
		if type(starting_round_for_malicious_behaviour) != int:
			raise TypeError("PosBasedLabelFlippingNode constructor")
		elif starting_round_for_malicious_behaviour < 1:
			raise ValueError("PosBasedLabelFlippingNode constructor")

		# Initialization of specific variables of the class
		self._starting_round_for_malicious_behaviour = starting_round_for_malicious_behaviour

		# Initialization of pos based node (it may happen that variables of generic node are overwritten by pos based node)
		# prepare_node_for_first_round is False because we want to delay the start of the node
		super().__init__(logger, node_id, host, port, dataset_quanta_paths, archive_host, archive_port, lazy_loading, test_set_path, val_set_path, malicious_training_semaphore, honest_training_semaphore, validation_semaphore, fit_epochs, batch_size, already_available_peers, allowed_to_write_redudant_log_messages, test_set_validation_of_global_model_after_each_round, genesis_block, is_active_trainer_in_first_round, store_weights_directly_in_archive_tmp_dir, archive_tmp_dir, False)

		# Based initialization of generic node and label flipping node
		self._label_flipping_node = shared_an.LabelFlippingNode(self, selected_classes, target_classes, num_of_samples)

		# Finalize the preparation of the node for the first round
		self.prepare_node_for_the_first_round(genesis_block)

	def _define_nodes_active_in_the_next_round(self, seed: int = SEED, base_list_of_nodes: (list | None) = None):
		if self.aggregation_round() < self._starting_round_for_malicious_behaviour:
			return super()._define_nodes_active_in_the_next_round(seed, base_list_of_nodes)
		
		# If the node can behave maliciously
		return self._label_flipping_node._define_nodes_active_in_the_next_round()
		
	def _weight_fit(self):
		if self.aggregation_round() < self._starting_round_for_malicious_behaviour:
			return super()._weight_fit()
		
		return self._label_flipping_node._weight_fit()
	
	def _gradient_fit(self):
		if self.aggregation_round() < self._starting_round_for_malicious_behaviour:
			return super()._gradient_fit()
		
		return self._label_flipping_node._gradient_fit()
		
class PosBasedTargetedPoisoningNode(PosBasedNode):
	def __init__(self, logger: diagnostic.Diagnostic, node_id: int, host: str, port: int, dataset_quanta_paths: list, archive_host: str, archive_port: int, lazy_loading: bool, test_set_path: str, val_set_path: str, malicious_training_semaphore: BoundedSemaphoreClass, honest_training_semaphore: BoundedSemaphoreClass, validation_semaphore: BoundedSemaphoreClass, fit_epochs: int = 3, batch_size: int = 32, already_available_peers: dict = {}, allowed_to_write_redudant_log_messages: bool = False, test_set_validation_of_global_model_after_each_round: bool = False, genesis_block = None, is_active_trainer_in_first_round: (bool | None) = None, store_weights_directly_in_archive_tmp_dir: bool = False, archive_tmp_dir: (str | None) = None, starting_round_for_malicious_behaviour: int = 1, target_class: (int | None) = None, size: int = 0, num_of_samples: int = 0):
		if type(starting_round_for_malicious_behaviour) != int:
			raise TypeError("PosBasedTargetedPoisoningNode constructor")
		elif starting_round_for_malicious_behaviour < 1:
			raise ValueError("PosBasedTargetedPoisoningNode constructor")

		# Initialization of specific variables of the class
		self._starting_round_for_malicious_behaviour = starting_round_for_malicious_behaviour

		# Initialization of pos based node (it may happen that variables of generic node are overwritten by pos based node)
		# prepare_node_for_first_round is False because we want to delay the start of the node
		super().__init__(logger, node_id, host, port, dataset_quanta_paths, archive_host, archive_port, lazy_loading, test_set_path, val_set_path, malicious_training_semaphore, honest_training_semaphore, validation_semaphore, fit_epochs, batch_size, already_available_peers, allowed_to_write_redudant_log_messages, test_set_validation_of_global_model_after_each_round, genesis_block, is_active_trainer_in_first_round, store_weights_directly_in_archive_tmp_dir, archive_tmp_dir, False)

		# Based initialization of generic node and targeted poisoning node
		self._targeted_poisoning_node = shared_an.TargetedPoisoningNode(self, target_class, size, num_of_samples)

		# Finalize the preparation of the node for the first round
		self.prepare_node_for_the_first_round(genesis_block)

	def _define_nodes_active_in_the_next_round(self, seed: int = SEED, base_list_of_nodes: (list | None) = None):
		if self.aggregation_round() < self._starting_round_for_malicious_behaviour:
			return super()._define_nodes_active_in_the_next_round(seed, base_list_of_nodes)
		
		# If the node can behave maliciously
		return self._targeted_poisoning_node._define_nodes_active_in_the_next_round()
		
	def _weight_fit(self):
		if self.aggregation_round() < self._starting_round_for_malicious_behaviour:
			return super()._weight_fit()
		
		return self._targeted_poisoning_node._weight_fit()
	
	def _gradient_fit(self):
		if self.aggregation_round() < self._starting_round_for_malicious_behaviour:
			return super()._gradient_fit()
		
		return self._targeted_poisoning_node._gradient_fit()

class PosBasedRandomLabelByzantineNode(PosBasedNode):
	def __init__(self, logger: diagnostic.Diagnostic, node_id: int, host: str, port: int, dataset_quanta_paths: list, archive_host: str, archive_port: int, lazy_loading: bool, test_set_path: str, val_set_path: str, malicious_training_semaphore: BoundedSemaphoreClass, honest_training_semaphore: BoundedSemaphoreClass, validation_semaphore: BoundedSemaphoreClass, fit_epochs: int = 3, batch_size: int = 32, already_available_peers: dict = {}, allowed_to_write_redudant_log_messages: bool = False, test_set_validation_of_global_model_after_each_round: bool = False, genesis_block = None, is_active_trainer_in_first_round: (bool | None) = None, store_weights_directly_in_archive_tmp_dir: bool = False, archive_tmp_dir: (str | None) = None, starting_round_for_malicious_behaviour: int = 1, num_of_samples: int = 0):
		if type(starting_round_for_malicious_behaviour) != int:
			raise TypeError("PosBasedRandomLabelByzantineNode constructor")
		elif starting_round_for_malicious_behaviour < 1:
			raise ValueError("PosBasedRandomLabelByzantineNode constructor")

		# Initialization of specific variables of the class
		self._starting_round_for_malicious_behaviour = starting_round_for_malicious_behaviour

		# Initialization of pos based node (it may happen that variables of generic node are overwritten by pos based node)
		# prepare_node_for_first_round is False because we want to delay the start of the node
		super().__init__(logger, node_id, host, port, dataset_quanta_paths, archive_host, archive_port, lazy_loading, test_set_path, val_set_path, malicious_training_semaphore, honest_training_semaphore, validation_semaphore, fit_epochs, batch_size, already_available_peers, allowed_to_write_redudant_log_messages, test_set_validation_of_global_model_after_each_round, genesis_block, is_active_trainer_in_first_round, store_weights_directly_in_archive_tmp_dir, archive_tmp_dir, False)

		# Based initialization of generic node and random label byzantine node
		self._random_label_byzantine_node = shared_an.RandomLabelByzantineNode(self, num_of_samples)

		# Finalize the preparation of the node for the first round
		self.prepare_node_for_the_first_round(genesis_block)

	def _define_nodes_active_in_the_next_round(self, seed: int = SEED, base_list_of_nodes: (list | None) = None):
		if self.aggregation_round() < self._starting_round_for_malicious_behaviour:
			return super()._define_nodes_active_in_the_next_round(seed, base_list_of_nodes)
		
		# If the node can behave maliciously
		return self._random_label_byzantine_node._define_nodes_active_in_the_next_round()
	
	def _weight_fit(self):
		if self.aggregation_round() < self._starting_round_for_malicious_behaviour:
			return super()._weight_fit()
		
		return self._random_label_byzantine_node._weight_fit()
	
	def _gradient_fit(self):
		if self.aggregation_round() < self._starting_round_for_malicious_behaviour:
			return super()._gradient_fit()
		
		return self._random_label_byzantine_node._gradient_fit()
		
class PosBasedAdditiveNoiseByzantineNode(PosBasedNode):
	def __init__(self, logger: diagnostic.Diagnostic, node_id: int, host: str, port: int, dataset_quanta_paths: list, archive_host: str, archive_port: int, lazy_loading: bool, test_set_path: str, val_set_path: str, malicious_training_semaphore: BoundedSemaphoreClass, honest_training_semaphore: BoundedSemaphoreClass, validation_semaphore: BoundedSemaphoreClass, fit_epochs: int = 3, batch_size: int = 32, already_available_peers: dict = {}, allowed_to_write_redudant_log_messages: bool = False, test_set_validation_of_global_model_after_each_round: bool = False, genesis_block = None, is_active_trainer_in_first_round: (bool | None) = None, store_weights_directly_in_archive_tmp_dir: bool = False, archive_tmp_dir: (str | None) = None, starting_round_for_malicious_behaviour: int = 1, sigma: float = 0.1, num_of_samples: int = 0):
		if type(starting_round_for_malicious_behaviour) != int:
			raise TypeError("PosBasedAdditiveNoiseByzantineNode constructor")
		elif starting_round_for_malicious_behaviour < 1:
			raise ValueError("PosBasedAdditiveNoiseByzantineNode constructor")

		# Initialization of specific variables of the class
		self._starting_round_for_malicious_behaviour = starting_round_for_malicious_behaviour

		# Initialization of pos based node (it may happen that variables of generic node are overwritten by pos based node)
		# prepare_node_for_first_round is False because we want to delay the start of the node
		super().__init__(logger, node_id, host, port, dataset_quanta_paths, archive_host, archive_port, lazy_loading, test_set_path, val_set_path, malicious_training_semaphore, honest_training_semaphore, validation_semaphore, fit_epochs, batch_size, already_available_peers, allowed_to_write_redudant_log_messages, test_set_validation_of_global_model_after_each_round, genesis_block, is_active_trainer_in_first_round, store_weights_directly_in_archive_tmp_dir, archive_tmp_dir, False)

		# Based initialization of generic node and additive noise byzantine node
		self._additive_noise_byzantine_node = shared_an.AdditiveNoiseByzantineNode(self, sigma, num_of_samples)

		# Finalize the preparation of the node for the first round
		self.prepare_node_for_the_first_round(genesis_block)

	def _define_nodes_active_in_the_next_round(self, seed: int = SEED, base_list_of_nodes: (list | None) = None):
		if self.aggregation_round() < self._starting_round_for_malicious_behaviour:
			return super()._define_nodes_active_in_the_next_round(seed, base_list_of_nodes)
		
		# If the node can behave maliciously
		return self._additive_noise_byzantine_node._define_nodes_active_in_the_next_round()
		
	def _weight_fit(self):
		if self.aggregation_round() < self._starting_round_for_malicious_behaviour:
			return super()._weight_fit()
		
		return self._additive_noise_byzantine_node._weight_fit()
	
	def _gradient_fit(self):
		if self.aggregation_round() < self._starting_round_for_malicious_behaviour:
			return super()._gradient_fit()
		
		return self._additive_noise_byzantine_node._gradient_fit()

class PosBasedRandomNoiseByzantineNode(PosBasedNode):
	def __init__(self, logger: diagnostic.Diagnostic, node_id: int, host: str, port: int, dataset_quanta_paths: list, archive_host: str, archive_port: int, lazy_loading: bool, test_set_path: str, val_set_path: str, malicious_training_semaphore: BoundedSemaphoreClass, honest_training_semaphore: BoundedSemaphoreClass, validation_semaphore: BoundedSemaphoreClass, fit_epochs: int = 3, batch_size: int = 32, already_available_peers: dict = {}, allowed_to_write_redudant_log_messages: bool = False, test_set_validation_of_global_model_after_each_round: bool = False, genesis_block = None, is_active_trainer_in_first_round: (bool | None) = None, store_weights_directly_in_archive_tmp_dir: bool = False, archive_tmp_dir: (str | None) = None, starting_round_for_malicious_behaviour: int = 1, num_of_samples: int = 0):
		if type(starting_round_for_malicious_behaviour) != int:
			raise TypeError("PosBasedRandomNoiseByzantineNode constructor")
		elif starting_round_for_malicious_behaviour < 1:
			raise ValueError("PosBasedRandomNoiseByzantineNode constructor")

		# Initialization of specific variables of the class
		self._starting_round_for_malicious_behaviour = starting_round_for_malicious_behaviour

		# Initialization of pos based node (it may happen that variables of generic node are overwritten by pos based node)
		# prepare_node_for_first_round is False because we want to delay the start of the node
		super().__init__(logger, node_id, host, port, dataset_quanta_paths, archive_host, archive_port, lazy_loading, test_set_path, val_set_path, malicious_training_semaphore, honest_training_semaphore, validation_semaphore, fit_epochs, batch_size, already_available_peers, allowed_to_write_redudant_log_messages, test_set_validation_of_global_model_after_each_round, genesis_block, is_active_trainer_in_first_round, store_weights_directly_in_archive_tmp_dir, archive_tmp_dir, False)

		# Based initialization of generic node and random noise byzantine node
		self._random_noise_byzantine_node = shared_an.RandomNoiseByzantineNode(self, num_of_samples)

		# Finalize the preparation of the node for the first round
		self.prepare_node_for_the_first_round(genesis_block)

	def _define_nodes_active_in_the_next_round(self, seed: int = SEED, base_list_of_nodes: (list | None) = None):
		if self.aggregation_round() < self._starting_round_for_malicious_behaviour:
			return super()._define_nodes_active_in_the_next_round(seed, base_list_of_nodes)
		
		# If the node can behave maliciously
		return self._random_noise_byzantine_node._define_nodes_active_in_the_next_round()
		
	def _weight_fit(self):
		if self.aggregation_round() < self._starting_round_for_malicious_behaviour:
			return super()._weight_fit()
		
		return self._random_noise_byzantine_node._weight_fit()
	
	def _gradient_fit(self):
		if self.aggregation_round() < self._starting_round_for_malicious_behaviour:
			return super()._gradient_fit()
		
		return self._random_noise_byzantine_node._gradient_fit()

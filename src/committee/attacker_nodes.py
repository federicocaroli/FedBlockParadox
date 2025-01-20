import time

from multiprocessing.synchronize import BoundedSemaphore as BoundedSemaphoreClass
from threading import Lock

from .node import CommitteeBasedNode

from ..shared import attacker_nodes as shared_an, diagnostic
from ..shared.constants import SEED
from ..shared.enums import node_generic as ng, node_messages as nm, common as cm, peer_generic as pg

class CommitteeBasedLabelFlippingNode(CommitteeBasedNode):
	def __init__(self, logger: diagnostic.Diagnostic, node_id: int, host: str, port: int, dataset_quanta_paths: list, archive_host: str, archive_port: int, lazy_loading: bool, test_set_path: str, val_set_path: str, malicious_training_semaphore: BoundedSemaphoreClass, honest_training_semaphore: BoundedSemaphoreClass, validation_semaphore: BoundedSemaphoreClass, fit_epochs: int = 3, batch_size: int = 32, already_available_peers: dict = {}, allowed_to_write_redudant_log_messages: bool = False, test_set_validation_of_global_model_after_each_round: bool = False, genesis_block = None, is_active_trainer_in_first_round: (bool | None) = None, store_weights_directly_in_archive_tmp_dir: bool = False, archive_tmp_dir: (str | None) = None, starting_round_for_malicious_behaviour: int = 1, selected_classes: list=[], target_classes: list=[], num_of_samples: int = 0, malicious_behaviour_only_with_majority_in_committee: bool = False, collusion_peers: list = []):
		if type(starting_round_for_malicious_behaviour) != int or type(malicious_behaviour_only_with_majority_in_committee) != bool or type(collusion_peers) != list or any(type(peer) != int for peer in collusion_peers):
			raise TypeError("CommitteeBasedLabelFlippingNode constructor")
		elif starting_round_for_malicious_behaviour < 1 or (malicious_behaviour_only_with_majority_in_committee is True and len(collusion_peers) == 0) or (malicious_behaviour_only_with_majority_in_committee is True and node_id not in collusion_peers):
			raise ValueError("CommitteeBasedLabelFlippingNode constructor")
		elif type(selected_classes) != list or type(target_classes) != list or type(num_of_samples) != int:
			raise TypeError("CommitteeBasedLabelFlippingNode costructor")
		elif num_of_samples < 0 or len(selected_classes) != len(target_classes) or len(selected_classes) == 0:
			raise ValueError("CommitteeBasedLabelFlippingNode constructor")

		# Initialization of specific variables of the class
		self._starting_round_for_malicious_behaviour = starting_round_for_malicious_behaviour
		self._malicious_behaviour_only_with_majority_in_committee = malicious_behaviour_only_with_majority_in_committee
		self._collusion_peers = collusion_peers

		self._committee_majority = False
		self._collusion_peers_update_names = dict()				# Dict that contains the update names uploaded by the collusion peers
		self._collusion_peers_update_names_lock = Lock()
		self._num_of_malicious_trainers_during_the_round = 0	# This is the theorethical number of malicious trainers during the round, excluding the ones that are malicious validators when the malicious nodes have the majority in the committee

		# Initialization of committee based node (it may happen that variables of generic node are overwritten by committee based node)
		# prepare_node_for_first_round is False because we want to delay the start of the node
		super().__init__(logger, node_id, host, port, dataset_quanta_paths, archive_host, archive_port, lazy_loading, test_set_path, val_set_path, malicious_training_semaphore, honest_training_semaphore, validation_semaphore, fit_epochs, batch_size, already_available_peers, allowed_to_write_redudant_log_messages, test_set_validation_of_global_model_after_each_round, genesis_block, is_active_trainer_in_first_round, store_weights_directly_in_archive_tmp_dir, archive_tmp_dir, False)

		# Based initialization of generic node and label flipping node
		self._label_flipping_node = shared_an.LabelFlippingNode(self, selected_classes, target_classes, num_of_samples)

		# Check if collusion is valid already in the first round
		self._update_committee_majority()

		# Finalize the preparation of the node for the first round
		self.prepare_node_for_the_first_round(genesis_block)		

	def _define_nodes_active_in_the_next_round(self, seed: int = SEED, base_list_of_nodes: (list | None) = None):
		# If the current round is less than the starting round for malicious behaviour, the node behaves as a normal node
		if self.aggregation_round() < self._starting_round_for_malicious_behaviour:
			return super()._define_nodes_active_in_the_next_round(seed, base_list_of_nodes)

		# If the node can behave maliciously but malicious nodes do not have the majority in the committee, the node behaves as a normal node
		elif self._malicious_behaviour_only_with_majority_in_committee and self._committee_majority is False:
			return super()._define_nodes_active_in_the_next_round(seed, base_list_of_nodes)
				
		# If the node can behave maliciously and malicious nodes have the majority in the committee, the node behaves as a malicious node (if it is a validator it will not be an active trainer)
		# If the node can behave maliciously and the malicious behaviour is not conditioned by the majority in the committee, the node behaves as a malicious node
		return self._label_flipping_node._define_nodes_active_in_the_next_round()
		
	def _weight_fit(self):
		if self.aggregation_round() < self._starting_round_for_malicious_behaviour:
			return super()._weight_fit()

		elif self._malicious_behaviour_only_with_majority_in_committee and self._committee_majority is False:
			return super()._weight_fit()

		return self._label_flipping_node._weight_fit()
	
	def _gradient_fit(self):
		if self.aggregation_round() < self._starting_round_for_malicious_behaviour:
			return super()._gradient_fit()

		elif self._malicious_behaviour_only_with_majority_in_committee and self._committee_majority is False:
			return super()._gradient_fit()

		return self._label_flipping_node._gradient_fit()
	
	def _validate_update_created_by_node(self, update_name: str, updater_id: int, update_weights: list):
		try:
			if type(updater_id) != int or type(update_weights) != list or type(update_name) != str:
				raise TypeError("CommitteeBasedLabelFlippingNode _validate_update_created_by_node method")
			
			# Save the update name uploaded by the collusion peers
			if updater_id in self._collusion_peers:
				with self._collusion_peers_update_names_lock:
					if updater_id in self._collusion_peers_update_names:
						raise Exception(f"An update from the same malicious trainer has already been validated. Updater id: {updater_id}. Update name: {update_name}")
					
					self._collusion_peers_update_names[updater_id] = update_name

			super()._validate_update_created_by_node(update_name, updater_id, update_weights)

		except Exception as e:
			self._logger.record(msg = f"Error while validating the update created by a trainer inside a CommitteeBasedLabelFlippingNode", exc = e, logLevel = diagnostic.ERROR, identifier= self._str_identifier)
			raise e

	def _perform_validation_by_means_of_gradients(self):
		"""
		Perform the validation of the updates by means of gradients. If the node is malicious and malicious nodes have the majority in the committee, the validation is always positive.
		"""
		try:
			current_round = self.aggregation_round()

			# If the node can behave maliciously and malicious nodes have the majority in the committee, the validation is falsified
			if current_round >= self._starting_round_for_malicious_behaviour and self._malicious_behaviour_only_with_majority_in_committee and self._committee_majority:
				while self._validation_mechanism.is_validation_ready_to_start() is False:
					time.sleep(1)
				
				update_names_to_validate = self._validation_mechanism.get_list_of_update_names_to_validate()

				with self._collusion_peers_update_names_lock:
					malicious_updates_received_during_the_round = list(self._collusion_peers_update_names.values())
				
				if any(malicious_update_name not in update_names_to_validate for malicious_update_name in malicious_updates_received_during_the_round):
					raise Exception(f"Some malicious updates received during the round have not been considered by the validation mechanism. Malicious updates: {malicious_updates_received_during_the_round}. Updates to validate: {update_names_to_validate}")

				num_of_malicious_updates = len(malicious_updates_received_during_the_round)
				good_updates_names_scores = list()
				validation_results = list()

				for name in update_names_to_validate:
					# If the trainer is malicious and the malicious nodes have the majority in the committee, the validation is always positive
					if name in malicious_updates_received_during_the_round:
						validation_results.append({ng.ValidationResultsListElemFields.UPDATE_NAME: name, ng.ValidationResultsListElemFields.VALIDATION_RESULT: True, ng.ValidationResultsListElemFields.VALIDATION_SCORE: 1})
						good_updates_names_scores.append((name, 1))
				
					# If the number of malicious updates is greater than or equal to the number of validators, the validation of honest updates is always negative because we are able to build a complete committee of malicious nodes
					elif num_of_malicious_updates >= self._num_of_validators:
						validation_results.append({ng.ValidationResultsListElemFields.UPDATE_NAME: name, ng.ValidationResultsListElemFields.VALIDATION_RESULT: False, ng.ValidationResultsListElemFields.VALIDATION_SCORE: 1000})

					# If the number of malicious updates is less than the number of validators, the validation of honest updates is always positive because we need to build a complete committee, so we need few honest nodes
					else:
						validation_results.append({ng.ValidationResultsListElemFields.UPDATE_NAME: name, ng.ValidationResultsListElemFields.VALIDATION_RESULT: True, ng.ValidationResultsListElemFields.VALIDATION_SCORE: 1000})
						good_updates_names_scores.append((name, 1000))

				if self._allowed_to_write_redudant_log_messages:
					self._logger.record(msg = f"Updates to validate: {update_names_to_validate}", logLevel = diagnostic.DEBUG, identifier= self._str_identifier, skipLengthTruncation= True)
					self._logger.record(msg = f"Good updates names and scores: {good_updates_names_scores}", logLevel = diagnostic.DEBUG, identifier= self._str_identifier, skipLengthTruncation= True)
				
				if self._model_aggregation_in_progress.is_set() is False and self.is_peer_alive() and current_round == self.aggregation_round():
					self._send_message(nm.BroadcastMultipleValidationResults.builder(validation_results, self._peer_id, current_round))
					self._handle_multiple_validation_results(validation_results, self._peer_id)
				
				else:
					self._logger.record(msg = f"Validation results not sent because the model aggregation is in progress or the node is not alive", logLevel = diagnostic.DEBUG, identifier= self._str_identifier)
			
			else:
				super()._perform_validation_by_means_of_gradients()
		
		except Exception as e:
			self._logger.record(msg = f"Error while performing the validation by means of a gradients validation mechanism inside a CommitteeBasedLabelFlippingNode", exc = e, logLevel = diagnostic.ERROR, identifier= self._str_identifier)
			raise e

	def _validate_update_by_means_of_weights(self, update_name: str, updater_id: int, update_weights: list):
		"""
		Perform the validation of the updates by means of weights. If the node is malicious and malicious nodes have the majority in the committee, the validation is always positive.
		"""
		try:	
			if type(update_weights) != list or type(update_name) != str or type(updater_id) != int:
				raise TypeError("CommitteeBasedLabelFlippingNode _validate_update_by_means_of_weights method")
			elif self._validation_mechanism_type not in cm.WeightsBasedValidationAlgorithmType.list():
				raise Exception("CommitteeBasedLabelFlippingNode _validate_update_by_means_of_weights method. Validation mechanism is not Local dataset validation or Pass")

			current_round = self.aggregation_round()

			# If the node can behave maliciously and malicious nodes have the majority in the committee, the validation is falsified
			if current_round >= self._starting_round_for_malicious_behaviour and self._malicious_behaviour_only_with_majority_in_committee and self._committee_majority:
				# If the trainer is malicious and the malicious nodes have the majority in the committee, the validation is always positive
				if updater_id in self._collusion_peers:
					if self._validation_mechanism_type in [cm.ValidationAlgorithmType.LOCAL_DATASET_VALIDATION, cm.ValidationAlgorithmType.GLOBAL_DATASET_VALIDATION]:
						accuracy = self._validation_mechanism.get_min_update_validation_score() + 0.0005
						if accuracy > 1.0:
							accuracy = 1.0
					elif self._validation_mechanism_type == cm.ValidationAlgorithmType.PASS_WEIGHTS:
						accuracy = 1.0

					else:
						raise Exception("Unknown weights based validation mechanism")

					positive_validation = True

				# If the number of malicious updates is greater than or equal to the number of validators, the validation of honest updates is always negative because we are able to build a complete committee of malicious nodes
				elif self._num_of_malicious_trainers_during_the_round >= self._num_of_validators:
					accuracy = 0.0
					positive_validation = False
				
				# If the number of malicious updates is less than the number of validators, the validation of honest updates is always positive because we need to build a complete committee, so we need few honest nodes
				else:
					if self._validation_mechanism_type in [cm.ValidationAlgorithmType.LOCAL_DATASET_VALIDATION, cm.ValidationAlgorithmType.GLOBAL_DATASET_VALIDATION]:
						accuracy = self._validation_mechanism.get_min_update_validation_score()
						if accuracy > 1.0:
							accuracy = 1.0
					
					elif self._validation_mechanism_type == cm.ValidationAlgorithmType.PASS_WEIGHTS:
						accuracy = 0.01
					
					else:
						raise Exception("Unknown weights based validation mechanism")

					positive_validation = True

				if self._model_aggregation_in_progress.is_set() is False and self.is_peer_alive() and current_round == self.aggregation_round():
					if self._allowed_to_write_redudant_log_messages:
						self._logger.record(msg = f"Update validated. Update name: {update_name}. Positive validation: {positive_validation}. Score: {accuracy}", logLevel = diagnostic.DEBUG, identifier= self._str_identifier)
				
					self._send_message(nm.BroadcastValidationResult.builder(positive_validation, accuracy, update_name, updater_id, self._peer_id, current_round))
					self._handle_single_validation_result(positive_validation, accuracy, update_name, updater_id, self._peer_id)
			
				else:
					if self._allowed_to_write_redudant_log_messages:
						self._logger.record(msg = f"Update validated but ignored because model aggregation is in progress or node is not alive. Update name: {update_name}", logLevel = diagnostic.DEBUG, identifier= self._str_identifier)

			else:
				super()._validate_update_by_means_of_weights(update_name, updater_id, update_weights)

		except Exception as e:
			self._logger.record(msg = f"Error while performing the validation by means of a weights validation mechanism inside a CommitteeBasedLabelFlippingNode", exc = e, logLevel = diagnostic.ERROR, identifier= self._str_identifier)
			raise e
		
	def _elect_new_validators(self, updater_id_aggregated_validation_scores: dict):
		"""
		Elect the new validators based on the scores of the last aggregation round and check if malicious nodes have the majority in the committee
		"""
		try:
			self._committee_majority = False
			
			with self._collusion_peers_update_names_lock:
				self._collusion_peers_update_names = dict()
			
			self._num_of_malicious_trainers_during_the_round = 0

			super()._elect_new_validators(updater_id_aggregated_validation_scores)

			self._update_committee_majority()
			
		except Exception as e:
			self._logger.record(msg = f"Error while electing the new committee inside a CommitteeBasedLabelFlippingNode", exc = e, logLevel = diagnostic.ERROR, identifier= self._str_identifier)
			raise e

	def _update_committee_majority(self):
		"""
		Update the committee majority
		"""
		try:
			with self._validators_list_lock:
				collusion_peers_in_committee = [peer for peer in self._validators_list if peer in self._collusion_peers]
				if len(collusion_peers_in_committee) >= self._positive_threshold_to_pass_validation:
					self._committee_majority = True
				else:
					self._committee_majority = False
			
				if self._committee_majority:

					malicious_trainers_during_the_round = list()

					with self._peers_lock:
						for malicious_node_id in self._collusion_peers:
							if malicious_node_id not in self._validators_list:
								if malicious_node_id in self._peers and self._peers[malicious_node_id][pg.PeersListElement.STATUS]:
									self._num_of_malicious_trainers_during_the_round += 1
									malicious_trainers_during_the_round.append(malicious_node_id)
								
								else:
									self._logger.record(msg = f"Malicious node {malicious_node_id} is not alive", logLevel = diagnostic.WARNING, identifier= self._str_identifier)

				if self._allowed_to_write_redudant_log_messages:
					if self.aggregation_round() >= self._starting_round_for_malicious_behaviour and self._malicious_behaviour_only_with_majority_in_committee:
						if self._committee_majority:
							self._logger.record(msg = f"Malicious nodes have the majority in the committee. Number of malicious nodes in committee: {len(collusion_peers_in_committee)}. Number of malicious trainers available: {self._num_of_malicious_trainers_during_the_round}. Malicious trainers: {malicious_trainers_during_the_round}", logLevel = diagnostic.INFO, identifier= self._str_identifier)
						else:
							self._logger.record(msg = f"Malicious nodes do NOT have the majority in the committee. Number of malicious nodes in committee: {len(collusion_peers_in_committee)}", logLevel = diagnostic.DEBUG, identifier= self._str_identifier)

		except Exception as e:
			self._logger.record(msg = f"Error while updating the committee majority inside a CommitteeBasedLabelFlippingNode", exc = e, logLevel = diagnostic.ERROR, identifier= self._str_identifier)
			raise e	

class CommitteeBasedTargetedPoisoningNode(CommitteeBasedNode):
	def __init__(self, logger: diagnostic.Diagnostic, node_id: int, host: str, port: int, dataset_quanta_paths: list, archive_host: str, archive_port: int, lazy_loading: bool, test_set_path: str, val_set_path: str, malicious_training_semaphore: BoundedSemaphoreClass, honest_training_semaphore: BoundedSemaphoreClass, validation_semaphore: BoundedSemaphoreClass, fit_epochs: int = 3, batch_size: int = 32, already_available_peers: dict = {}, allowed_to_write_redudant_log_messages: bool = False, test_set_validation_of_global_model_after_each_round: bool = False, genesis_block = None, is_active_trainer_in_first_round: (bool | None) = None, store_weights_directly_in_archive_tmp_dir: bool = False, archive_tmp_dir: (str | None) = None, starting_round_for_malicious_behaviour: int = 1, target_class: (int | None) = None, size: int = 0, num_of_samples: int = 0, malicious_behaviour_only_with_majority_in_committee: bool = False, collusion_peers: list = []):
		if type(starting_round_for_malicious_behaviour) != int or type(malicious_behaviour_only_with_majority_in_committee) != bool or type(collusion_peers) != list or any(type(peer) != int for peer in collusion_peers):
			raise TypeError("CommitteeBasedTargetedPoisoningNode constructor")
		elif starting_round_for_malicious_behaviour < 1 or (malicious_behaviour_only_with_majority_in_committee and len(collusion_peers) == 0) or (malicious_behaviour_only_with_majority_in_committee and node_id not in collusion_peers):
			raise ValueError("CommitteeBasedTargetedPoisoningNode constructor")
				
		# Initialization of specific variables of the class
		self._starting_round_for_malicious_behaviour = starting_round_for_malicious_behaviour
		self._malicious_behaviour_only_with_majority_in_committee = malicious_behaviour_only_with_majority_in_committee
		self._collusion_peers = collusion_peers

		self._committee_majority = False
		self._collusion_peers_update_names = dict()				# Dict that contains the update names uploaded by the collusion peers
		self._collusion_peers_update_names_lock = Lock()
		self._num_of_malicious_trainers_during_the_round = 0	# This is the theorethical number of malicious trainers during the round, excluding the ones that are malicious validators when the malicious nodes have the majority in the committee

		# Initialization of committee based node (it may happen that variables of generic node are overwritten by committee based node)
		# prepare_node_for_first_round is False because we want to delay the start of the node
		super().__init__(logger, node_id, host, port, dataset_quanta_paths, archive_host, archive_port, lazy_loading, test_set_path, val_set_path, malicious_training_semaphore, honest_training_semaphore, validation_semaphore, fit_epochs, batch_size, already_available_peers, allowed_to_write_redudant_log_messages, test_set_validation_of_global_model_after_each_round, genesis_block, is_active_trainer_in_first_round, store_weights_directly_in_archive_tmp_dir, archive_tmp_dir, False)

		# Based initialization of generic node and targeted poisoning node
		self._targeted_poisoning_node = shared_an.TargetedPoisoningNode(self, target_class, size, num_of_samples)

		# Check if collusion is valid already in the first round
		self._update_committee_majority()

		# Finalize the preparation of the node for the first round
		self.prepare_node_for_the_first_round(genesis_block)

	def _define_nodes_active_in_the_next_round(self, seed: int = SEED, base_list_of_nodes: (list | None) = None):
		# If the current round is less than the starting round for malicious behaviour, the node behaves as a normal node
		if self.aggregation_round() < self._starting_round_for_malicious_behaviour:
			return super()._define_nodes_active_in_the_next_round(seed, base_list_of_nodes)

		# If the node can behave maliciously but malicious nodes do not have the majority in the committee, the node behaves as a normal node
		elif self._malicious_behaviour_only_with_majority_in_committee and self._committee_majority is False:
			return super()._define_nodes_active_in_the_next_round(seed, base_list_of_nodes)
		
		# If the node can behave maliciously and malicious nodes have the majority in the committee, the node behaves as a malicious node (if it is a validator it will not be an active trainer)
		# If the node can behave maliciously and the malicious behaviour is not conditioned by the majority in the committee, the node behaves as a malicious node
		return self._targeted_poisoning_node._define_nodes_active_in_the_next_round()
		
	def _weight_fit(self):
		if self.aggregation_round() < self._starting_round_for_malicious_behaviour:
			return super()._weight_fit()
		
		elif self._malicious_behaviour_only_with_majority_in_committee and self._committee_majority is False:
			return super()._weight_fit()

		return self._targeted_poisoning_node._weight_fit()
	
	def _gradient_fit(self):
		if self.aggregation_round() < self._starting_round_for_malicious_behaviour:
			return super()._gradient_fit()

		elif self._malicious_behaviour_only_with_majority_in_committee and self._committee_majority:
			return super()._gradient_fit()

		return self._targeted_poisoning_node._gradient_fit()

	def _validate_update_created_by_node(self, update_name: str, updater_id: int, update_weights: list):
		try:
			if type(updater_id) != int or type(update_weights) != list or type(update_name) != str:
				raise TypeError("CommitteeBasedTargetedPoisoningNode _validate_update_created_by_node method")
			
			# Save the update name uploaded by the collusion peers
			if updater_id in self._collusion_peers:
				with self._collusion_peers_update_names_lock:
					if updater_id in self._collusion_peers_update_names:
						raise Exception(f"An update from the same malicious trainer has already been validated. Updater id: {updater_id}. Update name: {update_name}")
					
					self._collusion_peers_update_names[updater_id] = update_name

			super()._validate_update_created_by_node(update_name, updater_id, update_weights)

		except Exception as e:
			self._logger.record(msg = f"Error while validating the update created by a trainer inside a CommitteeBasedTargetedPoisoningNode", exc = e, logLevel = diagnostic.ERROR, identifier= self._str_identifier)
			raise e

	def _perform_validation_by_means_of_gradients(self):
		"""
		Perform the validation of the updates by means of gradients. If the node is malicious and malicious nodes have the majority in the committee, the validation is always positive.
		"""
		try:
			current_round = self.aggregation_round()

			# If the node can behave maliciously and malicious nodes have the majority in the committee, the validation is falsified
			if current_round >= self._starting_round_for_malicious_behaviour and self._malicious_behaviour_only_with_majority_in_committee and self._committee_majority:
				while self._validation_mechanism.is_validation_ready_to_start() is False:
					time.sleep(1)
				
				update_names_to_validate = self._validation_mechanism.get_list_of_update_names_to_validate()

				with self._collusion_peers_update_names_lock:
					malicious_updates_received_during_the_round = list(self._collusion_peers_update_names.values())
				
				if any(malicious_update_name not in update_names_to_validate for malicious_update_name in malicious_updates_received_during_the_round):
					raise Exception(f"Some malicious updates received during the round have not been considered by the validation mechanism. Malicious updates: {malicious_updates_received_during_the_round}. Updates to validate: {update_names_to_validate}")

				num_of_malicious_updates = len(malicious_updates_received_during_the_round)
				good_updates_names_scores = list()
				validation_results = list()

				for name in update_names_to_validate:
					# If the trainer is malicious and the malicious nodes have the majority in the committee, the validation is always positive
					if name in malicious_updates_received_during_the_round:
						validation_results.append({ng.ValidationResultsListElemFields.UPDATE_NAME: name, ng.ValidationResultsListElemFields.VALIDATION_RESULT: True, ng.ValidationResultsListElemFields.VALIDATION_SCORE: 0})
						good_updates_names_scores.append((name, 1))
				
					# If the number of malicious updates is greater than or equal to the number of validators, the validation of honest updates is always negative because we are able to build a complete committee of malicious nodes
					elif num_of_malicious_updates >= self._num_of_validators:
						validation_results.append({ng.ValidationResultsListElemFields.UPDATE_NAME: name, ng.ValidationResultsListElemFields.VALIDATION_RESULT: False, ng.ValidationResultsListElemFields.VALIDATION_SCORE: 1000})

					# If the number of malicious updates is less than the number of validators, the validation of honest updates is always positive because we need to build a complete committee, so we need few honest nodes
					else:
						validation_results.append({ng.ValidationResultsListElemFields.UPDATE_NAME: name, ng.ValidationResultsListElemFields.VALIDATION_RESULT: True, ng.ValidationResultsListElemFields.VALIDATION_SCORE: 1000})
						good_updates_names_scores.append((name, 1000))

				if self._allowed_to_write_redudant_log_messages:
					self._logger.record(msg = f"Updates to validate: {update_names_to_validate}", logLevel = diagnostic.DEBUG, identifier= self._str_identifier, skipLengthTruncation= True)
					self._logger.record(msg = f"Good updates names and scores: {good_updates_names_scores}", logLevel = diagnostic.DEBUG, identifier= self._str_identifier, skipLengthTruncation= True)
				
				if self._model_aggregation_in_progress.is_set() is False and self.is_peer_alive() and current_round == self.aggregation_round():
					self._send_message(nm.BroadcastMultipleValidationResults.builder(validation_results, self._peer_id, current_round))
					self._handle_multiple_validation_results(validation_results, self._peer_id)
				
				else:
					self._logger.record(msg = f"Validation results not sent because the model aggregation is in progress or the node is not alive", logLevel = diagnostic.DEBUG, identifier= self._str_identifier)
			
			else:
				super()._perform_validation_by_means_of_gradients()
		
		except Exception as e:
			self._logger.record(msg = f"Error while performing the validation by means of a gradients validation mechanism inside a CommitteeBasedTargetedPoisoningNode", exc = e, logLevel = diagnostic.ERROR, identifier= self._str_identifier)
			raise e

	def _validate_update_by_means_of_weights(self, update_name: str, updater_id: int, update_weights: list):
		"""
		Perform the validation of the updates by means of weights. If the node is malicious and malicious nodes have the majority in the committee, the validation is always positive.
		"""
		try:	
			if type(update_weights) != list or type(update_name) != str or type(updater_id) != int:
				raise TypeError("CommitteeBasedTargetedPoisoningNode _validate_update_by_means_of_weights method")
			elif self._validation_mechanism_type not in cm.WeightsBasedValidationAlgorithmType.list():
				raise Exception("CommitteeBasedTargetedPoisoningNode _validate_update_by_means_of_weights method. Validation mechanism is not Local dataset validation or Pass")

			current_round = self.aggregation_round()

			# If the node can behave maliciously and malicious nodes have the majority in the committee, the validation is falsified
			if current_round >= self._starting_round_for_malicious_behaviour and self._malicious_behaviour_only_with_majority_in_committee and self._committee_majority:				# If the trainer is malicious and the malicious nodes have the majority in the committee, the validation is always positive
				if updater_id in self._collusion_peers:
					if self._validation_mechanism_type in [cm.ValidationAlgorithmType.LOCAL_DATASET_VALIDATION, cm.ValidationAlgorithmType.GLOBAL_DATASET_VALIDATION]:
						accuracy = self._validation_mechanism.get_min_update_validation_score() + 0.0005
						if accuracy > 1.0:
							accuracy = 1.0
					elif self._validation_mechanism_type == cm.ValidationAlgorithmType.PASS_WEIGHTS:
						accuracy = 1.0

					else:
						raise Exception("Unknown weights based validation mechanism")
					
					positive_validation = True

				# If the number of malicious updates is greater than or equal to the number of validators, the validation of honest updates is always negative because we are able to build a complete committee of malicious nodes
				elif self._num_of_malicious_trainers_during_the_round >= self._num_of_validators:
					accuracy = 0.0
					positive_validation = False
				
				# If the number of malicious updates is less than the number of validators, the validation of honest updates is always positive because we need to build a complete committee, so we need few honest nodes
				else:
					if self._validation_mechanism_type in [cm.ValidationAlgorithmType.LOCAL_DATASET_VALIDATION, cm.ValidationAlgorithmType.GLOBAL_DATASET_VALIDATION]:
						accuracy = self._validation_mechanism.get_min_update_validation_score()
						if accuracy > 1.0:
							accuracy = 1.0
					elif self._validation_mechanism_type == cm.ValidationAlgorithmType.PASS_WEIGHTS:
						accuracy = 0.01
					
					else:
						raise Exception("Unknown weights based validation mechanism")

					positive_validation = True

				if self._model_aggregation_in_progress.is_set() is False and self.is_peer_alive() and current_round == self.aggregation_round():
					if self._allowed_to_write_redudant_log_messages:
						self._logger.record(msg = f"Update validated. Update name: {update_name}. Positive validation: {positive_validation}. Score: {accuracy}", logLevel = diagnostic.DEBUG, identifier= self._str_identifier)
				
					self._send_message(nm.BroadcastValidationResult.builder(positive_validation, accuracy, update_name, updater_id, self._peer_id, current_round))
					self._handle_single_validation_result(positive_validation, accuracy, update_name, updater_id, self._peer_id)
			
				else:
					if self._allowed_to_write_redudant_log_messages:
						self._logger.record(msg = f"Update validated but ignored because model aggregation is in progress or node is not alive. Update name: {update_name}", logLevel = diagnostic.DEBUG, identifier= self._str_identifier)

			else:
				super()._validate_update_by_means_of_weights(update_name, updater_id, update_weights)

		except Exception as e:
			self._logger.record(msg = f"Error while performing the validation by means of a weights validation mechanism inside a CommitteeBasedTargetedPoisoningNode", exc = e, logLevel = diagnostic.ERROR, identifier= self._str_identifier)
			raise e
		
	def _elect_new_validators(self, updater_id_aggregated_validation_scores: dict):
		"""
		Elect the new validators based on the scores of the last aggregation round and check if malicious nodes have the majority in the committee
		"""
		try:
			self._committee_majority = False
			
			with self._collusion_peers_update_names_lock:
				self._collusion_peers_update_names = dict()
			
			self._num_of_malicious_trainers_during_the_round = 0

			super()._elect_new_validators(updater_id_aggregated_validation_scores)

			self._update_committee_majority()
			
		except Exception as e:
			self._logger.record(msg = f"Error while electing the new committee inside a CommitteeBasedTargetedPoisoningNode", exc = e, logLevel = diagnostic.ERROR, identifier= self._str_identifier)
			raise e

	def _update_committee_majority(self):
		"""
		Update the committee majority
		"""
		try:
			with self._validators_list_lock:
				collusion_peers_in_committee = [peer for peer in self._validators_list if peer in self._collusion_peers]
				if len(collusion_peers_in_committee) >= self._positive_threshold_to_pass_validation:
					self._committee_majority = True
				else:
					self._committee_majority = False
			
				if self._committee_majority:

					malicious_trainers_during_the_round = list()

					with self._peers_lock:
						for malicious_node_id in self._collusion_peers:
							if malicious_node_id not in self._validators_list:
								if malicious_node_id in self._peers and self._peers[malicious_node_id][pg.PeersListElement.STATUS]:
									self._num_of_malicious_trainers_during_the_round += 1
									malicious_trainers_during_the_round.append(malicious_node_id)
								else:
									self._logger.record(msg = f"Malicious node {malicious_node_id} is not alive", logLevel = diagnostic.WARNING, identifier= self._str_identifier)

				if self._allowed_to_write_redudant_log_messages:
					if self.aggregation_round() >= self._starting_round_for_malicious_behaviour and self._malicious_behaviour_only_with_majority_in_committee:
						if self._committee_majority:
							self._logger.record(msg = f"Malicious nodes have the majority in the committee. Number of malicious nodes in committee: {len(collusion_peers_in_committee)}. Number of malicious trainers available: {self._num_of_malicious_trainers_during_the_round}. Malicious trainers: {malicious_trainers_during_the_round}", logLevel = diagnostic.INFO, identifier= self._str_identifier)
						else:
							self._logger.record(msg = f"Malicious nodes do NOT have the majority in the committee. Number of malicious nodes in committee: {len(collusion_peers_in_committee)}", logLevel = diagnostic.DEBUG, identifier= self._str_identifier)

		except Exception as e:
			self._logger.record(msg = f"Error while updating the committee majority inside a CommitteeBasedTargetedPoisoningNode", exc = e, logLevel = diagnostic.ERROR, identifier= self._str_identifier)
			raise e		
		
class CommitteeBasedRandomLabelByzantineNode(CommitteeBasedNode):
	def __init__(self, logger: diagnostic.Diagnostic, node_id: int, host: str, port: int, dataset_quanta_paths: list, archive_host: str, archive_port: int, lazy_loading: bool, test_set_path: str, val_set_path: str, malicious_training_semaphore: BoundedSemaphoreClass, honest_training_semaphore: BoundedSemaphoreClass, validation_semaphore: BoundedSemaphoreClass, fit_epochs: int = 3, batch_size: int = 32, already_available_peers: dict = {}, allowed_to_write_redudant_log_messages: bool = False, test_set_validation_of_global_model_after_each_round: bool = False, genesis_block = None, is_active_trainer_in_first_round: (bool | None) = None, store_weights_directly_in_archive_tmp_dir: bool = False, archive_tmp_dir: (str | None) = None, starting_round_for_malicious_behaviour: int = 1, num_of_samples: int = 0, malicious_behaviour_only_with_majority_in_committee: bool = False, collusion_peers: list = []):
		if type(starting_round_for_malicious_behaviour) != int or type(malicious_behaviour_only_with_majority_in_committee) != bool or type(collusion_peers) != list or any(type(peer) != int for peer in collusion_peers):
			raise TypeError("CommitteeBasedRandomLabelByzantineNode constructor")
		elif starting_round_for_malicious_behaviour < 1 or (malicious_behaviour_only_with_majority_in_committee and len(collusion_peers) == 0) or (malicious_behaviour_only_with_majority_in_committee and node_id not in collusion_peers):
			raise ValueError("CommitteeBasedRandomLabelByzantineNode constructor")
		
		# Initialization of specific variables of the class
		self._starting_round_for_malicious_behaviour = starting_round_for_malicious_behaviour
		self._malicious_behaviour_only_with_majority_in_committee = malicious_behaviour_only_with_majority_in_committee
		self._collusion_peers = collusion_peers

		self._committee_majority = False
		self._collusion_peers_update_names = dict()				# Dict that contains the update names uploaded by the collusion peers
		self._collusion_peers_update_names_lock = Lock()
		self._num_of_malicious_trainers_during_the_round = 0	# This is the theorethical number of malicious trainers during the round, excluding the ones that are malicious validators when the malicious nodes have the majority in the committee

		# Initialization of committee based node (it may happen that variables of generic node are overwritten by committee based node)
		# prepare_node_for_first_round is False because we want to delay the start of the node
		super().__init__(logger, node_id, host, port, dataset_quanta_paths, archive_host, archive_port, lazy_loading, test_set_path, val_set_path, malicious_training_semaphore, honest_training_semaphore, validation_semaphore, fit_epochs, batch_size, already_available_peers, allowed_to_write_redudant_log_messages, test_set_validation_of_global_model_after_each_round, genesis_block, is_active_trainer_in_first_round, store_weights_directly_in_archive_tmp_dir, archive_tmp_dir, False)

		# Based initialization of generic node and random label byzantine node
		self._random_label_byzantine_node = shared_an.RandomLabelByzantineNode(self, num_of_samples)

		# Check if collusion is valid already in the first round
		self._update_committee_majority()

		# Finalize the preparation of the node for the first round
		self.prepare_node_for_the_first_round(genesis_block)

	def _define_nodes_active_in_the_next_round(self, seed: int = SEED, base_list_of_nodes: (list | None) = None):
		if self.aggregation_round() < self._starting_round_for_malicious_behaviour:
			return super()._define_nodes_active_in_the_next_round(seed, base_list_of_nodes)
		
		elif self._malicious_behaviour_only_with_majority_in_committee and self._committee_majority is False:
			return super()._define_nodes_active_in_the_next_round(seed, base_list_of_nodes)
		
		# If the node can behave maliciously
		return self._random_label_byzantine_node._define_nodes_active_in_the_next_round()
	
	def _weight_fit(self):
		if self.aggregation_round() < self._starting_round_for_malicious_behaviour:
			return super()._weight_fit()
		
		elif self._malicious_behaviour_only_with_majority_in_committee and self._committee_majority is False:
			return super()._weight_fit()

		return self._random_label_byzantine_node._weight_fit()
	
	def _gradient_fit(self):
		if self.aggregation_round() < self._starting_round_for_malicious_behaviour:
			return super()._gradient_fit()
		
		elif self._malicious_behaviour_only_with_majority_in_committee and self._committee_majority is False:
			return super()._gradient_fit()
		
		return self._random_label_byzantine_node._gradient_fit()
	
	def _validate_update_created_by_node(self, update_name: str, updater_id: int, update_weights: list):
		try:
			if type(updater_id) != int or type(update_weights) != list or type(update_name) != str:
				raise TypeError("CommitteeBasedRandomLabelByzantineNode _validate_update_created_by_node method")
			
			# Save the update name uploaded by the collusion peers
			if updater_id in self._collusion_peers:
				with self._collusion_peers_update_names_lock:
					if updater_id in self._collusion_peers_update_names:
						raise Exception(f"An update from the same malicious trainer has already been validated. Updater id: {updater_id}. Update name: {update_name}")
					
					self._collusion_peers_update_names[updater_id] = update_name

			super()._validate_update_created_by_node(update_name, updater_id, update_weights)

		except Exception as e:
			self._logger.record(msg = f"Error while validating the update created by a trainer inside a CommitteeBasedRandomLabelByzantineNode", exc = e, logLevel = diagnostic.ERROR, identifier= self._str_identifier)
			raise e

	def _perform_validation_by_means_of_gradients(self):
		"""
		Perform the validation of the updates by means of gradients. If the node is malicious and malicious nodes have the majority in the committee, the validation is always positive.
		"""
		try:
			current_round = self.aggregation_round()

			# If the node can behave maliciously and malicious nodes have the majority in the committee, the validation is falsified
			if current_round >= self._starting_round_for_malicious_behaviour and self._malicious_behaviour_only_with_majority_in_committee and self._committee_majority:
				while self._validation_mechanism.is_validation_ready_to_start() is False:
					time.sleep(1)
				
				update_names_to_validate = self._validation_mechanism.get_list_of_update_names_to_validate()

				with self._collusion_peers_update_names_lock:
					malicious_updates_received_during_the_round = list(self._collusion_peers_update_names.values())
				
				if any(malicious_update_name not in update_names_to_validate for malicious_update_name in malicious_updates_received_during_the_round):
					raise Exception(f"Some malicious updates received during the round have not been considered by the validation mechanism. Malicious updates: {malicious_updates_received_during_the_round}. Updates to validate: {update_names_to_validate}")

				num_of_malicious_updates = len(malicious_updates_received_during_the_round)
				good_updates_names_scores = list()
				validation_results = list()

				for name in update_names_to_validate:
					# If the trainer is malicious and the malicious nodes have the majority in the committee, the validation is always positive
					if name in malicious_updates_received_during_the_round:
						validation_results.append({ng.ValidationResultsListElemFields.UPDATE_NAME: name, ng.ValidationResultsListElemFields.VALIDATION_RESULT: True, ng.ValidationResultsListElemFields.VALIDATION_SCORE: 1})
						good_updates_names_scores.append((name, 1))
				
					# If the number of malicious updates is greater than or equal to the number of validators, the validation of honest updates is always negative because we are able to build a complete committee of malicious nodes
					elif num_of_malicious_updates >= self._num_of_validators:
						validation_results.append({ng.ValidationResultsListElemFields.UPDATE_NAME: name, ng.ValidationResultsListElemFields.VALIDATION_RESULT: False, ng.ValidationResultsListElemFields.VALIDATION_SCORE: 1000})

					# If the number of malicious updates is less than the number of validators, the validation of honest updates is always positive because we need to build a complete committee, so we need few honest nodes
					else:
						validation_results.append({ng.ValidationResultsListElemFields.UPDATE_NAME: name, ng.ValidationResultsListElemFields.VALIDATION_RESULT: True, ng.ValidationResultsListElemFields.VALIDATION_SCORE: 1000})
						good_updates_names_scores.append((name, 1000))

				if self._allowed_to_write_redudant_log_messages:
					self._logger.record(msg = f"Updates to validate: {update_names_to_validate}", logLevel = diagnostic.DEBUG, identifier= self._str_identifier, skipLengthTruncation= True)
					self._logger.record(msg = f"Good updates names and scores: {good_updates_names_scores}", logLevel = diagnostic.DEBUG, identifier= self._str_identifier, skipLengthTruncation= True)
				
				if self._model_aggregation_in_progress.is_set() is False and self.is_peer_alive() and current_round == self.aggregation_round():
					self._send_message(nm.BroadcastMultipleValidationResults.builder(validation_results, self._peer_id, current_round))
					self._handle_multiple_validation_results(validation_results, self._peer_id)
				
				else:
					self._logger.record(msg = f"Validation results not sent because the model aggregation is in progress or the node is not alive", logLevel = diagnostic.DEBUG, identifier= self._str_identifier)
			
			else:
				super()._perform_validation_by_means_of_gradients()
		
		except Exception as e:
			self._logger.record(msg = f"Error while performing the validation by means of a gradients validation mechanism inside a CommitteeBasedRandomLabelByzantineNode", exc = e, logLevel = diagnostic.ERROR, identifier= self._str_identifier)
			raise e

	def _validate_update_by_means_of_weights(self, update_name: str, updater_id: int, update_weights: list):
		"""
		Perform the validation of the updates by means of weights. If the node is malicious and malicious nodes have the majority in the committee, the validation is always positive.
		"""
		try:
			if type(update_weights) != list or type(update_name) != str or type(updater_id) != int:
				raise TypeError("CommitteeBasedRandomLabelByzantineNode _validate_update_by_means_of_weights method")
			elif self._validation_mechanism_type not in cm.WeightsBasedValidationAlgorithmType.list():
				raise Exception("CommitteeBasedRandomLabelByzantineNode _validate_update_by_means_of_weights method. Validation mechanism is not Local dataset validation or Pass")

			current_round = self.aggregation_round()

			# If the node can behave maliciously and malicious nodes have the majority in the committee, the validation is falsified
			if current_round >= self._starting_round_for_malicious_behaviour and self._malicious_behaviour_only_with_majority_in_committee and self._committee_majority:
				if updater_id in self._collusion_peers:
					# If the trainer is malicious and the malicious nodes have the majority in the committee, the validation is always positive
					if self._validation_mechanism_type in [cm.ValidationAlgorithmType.LOCAL_DATASET_VALIDATION, cm.ValidationAlgorithmType.GLOBAL_DATASET_VALIDATION]:
						accuracy = self._validation_mechanism.get_min_update_validation_score() + 0.0005
						if accuracy > 1.0:
							accuracy = 1.0
					elif self._validation_mechanism_type == cm.ValidationAlgorithmType.PASS_WEIGHTS:
						accuracy = 1.0

					else:
						raise Exception("Unknown weights based validation mechanism")
					
					positive_validation = True

				# If the number of malicious updates is greater than or equal to the number of validators, the validation of honest updates is always negative because we are able to build a complete committee of malicious nodes
				elif self._num_of_malicious_trainers_during_the_round >= self._num_of_validators:
					accuracy = 0.0
					positive_validation = False
				
				# If the number of malicious updates is less than the number of validators, the validation of honest updates is always positive because we need to build a complete committee, so we need few honest nodes
				else:
					if self._validation_mechanism_type in [cm.ValidationAlgorithmType.LOCAL_DATASET_VALIDATION, cm.ValidationAlgorithmType.GLOBAL_DATASET_VALIDATION]:
						accuracy = self._validation_mechanism.get_min_update_validation_score()
						if accuracy > 1.0:
							accuracy = 1.0
					elif self._validation_mechanism_type == cm.ValidationAlgorithmType.PASS_WEIGHTS:
						accuracy = 0.01

					else:
						raise Exception("Unknown weights based validation mechanism")

					positive_validation = True

				if self._model_aggregation_in_progress.is_set() is False and self.is_peer_alive() and current_round == self.aggregation_round():
					if self._allowed_to_write_redudant_log_messages:
						self._logger.record(msg = f"Update validated. Update name: {update_name}. Positive validation: {positive_validation}. Score: {accuracy}", logLevel = diagnostic.DEBUG, identifier= self._str_identifier)
				
					self._send_message(nm.BroadcastValidationResult.builder(positive_validation, accuracy, update_name, updater_id, self._peer_id, current_round))
					self._handle_single_validation_result(positive_validation, accuracy, update_name, updater_id, self._peer_id)
			
				else:
					if self._allowed_to_write_redudant_log_messages:
						self._logger.record(msg = f"Update validated but ignored because model aggregation is in progress or node is not alive. Update name: {update_name}", logLevel = diagnostic.DEBUG, identifier= self._str_identifier)

			else:
				super()._validate_update_by_means_of_weights(update_name, updater_id, update_weights)

		except Exception as e:
			self._logger.record(msg = f"Error while performing the validation by means of a weights validation mechanism inside a CommitteeBasedRandomLabelByzantineNode", exc = e, logLevel = diagnostic.ERROR, identifier= self._str_identifier)
			raise e
		
	def _elect_new_validators(self, updater_id_aggregated_validation_scores: dict):
		"""
		Elect the new validators based on the scores of the last aggregation round and check if malicious nodes have the majority in the committee
		"""
		try:
			self._committee_majority = False
			
			with self._collusion_peers_update_names_lock:
				self._collusion_peers_update_names = dict()
			
			self._num_of_malicious_trainers_during_the_round = 0

			super()._elect_new_validators(updater_id_aggregated_validation_scores)

			self._update_committee_majority()
			
		except Exception as e:
			self._logger.record(msg = f"Error while electing the new committee inside a CommitteeBasedRandomLabelByzantineNode", exc = e, logLevel = diagnostic.ERROR, identifier= self._str_identifier)
			raise e

	def _update_committee_majority(self):
		"""
		Update the committee majority
		"""
		try:
			with self._validators_list_lock:
				collusion_peers_in_committee = [peer for peer in self._validators_list if peer in self._collusion_peers]
				if len(collusion_peers_in_committee) >= self._positive_threshold_to_pass_validation:
					self._committee_majority = True
				else:
					self._committee_majority = False
			
				if self._committee_majority:

					malicious_trainers_during_the_round = list()

					with self._peers_lock:
						for malicious_node_id in self._collusion_peers:
							if malicious_node_id not in self._validators_list:
								if malicious_node_id in self._peers and self._peers[malicious_node_id][pg.PeersListElement.STATUS]:
									self._num_of_malicious_trainers_during_the_round += 1
									malicious_trainers_during_the_round.append(malicious_node_id)
								
								else:
									self._logger.record(msg = f"Malicious node {malicious_node_id} is not alive", logLevel = diagnostic.WARNING, identifier= self._str_identifier)

				if self._allowed_to_write_redudant_log_messages:
					if self.aggregation_round() >= self._starting_round_for_malicious_behaviour and self._malicious_behaviour_only_with_majority_in_committee:
						if self._committee_majority:
							self._logger.record(msg = f"Malicious nodes have the majority in the committee. Number of malicious nodes in committee: {len(collusion_peers_in_committee)}. Number of malicious trainers available: {self._num_of_malicious_trainers_during_the_round}. Malicious trainers: {malicious_trainers_during_the_round}", logLevel = diagnostic.INFO, identifier= self._str_identifier)
						else:
							self._logger.record(msg = f"Malicious nodes do NOT have the majority in the committee. Number of malicious nodes in committee: {len(collusion_peers_in_committee)}", logLevel = diagnostic.DEBUG, identifier= self._str_identifier)

		except Exception as e:
			self._logger.record(msg = f"Error while updating the committee majority inside a CommitteeBasedRandomLabelByzantineNode", exc = e, logLevel = diagnostic.ERROR, identifier= self._str_identifier)
			raise e	
			
class CommitteeBasedAdditiveNoiseByzantineNode(CommitteeBasedNode):
	def __init__(self, logger: diagnostic.Diagnostic, node_id: int, host: str, port: int, dataset_quanta_paths: list, archive_host: str, archive_port: int, lazy_loading: bool, test_set_path: str, val_set_path: str, malicious_training_semaphore: BoundedSemaphoreClass, honest_training_semaphore: BoundedSemaphoreClass, validation_semaphore: BoundedSemaphoreClass, fit_epochs: int = 3, batch_size: int = 32, already_available_peers: dict = {}, allowed_to_write_redudant_log_messages: bool = False, test_set_validation_of_global_model_after_each_round: bool = False, genesis_block = None, is_active_trainer_in_first_round: (bool | None) = None, store_weights_directly_in_archive_tmp_dir: bool = False, archive_tmp_dir: (str | None) = None, starting_round_for_malicious_behaviour: int = 1, sigma: float = 0.1, num_of_samples: int = 0, malicious_behaviour_only_with_majority_in_committee: bool = False, collusion_peers: list = []):
		if type(starting_round_for_malicious_behaviour) != int or type(malicious_behaviour_only_with_majority_in_committee) != bool or type(collusion_peers) != list or any(type(peer) != int for peer in collusion_peers):
			raise TypeError("CommitteeBasedAdditiveNoiseByzantineNode constructor")
		elif starting_round_for_malicious_behaviour < 1 or (malicious_behaviour_only_with_majority_in_committee and len(collusion_peers) == 0) or (malicious_behaviour_only_with_majority_in_committee and node_id not in collusion_peers):
			raise ValueError("CommitteeBasedAdditiveNoiseByzantineNode constructor")
	
		# Initialization of specific variables of the class
		self._starting_round_for_malicious_behaviour = starting_round_for_malicious_behaviour
		self._malicious_behaviour_only_with_majority_in_committee = malicious_behaviour_only_with_majority_in_committee
		self._collusion_peers = collusion_peers

		self._committee_majority = False
		self._collusion_peers_update_names = dict()				# Dict that contains the update names uploaded by the collusion peers
		self._collusion_peers_update_names_lock = Lock()
		self._num_of_malicious_trainers_during_the_round = 0	# This is the theorethical number of malicious trainers during the round, excluding the ones that are malicious validators when the malicious nodes have the majority in the committee

		# Initialization of committee based node (it may happen that variables of generic node are overwritten by committee based node)
		# prepare_node_for_first_round is False because we want to delay the start of the node
		super().__init__(logger, node_id, host, port, dataset_quanta_paths, archive_host, archive_port, lazy_loading, test_set_path, val_set_path, malicious_training_semaphore, honest_training_semaphore, validation_semaphore, fit_epochs, batch_size, already_available_peers, allowed_to_write_redudant_log_messages, test_set_validation_of_global_model_after_each_round, genesis_block, is_active_trainer_in_first_round, store_weights_directly_in_archive_tmp_dir, archive_tmp_dir, False)

		# Based initialization of generic node and additive noise byzantine node
		self._additive_noise_byzantine_node = shared_an.AdditiveNoiseByzantineNode(self, sigma, num_of_samples)

		# Check if collusion is valid already in the first round
		self._update_committee_majority()

		# Finalize the preparation of the node for the first round
		self.prepare_node_for_the_first_round(genesis_block)

	def _define_nodes_active_in_the_next_round(self, seed: int = SEED, base_list_of_nodes: (list | None) = None):
		if self.aggregation_round() < self._starting_round_for_malicious_behaviour:
			return super()._define_nodes_active_in_the_next_round(seed, base_list_of_nodes)

		elif self._malicious_behaviour_only_with_majority_in_committee and self._committee_majority is False:
			return super()._define_nodes_active_in_the_next_round(seed, base_list_of_nodes)
		
		# If the node can behave maliciously
		return self._additive_noise_byzantine_node._define_nodes_active_in_the_next_round()
		
	def _weight_fit(self):
		if self.aggregation_round() < self._starting_round_for_malicious_behaviour:
			return super()._weight_fit()
		
		elif self._malicious_behaviour_only_with_majority_in_committee and self._committee_majority is False:
			return super()._weight_fit()

		return self._additive_noise_byzantine_node._weight_fit()
	
	def _gradient_fit(self):
		if self.aggregation_round() < self._starting_round_for_malicious_behaviour:
			return super()._gradient_fit()
		
		elif self._malicious_behaviour_only_with_majority_in_committee and self._committee_majority is False:
			return super()._gradient_fit()

		return self._additive_noise_byzantine_node._gradient_fit()

	def _validate_update_created_by_node(self, update_name: str, updater_id: int, update_weights: list):
		try:
			if type(updater_id) != int or type(update_weights) != list or type(update_name) != str:
				raise TypeError("CommitteeBasedAdditiveNoiseByzantineNode _validate_update_created_by_node method")
			
			# Save the update name uploaded by the collusion peers
			if updater_id in self._collusion_peers:
				with self._collusion_peers_update_names_lock:
					if updater_id in self._collusion_peers_update_names:
						raise Exception(f"An update from the same malicious trainer has already been validated. Updater id: {updater_id}. Update name: {update_name}")
					
					self._collusion_peers_update_names[updater_id] = update_name

			super()._validate_update_created_by_node(update_name, updater_id, update_weights)

		except Exception as e:
			self._logger.record(msg = f"Error while validating the update created by a trainer inside a CommitteeBasedAdditiveNoiseByzantineNode", exc = e, logLevel = diagnostic.ERROR, identifier= self._str_identifier)
			raise e

	def _perform_validation_by_means_of_gradients(self):
		"""
		Perform the validation of the updates by means of gradients. If the node is malicious and malicious nodes have the majority in the committee, the validation is always positive.
		"""
		try:
			current_round = self.aggregation_round()

			# If the node can behave maliciously and malicious nodes have the majority in the committee, the validation is falsified
			if current_round >= self._starting_round_for_malicious_behaviour and self._malicious_behaviour_only_with_majority_in_committee and self._committee_majority:
				while self._validation_mechanism.is_validation_ready_to_start() is False:
					time.sleep(1)
				
				update_names_to_validate = self._validation_mechanism.get_list_of_update_names_to_validate()

				with self._collusion_peers_update_names_lock:
					malicious_updates_received_during_the_round = list(self._collusion_peers_update_names.values())
				
				if any(malicious_update_name not in update_names_to_validate for malicious_update_name in malicious_updates_received_during_the_round):
					raise Exception(f"Some malicious updates received during the round have not been considered by the validation mechanism. Malicious updates: {malicious_updates_received_during_the_round}. Updates to validate: {update_names_to_validate}")

				num_of_malicious_updates = len(malicious_updates_received_during_the_round)
				good_updates_names_scores = list()
				validation_results = list()

				for name in update_names_to_validate:
					# If the trainer is malicious and the malicious nodes have the majority in the committee, the validation is always positive
					if name in malicious_updates_received_during_the_round:
						validation_results.append({ng.ValidationResultsListElemFields.UPDATE_NAME: name, ng.ValidationResultsListElemFields.VALIDATION_RESULT: True, ng.ValidationResultsListElemFields.VALIDATION_SCORE: 1})
						good_updates_names_scores.append((name, 1))
				
					# If the number of malicious updates is greater than or equal to the number of validators, the validation of honest updates is always negative because we are able to build a complete committee of malicious nodes
					elif num_of_malicious_updates >= self._num_of_validators:
						validation_results.append({ng.ValidationResultsListElemFields.UPDATE_NAME: name, ng.ValidationResultsListElemFields.VALIDATION_RESULT: False, ng.ValidationResultsListElemFields.VALIDATION_SCORE: 1000})

					# If the number of malicious updates is less than the number of validators, the validation of honest updates is always positive because we need to build a complete committee, so we need few honest nodes
					else:
						validation_results.append({ng.ValidationResultsListElemFields.UPDATE_NAME: name, ng.ValidationResultsListElemFields.VALIDATION_RESULT: True, ng.ValidationResultsListElemFields.VALIDATION_SCORE: 1000})
						good_updates_names_scores.append((name, 1000))

				if self._allowed_to_write_redudant_log_messages:
					self._logger.record(msg = f"Updates to validate: {update_names_to_validate}", logLevel = diagnostic.DEBUG, identifier= self._str_identifier, skipLengthTruncation= True)
					self._logger.record(msg = f"Good updates names and scores: {good_updates_names_scores}", logLevel = diagnostic.DEBUG, identifier= self._str_identifier, skipLengthTruncation= True)
				
				if self._model_aggregation_in_progress.is_set() is False and self.is_peer_alive() and current_round == self.aggregation_round():
					self._send_message(nm.BroadcastMultipleValidationResults.builder(validation_results, self._peer_id, current_round))
					self._handle_multiple_validation_results(validation_results, self._peer_id)
				
				else:
					self._logger.record(msg = f"Validation results not sent because the model aggregation is in progress or the node is not alive", logLevel = diagnostic.DEBUG, identifier= self._str_identifier)
			
			else:
				super()._perform_validation_by_means_of_gradients()
		
		except Exception as e:
			self._logger.record(msg = f"Error while performing the validation by means of a gradients validation mechanism inside a CommitteeBasedAdditiveNoiseByzantineNode", exc = e, logLevel = diagnostic.ERROR, identifier= self._str_identifier)
			raise e

	def _validate_update_by_means_of_weights(self, update_name: str, updater_id: int, update_weights: list):
		"""
		Perform the validation of the updates by means of weights. If the node is malicious and malicious nodes have the majority in the committee, the validation is always positive.
		"""
		try:	
			if type(update_weights) != list or type(update_name) != str or type(updater_id) != int:
				raise TypeError("CommitteeBasedAdditiveNoiseByzantineNode _validate_update_by_means_of_weights method")
			elif self._validation_mechanism_type not in cm.WeightsBasedValidationAlgorithmType.list():
				raise Exception("CommitteeBasedAdditiveNoiseByzantineNode _validate_update_by_means_of_weights method. Validation mechanism is not Local dataset validation or Pass")
			
			current_round = self.aggregation_round()

			# If the node can behave maliciously and malicious nodes have the majority in the committee, the validation is falsified
			if current_round >= self._starting_round_for_malicious_behaviour and self._malicious_behaviour_only_with_majority_in_committee and self._committee_majority:
				# If the trainer is malicious and the malicious nodes have the majority in the committee, the validation is always positive
				if updater_id in self._collusion_peers:
					if self._validation_mechanism_type in [cm.ValidationAlgorithmType.LOCAL_DATASET_VALIDATION, cm.ValidationAlgorithmType.GLOBAL_DATASET_VALIDATION]:
						accuracy = self._validation_mechanism.get_min_update_validation_score() + 0.0005
						if accuracy > 1.0:
							accuracy = 1.0
					elif self._validation_mechanism_type == cm.ValidationAlgorithmType.PASS_WEIGHTS:
						accuracy = 1.0

					else:
						raise Exception("Unknown weights based validation mechanism")

					positive_validation = True

				# If the number of malicious updates is greater than or equal to the number of validators, the validation of honest updates is always negative because we are able to build a complete committee of malicious nodes
				elif self._num_of_malicious_trainers_during_the_round >= self._num_of_validators:
					accuracy = 0.0
					positive_validation = False
				
				# If the number of malicious updates is less than the number of validators, the validation of honest updates is always positive because we need to build a complete committee, so we need few honest nodes
				else:
					if self._validation_mechanism_type in [cm.ValidationAlgorithmType.LOCAL_DATASET_VALIDATION, cm.ValidationAlgorithmType.GLOBAL_DATASET_VALIDATION]:
						accuracy = self._validation_mechanism.get_min_update_validation_score()
						if accuracy > 1.0:
							accuracy = 1.0					
					elif self._validation_mechanism_type == cm.ValidationAlgorithmType.PASS_WEIGHTS:
						accuracy = 0.1

					else:
						raise Exception("Unknown weights based validation mechanism")

					positive_validation = True

				if self._model_aggregation_in_progress.is_set() is False and self.is_peer_alive() and current_round == self.aggregation_round():
					if self._allowed_to_write_redudant_log_messages:
						self._logger.record(msg = f"Update validated. Update name: {update_name}. Positive validation: {positive_validation}. Score: {accuracy}", logLevel = diagnostic.DEBUG, identifier= self._str_identifier)
				
					self._send_message(nm.BroadcastValidationResult.builder(positive_validation, accuracy, update_name, updater_id, self._peer_id, current_round))
					self._handle_single_validation_result(positive_validation, accuracy, update_name, updater_id, self._peer_id)
			
				else:
					if self._allowed_to_write_redudant_log_messages:
						self._logger.record(msg = f"Update validated but ignored because model aggregation is in progress or node is not alive. Update name: {update_name}", logLevel = diagnostic.DEBUG, identifier= self._str_identifier)

			else:
				super()._validate_update_by_means_of_weights(update_name, updater_id, update_weights)

		except Exception as e:
			self._logger.record(msg = f"Error while performing the validation by means of a weights validation mechanism inside a CommitteeBasedAdditiveNoiseByzantineNode", exc = e, logLevel = diagnostic.ERROR, identifier= self._str_identifier)
			raise e
		
	def _elect_new_validators(self, updater_id_aggregated_validation_scores: dict):
		"""
		Elect the new validators based on the scores of the last aggregation round and check if malicious nodes have the majority in the committee
		"""
		try:
			self._committee_majority = False
			
			with self._collusion_peers_update_names_lock:
				self._collusion_peers_update_names = dict()
			
			self._num_of_malicious_trainers_during_the_round = 0

			super()._elect_new_validators(updater_id_aggregated_validation_scores)

			self._update_committee_majority()
			
		except Exception as e:
			self._logger.record(msg = f"Error while electing the new committee inside a CommitteeBasedAdditiveNoiseByzantineNode", exc = e, logLevel = diagnostic.ERROR, identifier= self._str_identifier)
			raise e

	def _update_committee_majority(self):
		"""
		Update the committee majority
		"""
		try:
			with self._validators_list_lock:
				collusion_peers_in_committee = [peer for peer in self._validators_list if peer in self._collusion_peers]
				if len(collusion_peers_in_committee) >= self._positive_threshold_to_pass_validation:
					self._committee_majority = True
				else:
					self._committee_majority = False
			
				if self._committee_majority:

					malicious_trainers_during_the_round = list()

					with self._peers_lock:
						for malicious_node_id in self._collusion_peers:
							if malicious_node_id not in self._validators_list:
								if malicious_node_id in self._peers and self._peers[malicious_node_id][pg.PeersListElement.STATUS]:
									self._num_of_malicious_trainers_during_the_round += 1
									malicious_trainers_during_the_round.append(malicious_node_id)
								
								else:
									self._logger.record(msg = f"Malicious node {malicious_node_id} is not alive", logLevel = diagnostic.WARNING, identifier= self._str_identifier)

				if self._allowed_to_write_redudant_log_messages:
					if self.aggregation_round() >= self._starting_round_for_malicious_behaviour and self._malicious_behaviour_only_with_majority_in_committee:
						if self._committee_majority:
							self._logger.record(msg = f"Malicious nodes have the majority in the committee. Number of malicious nodes in committee: {len(collusion_peers_in_committee)}. Number of malicious trainers available: {self._num_of_malicious_trainers_during_the_round}. Malicious trainers: {malicious_trainers_during_the_round}", logLevel = diagnostic.INFO, identifier= self._str_identifier)
						else:
							self._logger.record(msg = f"Malicious nodes do NOT have the majority in the committee. Number of malicious nodes in committee: {len(collusion_peers_in_committee)}", logLevel = diagnostic.DEBUG, identifier= self._str_identifier)

		except Exception as e:
			self._logger.record(msg = f"Error while updating the committee majority inside a CommitteeBasedAdditiveNoiseByzantineNode", exc = e, logLevel = diagnostic.ERROR, identifier= self._str_identifier)
			raise e		
	
class CommitteeBasedRandomNoiseByzantineNode(CommitteeBasedNode):
	def __init__(self, logger: diagnostic.Diagnostic, node_id: int, host: str, port: int, dataset_quanta_paths: list, archive_host: str, archive_port: int, lazy_loading: bool, test_set_path: str, val_set_path: str, malicious_training_semaphore: BoundedSemaphoreClass, honest_training_semaphore: BoundedSemaphoreClass, validation_semaphore: BoundedSemaphoreClass, fit_epochs: int = 3, batch_size: int = 32, already_available_peers: dict = {}, allowed_to_write_redudant_log_messages: bool = False, test_set_validation_of_global_model_after_each_round: bool = False, genesis_block = None, is_active_trainer_in_first_round: (bool | None) = None, store_weights_directly_in_archive_tmp_dir: bool = False, archive_tmp_dir: (str | None) = None, starting_round_for_malicious_behaviour: int = 1, num_of_samples: int = 0, malicious_behaviour_only_with_majority_in_committee: bool = False, collusion_peers: list = []):
		if type(starting_round_for_malicious_behaviour) != int or type(malicious_behaviour_only_with_majority_in_committee) != bool or type(collusion_peers) != list or any(type(peer) != int for peer in collusion_peers):
			raise TypeError("CommitteeBasedRandomNoiseByzantineNode constructor")
		elif starting_round_for_malicious_behaviour < 1 or (malicious_behaviour_only_with_majority_in_committee and len(collusion_peers) == 0) or (malicious_behaviour_only_with_majority_in_committee and node_id not in collusion_peers):
			raise ValueError("CommitteeBasedRandomNoiseByzantineNode constructor")

		# Initialization of specific variables of the class
		self._starting_round_for_malicious_behaviour = starting_round_for_malicious_behaviour
		self._malicious_behaviour_only_with_majority_in_committee = malicious_behaviour_only_with_majority_in_committee
		self._collusion_peers = collusion_peers

		self._committee_majority = False
		self._collusion_peers_update_names = dict()				# Dict that contains the update names uploaded by the collusion peers
		self._collusion_peers_update_names_lock = Lock()
		self._num_of_malicious_trainers_during_the_round = 0	# This is the theorethical number of malicious trainers during the round, excluding the ones that are malicious validators when the malicious nodes have the majority in the committee

		# Initialization of committee based node (it may happen that variables of generic node are overwritten by committee based node)
		# prepare_node_for_first_round is False because we want to delay the start of the node
		super().__init__(logger, node_id, host, port, dataset_quanta_paths, archive_host, archive_port, lazy_loading, test_set_path, val_set_path, malicious_training_semaphore, honest_training_semaphore, validation_semaphore, fit_epochs, batch_size, already_available_peers, allowed_to_write_redudant_log_messages, test_set_validation_of_global_model_after_each_round, genesis_block, is_active_trainer_in_first_round, store_weights_directly_in_archive_tmp_dir, archive_tmp_dir, False)

		# Based initialization of generic node and random noise byzantine node
		self._random_noise_byzantine_node = shared_an.RandomNoiseByzantineNode(self, num_of_samples)

		# Check if collusion is valid already in the first round
		self._update_committee_majority()

		# Finalize the preparation of the node for the first round
		self.prepare_node_for_the_first_round(genesis_block)

	def _define_nodes_active_in_the_next_round(self, seed: int = SEED, base_list_of_nodes: (list | None) = None):
		if self.aggregation_round() < self._starting_round_for_malicious_behaviour:
			return super()._define_nodes_active_in_the_next_round(seed, base_list_of_nodes)

		elif self._malicious_behaviour_only_with_majority_in_committee and self._committee_majority is False:
			return super()._define_nodes_active_in_the_next_round(seed, base_list_of_nodes)
		
		# If the node can behave maliciously
		return self._random_noise_byzantine_node._define_nodes_active_in_the_next_round()
		
	def _weight_fit(self):
		if self.aggregation_round() < self._starting_round_for_malicious_behaviour:
			return super()._weight_fit()
		
		elif self._malicious_behaviour_only_with_majority_in_committee and self._committee_majority is False:
			return super()._weight_fit()

		return self._random_noise_byzantine_node._weight_fit()
	
	def _gradient_fit(self):
		if self.aggregation_round() < self._starting_round_for_malicious_behaviour:
			return super()._gradient_fit()
		
		elif self._malicious_behaviour_only_with_majority_in_committee and self._committee_majority is False:
			return super()._gradient_fit()

		return self._random_noise_byzantine_node._gradient_fit()

	def _validate_update_created_by_node(self, update_name: str, updater_id: int, update_weights: list):
		try:
			if type(updater_id) != int or type(update_weights) != list or type(update_name) != str:
				raise TypeError("CommitteeBasedRandomNoiseByzantineNode _validate_update_created_by_node method")
			
			# Save the update name uploaded by the collusion peers
			if updater_id in self._collusion_peers:
				with self._collusion_peers_update_names_lock:
					if updater_id in self._collusion_peers_update_names:
						raise Exception(f"An update from the same malicious trainer has already been validated. Updater id: {updater_id}. Update name: {update_name}")
					
					self._collusion_peers_update_names[updater_id] = update_name

			super()._validate_update_created_by_node(update_name, updater_id, update_weights)

		except Exception as e:
			self._logger.record(msg = f"Error while validating the update created by a trainer inside a CommitteeBasedRandomNoiseByzantineNode", exc = e, logLevel = diagnostic.ERROR, identifier= self._str_identifier)
			raise e

	def _perform_validation_by_means_of_gradients(self):
		"""
		Perform the validation of the updates by means of gradients. If the node is malicious and malicious nodes have the majority in the committee, the validation is always positive.
		"""
		try:
			current_round = self.aggregation_round()

			# If the node can behave maliciously and malicious nodes have the majority in the committee, the validation is falsified
			if current_round >= self._starting_round_for_malicious_behaviour and self._malicious_behaviour_only_with_majority_in_committee and self._committee_majority:
				while self._validation_mechanism.is_validation_ready_to_start() is False:
					time.sleep(1)
				
				update_names_to_validate = self._validation_mechanism.get_list_of_update_names_to_validate()

				with self._collusion_peers_update_names_lock:
					malicious_updates_received_during_the_round = list(self._collusion_peers_update_names.values())
				
				if any(malicious_update_name not in update_names_to_validate for malicious_update_name in malicious_updates_received_during_the_round):
					raise Exception(f"Some malicious updates received during the round have not been considered by the validation mechanism. Malicious updates: {malicious_updates_received_during_the_round}. Updates to validate: {update_names_to_validate}")

				num_of_malicious_updates = len(malicious_updates_received_during_the_round)
				good_updates_names_scores = list()
				validation_results = list()

				for name in update_names_to_validate:
					# If the trainer is malicious and the malicious nodes have the majority in the committee, the validation is always positive
					if name in malicious_updates_received_during_the_round:
						validation_results.append({ng.ValidationResultsListElemFields.UPDATE_NAME: name, ng.ValidationResultsListElemFields.VALIDATION_RESULT: True, ng.ValidationResultsListElemFields.VALIDATION_SCORE: 1})
						good_updates_names_scores.append((name, 1))
				
					# If the number of malicious updates is greater than or equal to the number of validators, the validation of honest updates is always negative because we are able to build a complete committee of malicious nodes
					elif num_of_malicious_updates >= self._num_of_validators:
						validation_results.append({ng.ValidationResultsListElemFields.UPDATE_NAME: name, ng.ValidationResultsListElemFields.VALIDATION_RESULT: False, ng.ValidationResultsListElemFields.VALIDATION_SCORE: 1000})

					# If the number of malicious updates is less than the number of validators, the validation of honest updates is always positive because we need to build a complete committee, so we need few honest nodes
					else:
						validation_results.append({ng.ValidationResultsListElemFields.UPDATE_NAME: name, ng.ValidationResultsListElemFields.VALIDATION_RESULT: True, ng.ValidationResultsListElemFields.VALIDATION_SCORE: 1000})
						good_updates_names_scores.append((name, 1000))

				if self._allowed_to_write_redudant_log_messages:
					self._logger.record(msg = f"Updates to validate: {update_names_to_validate}", logLevel = diagnostic.DEBUG, identifier= self._str_identifier, skipLengthTruncation= True)
					self._logger.record(msg = f"Good updates names and scores: {good_updates_names_scores}", logLevel = diagnostic.DEBUG, identifier= self._str_identifier, skipLengthTruncation= True)
				
				if self._model_aggregation_in_progress.is_set() is False and self.is_peer_alive() and current_round == self.aggregation_round():
					self._send_message(nm.BroadcastMultipleValidationResults.builder(validation_results, self._peer_id, current_round))
					self._handle_multiple_validation_results(validation_results, self._peer_id)
				
				else:
					self._logger.record(msg = f"Validation results not sent because the model aggregation is in progress or the node is not alive", logLevel = diagnostic.DEBUG, identifier= self._str_identifier)
			
			else:
				super()._perform_validation_by_means_of_gradients()
		
		except Exception as e:
			self._logger.record(msg = f"Error while performing the validation by means of a gradients validation mechanism inside a CommitteeBasedRandomNoiseByzantineNode", exc = e, logLevel = diagnostic.ERROR, identifier= self._str_identifier)
			raise e

	def _validate_update_by_means_of_weights(self, update_name: str, updater_id: int, update_weights: list):
		"""
		Perform the validation of the updates by means of weights. If the node is malicious and malicious nodes have the majority in the committee, the validation is always positive.
		"""
		try:	
			if type(update_weights) != list or type(update_name) != str or type(updater_id) != int:
				raise TypeError("CommitteeBasedRandomNoiseByzantineNode _validate_update_by_means_of_weights method")
			elif self._validation_mechanism_type not in cm.WeightsBasedValidationAlgorithmType.list():
				raise Exception("CommitteeBasedRandomNoiseByzantineNode _validate_update_by_means_of_weights method. Validation mechanism is not Local dataset validation or Pass")
			
			current_round = self.aggregation_round()

			# If the node can behave maliciously and malicious nodes have the majority in the committee, the validation is falsified
			if current_round >= self._starting_round_for_malicious_behaviour and self._malicious_behaviour_only_with_majority_in_committee and self._committee_majority:				# If the trainer is malicious and the malicious nodes have the majority in the committee, the validation is always positive
				
				if updater_id in self._collusion_peers:
					if self._validation_mechanism_type in [cm.ValidationAlgorithmType.LOCAL_DATASET_VALIDATION, cm.ValidationAlgorithmType.GLOBAL_DATASET_VALIDATION]:
						accuracy = self._validation_mechanism.get_min_update_validation_score() + 0.0005
						if accuracy > 1.0:
							accuracy = 1.0
					elif self._validation_mechanism_type == cm.ValidationAlgorithmType.PASS_WEIGHTS:
						accuracy = 1.0

					else:
						raise Exception("Unknown weights based validation mechanism")

					positive_validation = True

				# If the number of malicious updates is greater than or equal to the number of validators, the validation of honest updates is always negative because we are able to build a complete committee of malicious nodes
				elif self._num_of_malicious_trainers_during_the_round >= self._num_of_validators:
					accuracy = 0.0
					positive_validation = False
				
				# If the number of malicious updates is less than the number of validators, the validation of honest updates is always positive because we need to build a complete committee, so we need few honest nodes
				else:
					if self._validation_mechanism_type in [cm.ValidationAlgorithmType.LOCAL_DATASET_VALIDATION, cm.ValidationAlgorithmType.GLOBAL_DATASET_VALIDATION]:
						accuracy = self._validation_mechanism.get_min_update_validation_score()
						if accuracy > 1.0:
							accuracy = 1.0
					
					elif self._validation_mechanism_type == cm.ValidationAlgorithmType.PASS_WEIGHTS:
						accuracy = 0.01
					
					else:
						raise Exception("Unknown weights based validation mechanism")

					positive_validation = True

				if self._model_aggregation_in_progress.is_set() is False and self.is_peer_alive() and current_round == self.aggregation_round():
					if self._allowed_to_write_redudant_log_messages:
						self._logger.record(msg = f"Update validated. Update name: {update_name}. Positive validation: {positive_validation}. Score: {accuracy}", logLevel = diagnostic.DEBUG, identifier= self._str_identifier)
				
					self._send_message(nm.BroadcastValidationResult.builder(positive_validation, accuracy, update_name, updater_id, self._peer_id, current_round))
					self._handle_single_validation_result(positive_validation, accuracy, update_name, updater_id, self._peer_id)
			
				else:
					if self._allowed_to_write_redudant_log_messages:
						self._logger.record(msg = f"Update validated but ignored because model aggregation is in progress or node is not alive. Update name: {update_name}", logLevel = diagnostic.DEBUG, identifier= self._str_identifier)

			else:
				super()._validate_update_by_means_of_weights(update_name, updater_id, update_weights)

		except Exception as e:
			self._logger.record(msg = f"Error while performing the validation by means of a weights validation mechanism inside a CommitteeBasedRandomNoiseByzantineNode", exc = e, logLevel = diagnostic.ERROR, identifier= self._str_identifier)
			raise e
		
	def _elect_new_validators(self, updater_id_aggregated_validation_scores: dict):
		"""
		Elect the new validators based on the scores of the last aggregation round and check if malicious nodes have the majority in the committee
		"""
		try:
			self._committee_majority = False
			
			with self._collusion_peers_update_names_lock:
				self._collusion_peers_update_names = dict()
			
			self._num_of_malicious_trainers_during_the_round = 0

			super()._elect_new_validators(updater_id_aggregated_validation_scores)

			self._update_committee_majority()
			
		except Exception as e:
			self._logger.record(msg = f"Error while electing the new committee inside a CommitteeBasedRandomNoiseByzantineNode", exc = e, logLevel = diagnostic.ERROR, identifier= self._str_identifier)
			raise e

	def _update_committee_majority(self):
		"""
		Update the committee majority
		"""
		try:
			with self._validators_list_lock:
				collusion_peers_in_committee = [peer for peer in self._validators_list if peer in self._collusion_peers]
				if len(collusion_peers_in_committee) >= self._positive_threshold_to_pass_validation:
					self._committee_majority = True
				else:
					self._committee_majority = False
			
				if self._committee_majority:

					malicious_trainers_during_the_round = list()

					with self._peers_lock:
						for malicious_node_id in self._collusion_peers:
							if malicious_node_id not in self._validators_list:
								if malicious_node_id in self._peers and self._peers[malicious_node_id][pg.PeersListElement.STATUS]:
									self._num_of_malicious_trainers_during_the_round += 1
									malicious_trainers_during_the_round.append(malicious_node_id)
								
								else:
									self._logger.record(msg = f"Malicious node {malicious_node_id} is not alive", logLevel = diagnostic.WARNING, identifier= self._str_identifier)

				if self._allowed_to_write_redudant_log_messages:
					if self.aggregation_round() >= self._starting_round_for_malicious_behaviour and self._malicious_behaviour_only_with_majority_in_committee:
						if self._committee_majority:
							self._logger.record(msg = f"Malicious nodes have the majority in the committee. Number of malicious nodes in committee: {len(collusion_peers_in_committee)}. Number of malicious trainers available: {self._num_of_malicious_trainers_during_the_round}. Malicious trainers: {malicious_trainers_during_the_round}", logLevel = diagnostic.INFO, identifier= self._str_identifier)
						else:
							self._logger.record(msg = f"Malicious nodes have NOT the majority in the committee. Number of malicious nodes in committee: {len(collusion_peers_in_committee)}", logLevel = diagnostic.DEBUG, identifier= self._str_identifier)

		except Exception as e:
			self._logger.record(msg = f"Error while updating the committee majority inside a CommitteeBasedRandomNoiseByzantineNode", exc = e, logLevel = diagnostic.ERROR, identifier= self._str_identifier)
			raise e	

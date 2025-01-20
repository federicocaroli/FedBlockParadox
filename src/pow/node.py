import gc, time, datetime, sys

from multiprocessing.synchronize import BoundedSemaphore as BoundedSemaphoreClass
from threading import Thread

from ..shared import diagnostic, utils
from ..shared.constants import SEED
from ..shared.node import GenericNode
from ..shared.enums import common as cm, peer_generic as pg, archive_generic as ag, archive_messages as shared_am, node_generic as ng, node_messages as shared_nm, peer_messages as pm
from ..shared.block import BlockType

from .enums import archive_messages as am, node_messages as nm
from .block import ModelBlock, ModelBlockInvolvedTrainerFields, GenesisBlock, NodesComputingPowerElemFields

class PowBasedNode(GenericNode):
	
	def __init__(self, logger: diagnostic.Diagnostic, node_id: int, host: str, port: int, dataset_quanta_paths: list, archive_host: str, archive_port: int, lazy_loading: bool, node_computing_power_factor: float, test_set_path: str, val_set_path: str, malicious_training_semaphore: BoundedSemaphoreClass, honest_training_semaphore: BoundedSemaphoreClass, validation_semaphore: BoundedSemaphoreClass, fit_epochs: int = 3, batch_size: int = 32, already_available_peers: dict = {}, allowed_to_write_redudant_log_messages: bool = False, test_set_validation_of_global_model_after_each_round: bool = False, genesis_block = None, is_active_trainer_in_first_round: (bool | None) = None, store_weights_directly_in_archive_tmp_dir: bool = False, archive_tmp_dir: (str | None) = None, prepare_node_for_first_round: bool = True):
		
		if type(node_computing_power_factor) not in [int, float] or type(prepare_node_for_first_round) != bool:
			raise TypeError("PowBasedNode constructor")

		# Pow-specific variables
		self._node_computing_power_factor = node_computing_power_factor
		self._nodes_computing_power_factor = dict()

		# Initialize the GenericNode class	
		super().__init__(logger, node_id, host, port, dataset_quanta_paths, archive_host, archive_port, lazy_loading, test_set_path, val_set_path, malicious_training_semaphore, honest_training_semaphore, validation_semaphore, fit_epochs, batch_size, already_available_peers, allowed_to_write_redudant_log_messages, test_set_validation_of_global_model_after_each_round, is_active_trainer_in_first_round, store_weights_directly_in_archive_tmp_dir, archive_tmp_dir)
		
		if genesis_block is None:
			# Retrieve Genesis informations
			genesis_block = self._download_genesis_block_from_archive(genesis_block_from_json_method= GenesisBlock.from_json)
		
		self._model_architecture = genesis_block.get_model_architecture()
		self._perc_of_nodes_active_in_a_round = genesis_block.get_perc_of_nodes_active_in_a_round()
		self._max_num_of_aggregation_rounds = genesis_block.get_max_num_of_aggregation_rounds()
		
		# Define the node computing power factor
		genesis_nodes_computing_power_factor = genesis_block.get_nodes_computing_power_factor()
		for elem in genesis_nodes_computing_power_factor:
			for node_id in elem[NodesComputingPowerElemFields.NODE_IDS]:
				self._nodes_computing_power_factor[node_id] = elem[NodesComputingPowerElemFields.COMPUTING_POWER_FACTOR]

		with self._peers_lock:
			list_of_available_nodes = [peer_id for peer_id in self._peers if self._peers[peer_id][pg.PeersListElement.STATUS]]

		# Elect the winning miner of the first round
		self._elect_new_validators(latest_block_hash= genesis_block.get_block_hash(), available_nodes_for_next_round= list_of_available_nodes)

		self._is_validator = self._peer_id in self._validators_list							
		self._is_active_trainer_in_first_round = is_active_trainer_in_first_round

		validation_params = genesis_block.get_validation_algorithm_params()
		aggregation_params = genesis_block.get_aggregation_algorithm_params()
		self._validation_mechanism_type, self._validation_mechanism, self._aggregation_strategy_type, self._aggregation_strategy = self._get_val_and_agg_mechanism(validation_params, aggregation_params)

		# I needed to introduce this mechanism to avoid the circular dependency between the PowBasedNode and the MaliciousNode classes
		if prepare_node_for_first_round:
			self.prepare_node_for_the_first_round(genesis_block= genesis_block)

	def prepare_node_for_the_first_round(self, genesis_block = None):
		"""
		Prepare the node for the first round
		"""

		self._logger.record(msg = f"Quanta paths used: {self._dataset_quanta_paths}", logLevel = diagnostic.DEBUG, identifier= self._str_identifier)

		if genesis_block is None:
			# Retrieve Genesis informations
			genesis_block = self._download_genesis_block_from_archive(genesis_block_from_json_method= GenesisBlock.from_json)

		# Check the existance and load the last aggregated model block if any, the most recent weights are not loaded because we will wait for the next round to start the trainings
		recent_aggregated_model_block_found = self._download_and_load_info_from_latest_aggregated_model_block(model_block_from_json_method= ModelBlock.from_json, load_model_weights= False)

		# If a recent aggregated model block is found we don't know the current validators, so we need to wait for the next round to start the training
		if recent_aggregated_model_block_found is False:
			if self.is_validator():
				if self._allowed_to_write_redudant_log_messages:
					self._logger.record(msg = f"Node is the winner miner in next round", logLevel = diagnostic.DEBUG, identifier= self._str_identifier)

			if self._is_active_trainer_in_first_round is True or (self._is_active_trainer_in_first_round is None and self._peer_id in self._define_nodes_active_in_the_next_round(SEED)):
				# Due to the fact that the node is a valid trainer, we load the weights from the genesis block and start the training
				self._model_to_fit_when_not_validator = {ng.ModelToFitDictFields.WEIGHTS: genesis_block.get_model_starting_weights(), ng.ModelToFitDictFields.OPTIMIZER: genesis_block.get_model_starting_optimizer_state()}

		else:
			self._logger.record(msg = f"Recent aggregated model block found, node is waiting the beginning of the next round", logLevel = diagnostic.WARNING, identifier= self._str_identifier)

	def broadcast_and_apply_node_computing_power_factor(self):
		try:
			if self.is_peer_alive() is False:
				raise Exception("Node is not alive")
			
			self._send_message(nm.BroadcastNodeComputingPowerFactor.builder(self._node_computing_power_factor, self._peer_id, self.aggregation_round()))

			if self._peer_id not in self._nodes_computing_power_factor:
				self._nodes_computing_power_factor[self._peer_id] = self._node_computing_power_factor

		except Exception as e:
			self._logger.record(msg = f"Error while broadcasting the node computing power factor", exc = e, logLevel = diagnostic.ERROR, identifier= self._str_identifier)
			raise e

	def run(self):

		# Start the node without blocking the initialization of the socket server

		# Only if the node is not a validator and we are in the first round we need to start the training otherwise we wait for the next round and the definition of the list of nodes active in the next round (possible trainers to be selected)
		if self.aggregation_round() == 1:

			# The node has been manually set as an active trainer in the first round
			if self._is_active_trainer_in_first_round is True:
				self._logger.record(msg = f"Node is an active trainer in the next round. Round: {self.aggregation_round()}. Starting fitting...", logLevel = diagnostic.INFO, identifier= self._str_identifier)
				self._do_training_event.set()
			elif self._is_active_trainer_in_first_round is None:
				# Start the thread to define the nodes active in the first round
				Thread(target= self._define_and_start_nodes_active_in_the_next_round, daemon=True, name=f"{self._node_name} start thread").start()
			
		super().run()
	
	# Implemented because we added a new node message type
	def _handle_custom_msg(self, msg: dict) -> (dict | None):

		try:
			if type(msg) != dict:
				raise TypeError("Invalid message")

			sender_peer_id = msg[pm.GenericPeerMessage.PEER_ID]

			if msg[shared_nm.GenericNodeMessage.TYPE] == nm.PowBasedNodeMessageTypes.BROADCAST_NODE_COMPUTING_POWER_FACTOR:
				if self._allowed_to_write_redudant_log_messages:
					self._logger.record(msg = f"Node computing power factor received. Computing power factor: {msg[nm.BroadcastNodeComputingPowerFactor.COMPUTING_POWER_FACTOR]}. Sender peer id: {sender_peer_id}. Round: {self.aggregation_round()}", logLevel = diagnostic.DEBUG, identifier= self._str_identifier)

				if sender_peer_id not in self._nodes_computing_power_factor:
					self._nodes_computing_power_factor[sender_peer_id] = msg[nm.BroadcastNodeComputingPowerFactor.COMPUTING_POWER_FACTOR]
				
				elif self._nodes_computing_power_factor[sender_peer_id] != msg[nm.BroadcastNodeComputingPowerFactor.COMPUTING_POWER_FACTOR]:
					raise Exception(f"Different node computing power factor already received for that node. Sender peer id: {sender_peer_id}. Computing power factor: {msg[nm.BroadcastNodeComputingPowerFactor.COMPUTING_POWER_FACTOR]}. Expected computing power factor: {self._nodes_computing_power_factor[sender_peer_id]}")

				return None

		except Exception as e:
			self._logger.record(msg = f"Error while handling a custom message inside PowBasedNode class", exc = e, logLevel = diagnostic.ERROR, identifier= self._str_identifier)
			raise e

		# If the message is not a pow-specific message, it will be handled by the generic handle_custom_msg method
		return super()._handle_custom_msg(msg)

	def _define_nodes_active_in_the_next_round(self, seed: int = SEED, base_list_of_nodes: (list | None) = None):
		if type(seed) != int or (base_list_of_nodes is not None and type(base_list_of_nodes) != list):
			raise TypeError("Node _define_nodes_active_in_the_next_round method")
		
		list_of_available_trainers = list()
		num_of_available_nodes = 0

		if base_list_of_nodes is not None:
			list_of_available_trainers = base_list_of_nodes
				
		else:		
			with self._peers_lock:
				for peer_id in self._peers:
					if self._peers[peer_id][pg.PeersListElement.STATUS]:
						list_of_available_trainers.append(peer_id)

			if len(list_of_available_trainers) != self.get_number_of_peers_alive():
				raise Exception(f"Invalid number of available nodes. Available nodes: {len(list_of_available_trainers)}. Expected number of available nodes: {self.get_number_of_peers_alive()}")

		num_of_available_nodes = round(self._perc_of_nodes_active_in_a_round * len(list_of_available_trainers))
		
		return self._get_random_elements_from_list(list_of_available_trainers, num_of_available_nodes, seed)
	
	def _elect_new_validators(self, latest_block_hash: str, available_nodes_for_next_round: list):
		"""
		Elect the new validators based on a random lottery with the stakes of the nodes
		"""
		try:
			if type(latest_block_hash) != str or type(available_nodes_for_next_round) != list:
				raise TypeError("Node _elect_new_validators method")
			elif len(available_nodes_for_next_round) == 0 or len(latest_block_hash) == 0:
				raise ValueError("Node _elect_new_validators method")
						
			lottery_tickets = dict()
			count_of_sold_tickets = 0
			max_num_of_decimals = 0
			numerical_factor_to_convert_compute_power_factor_to_int = 1
			
			with self._validators_list_lock:
				self._is_validator = False
				self._validators_list = []

				for node_id in available_nodes_for_next_round:
					if node_id not in self._nodes_computing_power_factor:
						raise Exception(f"Node id not in the computing power factor table. Node id: {node_id}")
					
					# Compute the max number of decimals
					if int(self._nodes_computing_power_factor[node_id]) != self._nodes_computing_power_factor[node_id]:				# If the number is not an integer (1.0 is considered integer)
						num_of_decimals = len(str(self._nodes_computing_power_factor[node_id]).split(".")[1])
						if num_of_decimals > max_num_of_decimals:
							max_num_of_decimals = num_of_decimals

				numerical_factor_to_convert_compute_power_factor_to_int = 10 ** max_num_of_decimals
					
				for node_id in available_nodes_for_next_round:
					tmp_count_of_sold_tickets = self._nodes_computing_power_factor[node_id] * numerical_factor_to_convert_compute_power_factor_to_int

					if tmp_count_of_sold_tickets != int(tmp_count_of_sold_tickets):
						raise Exception(f"Invalid integer number of tickets. Node id: {node_id}. Tickets: {tmp_count_of_sold_tickets}")
					
					tmp_count_of_sold_tickets = int(tmp_count_of_sold_tickets)

					lottery_tickets[node_id] = (count_of_sold_tickets, count_of_sold_tickets + tmp_count_of_sold_tickets)
					count_of_sold_tickets += tmp_count_of_sold_tickets

				if self._allowed_to_write_redudant_log_messages:
					self._logger.record(msg = f"Lottery tickets sold. Tickets: {lottery_tickets}. Round: {self.aggregation_round()}", logLevel= diagnostic.DEBUG, identifier= self._str_identifier)
				
				# Get single integer from last 8 bytes of the hash
				winner_idx = int(latest_block_hash[-16:], 16) % count_of_sold_tickets
				
				if self._allowed_to_write_redudant_log_messages:
					self._logger.record(msg = f"Winner ticket computed. Winner ticket: {winner_idx}. Round: {self.aggregation_round()}", logLevel= diagnostic.DEBUG, identifier= self._str_identifier)
				
				for node_id, tickets in lottery_tickets.items():
					if tickets[0] <= winner_idx < tickets[1]:
						self._validators_list.append(node_id)
						break

				if self._allowed_to_write_redudant_log_messages:
					self._logger.record(msg = f"Winner miner of this round elected. Winner miner: {self._validators_list}. Round: {self.aggregation_round()}", logLevel= diagnostic.DEBUG, identifier= self._str_identifier)

				if self._peer_id in self._validators_list:
					self._is_validator = True
					if self._allowed_to_write_redudant_log_messages:
						self._logger.record(msg = f"Node is going to be the winner miner in this round. Round: {self.aggregation_round()}", logLevel = diagnostic.DEBUG, identifier= self._str_identifier)

		except Exception as e:
			self._logger.record(msg = f"Error while electing the new miner", exc = e, logLevel = diagnostic.ERROR, identifier= self._str_identifier)
			raise e
	
	def _prepare_node_for_next_round(self, aggregated_model_block_hash: str, aggregated_model_weights: (list | None) = None, aggregated_model_optimizer_variables: (dict | None) = None, available_nodes_for_next_round: list = None, updater_ids_update_names_aggregated_validation_scores: list = None):
		"""
		Prepare the node for the next round
		"""

		try:
			if type(aggregated_model_block_hash) != str or (type(aggregated_model_weights) != list and aggregated_model_weights is not None) or (type(available_nodes_for_next_round) != list and available_nodes_for_next_round is not None) or (type(updater_ids_update_names_aggregated_validation_scores) != list and updater_ids_update_names_aggregated_validation_scores is not None) or (type(aggregated_model_optimizer_variables) != dict and aggregated_model_optimizer_variables is not None):
				raise TypeError("Node _prepare_node_for_next_round method")

			if self._model_aggregation_in_progress.is_set() is False:
				self._aggregation_round_during_aggregation = self.aggregation_round()
				self._model_aggregation_in_progress.set()

			# Waiting for the previous messages to be handled
			while True:
				with self._num_of_messages_being_handled_lock:
					# 1 because the current message is being handled
					if self._num_of_messages_being_handled <= 1:
						break
					else:
						if self._allowed_to_write_redudant_log_messages:
							self._logger.record(msg = f"Waiting for the current message to be handled. Num of messages being handled: {self._num_of_messages_being_handled}", logLevel = diagnostic.DEBUG, identifier= self._str_identifier)
				
				time.sleep(5)
			
			# A new round is going to start stop the training and clear the used variables
			self._clear_training_and_validation_variables()

			stop_training = False

			with self._aggregation_round_lock:
				if self._aggregation_round >= self._max_num_of_aggregation_rounds:
					stop_training = True

					if self._allowed_to_write_redudant_log_messages:
						self._logger.record(msg = f"End of the simulation. Last round number: {self._aggregation_round}", logLevel = diagnostic.DEBUG, identifier= self._str_identifier)

				else:
					self._aggregation_round += 1

			if stop_training is False:

				last_aggregated_model_block = None
				
				if available_nodes_for_next_round is None or updater_ids_update_names_aggregated_validation_scores is None:
					# Download the last aggregated model block
					block_json, block_type = self._download_blockchain_block_from_archive(aggregated_model_block_hash)

					if block_type != BlockType.MODEL:
						raise Exception(f"Block type is not a model block. Received block type: {block_type}")

					last_aggregated_model_block = ModelBlock.from_json(block_json)

					# Store the latest model filename
					self._latest_model_filename = last_aggregated_model_block.get_global_model_name()

					# Get the list of available nodes for the next round
					available_nodes_for_next_round = last_aggregated_model_block.get_available_nodes()

					# Get the list of updater ids, update names and aggregated validation scores
					updater_ids_update_names_aggregated_validation_scores = last_aggregated_model_block.get_involved_trainers()

					# Get the list of nodes computing power factor
					self._nodes_computing_power_factor = dict()

					for elem in last_aggregated_model_block.get_nodes_computing_power_factor():
						for node_id in elem[NodesComputingPowerElemFields.NODE_IDS]:
							self._nodes_computing_power_factor[node_id] = elem[NodesComputingPowerElemFields.COMPUTING_POWER_FACTOR]

				list_of_honest_updates_aggregated_validation_scores = list()
				
				for elem in updater_ids_update_names_aggregated_validation_scores:
					list_of_honest_updates_aggregated_validation_scores.append(elem[ModelBlockInvolvedTrainerFields.UPDATE_SCORE])

				# Elect the new validators based on the stakes
				self._elect_new_validators(latest_block_hash= aggregated_model_block_hash, available_nodes_for_next_round= available_nodes_for_next_round)

				# Reset the validation mechanism
				if self._validation_mechanism_type in cm.GradientsBasedValidationAlgorithmType.list():
					with self._validators_list_lock:
						self._validation_mechanism.start_new_round(self._validators_list)
				
				elif self._validation_mechanism_type in cm.WeightsBasedValidationAlgorithmType.list():
					with self._validators_list_lock:
						self._validation_mechanism.start_new_round(self._validators_list, list_of_honest_updates_aggregated_validation_scores)

						if self._allowed_to_write_redudant_log_messages:
							if self._validation_mechanism_type in [cm.WeightsBasedValidationAlgorithmType.LOCAL_DATASET_VALIDATION, cm.WeightsBasedValidationAlgorithmType.GLOBAL_DATASET_VALIDATION]:
								median_score = self._validation_mechanism.get_min_update_validation_score()
								self._logger.record(msg = f"Median score for the next round computed. Median score: {median_score}. List of scores: {list_of_honest_updates_aggregated_validation_scores}. Round: {self.aggregation_round()}", logLevel = diagnostic.DEBUG, identifier= self._str_identifier)

				else:
					raise NotImplementedError("Validation mechanism not implemented")

				# We need to check if the node is an active trainer in the next round, in that case we need to load the newest model weights and start the training
				# A node can be both a validator and a trainer
					
				seed_for_random_operations = sum(self._validators_list) + self.aggregation_round()

				if self._peer_id in self._define_nodes_active_in_the_next_round(seed_for_random_operations, available_nodes_for_next_round):
					if aggregated_model_weights is None:					
						if self._allowed_to_store_weights_directly_in_archive_tmp_dir:
							self._get_authorization_to_read_model_weights_directly_from_file(self._latest_model_filename)
							model_info = self._read_model_from_disk(self._latest_model_filename)

						else:
							model_info = self._download_model_from_archive(self._latest_model_filename)
						
						aggregated_model_weights = model_info[ag.AggregatedModel.WEIGHTS]
						aggregated_model_optimizer_variables = model_info[ag.AggregatedModel.OPTIMIZER]

						del model_info
					
					with self._model_lock:
						self._model_to_fit_when_not_validator = {ng.ModelToFitDictFields.WEIGHTS: aggregated_model_weights, ng.ModelToFitDictFields.OPTIMIZER: aggregated_model_optimizer_variables}

					del aggregated_model_weights

				del last_aggregated_model_block

				self._wait_to_start_next_round()

				self._define_and_start_nodes_active_in_the_next_round(seed_for_random_operations, available_nodes_for_next_round)
				
			else:
				self.stop()

			# A new model block has been uploaded
			self._model_aggregation_in_progress.clear()

		except Exception as e:
			self._logger.record(msg = f"Error while preparing the node for the next round", exc = e, logLevel = diagnostic.ERROR, identifier= self._str_identifier)
			raise e
		finally:
			gc.collect()

	def _create_and_upload_aggregated_model_block(self, update_names_and_aggregated_scores: dict):
		
		try:
			updates_weights_or_gradients = []
			updater_ids_update_names_aggregated_validation_scores = []
			
			if type(update_names_and_aggregated_scores) != dict:
				raise TypeError("Node _create_and_upload_aggregated_model_block method")

			update_names = list(update_names_and_aggregated_scores.keys())

			# Download the updates
			for update_name in update_names:
				if self._allowed_to_store_weights_directly_in_archive_tmp_dir:
					self._get_authorization_to_read_update_directly_from_file(update_name)	# Archive will confirm the authorization to read the update directly from the file
					updater_id, weights, num_of_samples = self._read_update_from_disk(update_name)

				else:
					# Validate the update
					updater_id, weights, num_of_samples = self._download_update_from_archive(update_name)

				updates_weights_or_gradients.append((weights, num_of_samples))
				updater_ids_update_names_aggregated_validation_scores.append({ModelBlockInvolvedTrainerFields.NODE_ID: updater_id, ModelBlockInvolvedTrainerFields.UPDATE_NAME: update_name, ModelBlockInvolvedTrainerFields.UPDATE_SCORE: update_names_and_aggregated_scores[update_name]})
			
			# Aggregate the updates and upload the model
			if self._validation_mechanism_type in cm.GradientsBasedValidationAlgorithmType.list():
				
				try:
					gradient_list = [grad for grad, _ in updates_weights_or_gradients]

					if self._aggregation_strategy_type in cm.GradientsBasedAggregationAlgorithmType.list():
							global_gradients = self._aggregation_strategy.aggregate(gradient_list) 
					elif self._aggregation_strategy_type in cm.WeightsBasedAggregationAlgorithmType.list():
						raise NotImplementedError(f"Weights based aggregation strategy {self._aggregation_strategy_type} not implemented for the gradients based validation mechanism {self._validation_mechanism_type}")
					else:
						raise ValueError(f"Invalid aggregation strategy type. Aggregation strategy type: {self._aggregation_strategy_type}")
				
				except Exception as e:
					self._logger.record(msg = f"Error while aggregating the gradients in _create_and_upload_aggregated_model_block with validation - {self._validation_mechanism_type} - and aggregation - {self._aggregation_strategy_type}", exc = e, logLevel = diagnostic.ERROR, identifier= self._str_identifier)
					raise e
				
				# I have to download the last model and update it with the new gradients
				if self._latest_model_filename is None:
					if self.aggregation_round() != 1:
						raise Exception(f"Latest model filename is None, but the aggregation round is different from 1. Aggregation round: {self.aggregation_round()}")
					
					# Download the genesis block because the node is in the first round
					genesis_block = self._download_genesis_block_from_archive(genesis_block_from_json_method= GenesisBlock.from_json)
					model_weights = genesis_block.get_model_starting_weights()
					optimizer_variables = genesis_block.get_model_starting_optimizer_state()

					if optimizer_variables is not None:
						self._logger.record(msg = f"Starting optimizer state found inside the genesis block. It is going to be used during the first round aggregation", logLevel = diagnostic.DEBUG, identifier= self._str_identifier)
					else:
						self._logger.record(msg = f"Starting optimizer state not found inside the genesis block. The optimizer will be randomly initialized during the first round aggregation", logLevel = diagnostic.DEBUG, identifier= self._str_identifier)

					del genesis_block

				else:
					if self._allowed_to_store_weights_directly_in_archive_tmp_dir:
						self._get_authorization_to_read_model_weights_directly_from_file(self._latest_model_filename)
						model_info = self._read_model_from_disk(self._latest_model_filename)

					else:
						model_info = self._download_model_from_archive(self._latest_model_filename)
					
					model_weights = model_info[ag.AggregatedModel.WEIGHTS]
					optimizer_variables = model_info[ag.AggregatedModel.OPTIMIZER]

					if optimizer_variables is None:
						raise Exception("Optimizer variables are None but the node is not in the first round and gradients are used")

					del model_info
				
				model = utils.build_model_from_architecture_and_weights(self._model_architecture, model_weights, global_gradients, optimizer_variables)
				global_weights = model.get_weights()
				global_weights = [arr.tolist() for arr in global_weights]
				
				new_optimizer_variables = dict()
				model.optimizer.save_own_variables(new_optimizer_variables)
				new_optimizer_variables = {key: arr.tolist() for key, arr in new_optimizer_variables.items()}
				
				del model, model_weights, global_gradients

			elif self._validation_mechanism_type in cm.WeightsBasedValidationAlgorithmType.list():
				new_optimizer_variables = None
				
				try:
					if self._aggregation_strategy_type in [cm.AggregationAlgorithmType.FEDAVG]:
						global_weights = self._aggregation_strategy.aggregate(updates_weights_or_gradients)

					elif self._aggregation_strategy_type in [cm.AggregationAlgorithmType.MEAN, cm.AggregationAlgorithmType.MEDIAN, cm.AggregationAlgorithmType.TRIMMED_MEAN]:
						only_weights_list = [weights for weights, _ in updates_weights_or_gradients]
						global_weights = self._aggregation_strategy.aggregate(only_weights_list)

					else:
						raise ValueError(f"Invalid aggregation strategy type. Aggregation strategy type: {self._aggregation_strategy_type}")
				
				except Exception as e:
					self._logger.record(msg = f"Error while aggregating the weights in _create_and_upload_aggregated_model_block with validation - {self._validation_mechanism_type} - and aggregation - {self._aggregation_strategy_type}", exc = e, logLevel = diagnostic.ERROR, identifier= self._str_identifier)
					raise e

			else:
				raise NotImplementedError("Validation mechanism not implemented")

			del updates_weights_or_gradients 
			gc.collect()

			# Save the new aggregated model
			model_name = self._create_weights_filename()

			if self._allowed_to_store_weights_directly_in_archive_tmp_dir:
				request = shared_am.GenericArchiveRequest.builder(shared_am.ArchiveRequestTypes.AUTHORIZATION_TO_STORE_AGGREGATED_MODEL_DIRECTLY, shared_am.AuthorizationToStoreAggregatedModelDirectlyRequestBody.builder(self._peer_id, model_name), self.aggregation_round())
			else:
				request = shared_am.GenericArchiveRequest.builder(shared_am.ArchiveRequestTypes.UPLOAD_AGGREGATED_MODEL, shared_am.UploadAggregatedModelRequestBody.builder(self._peer_id, model_name, global_weights, new_optimizer_variables), self.aggregation_round())
			
			message_to_send = self._prepare_socket_msg_to_send(request)
			del request
			gc.collect()

			response = self._handle_request_with_archive(message_to_send)
			del message_to_send
			gc.collect()

			if response is None:
				raise Exception("Archive response is None")
			elif response[shared_am.GenericArchiveResponse.SUBTYPE] == shared_am.ArchiveResponseTypes.GENERIC_ERROR:
				raise Exception(f"Archive response is an error message. Response: {response}")
			elif response[shared_am.GenericArchiveResponse.SUBTYPE] == shared_am.ArchiveResponseTypes.INVALID_ROUND_NUMBER:
				self._logger.record(msg = f"Invalid round number. Request type: {shared_am.ArchiveRequestTypes.UPLOAD_AGGREGATED_MODEL}. Message round: {response[shared_am.GenericArchiveResponse.ROUND]}. Node round: {self.aggregation_round()}", logLevel = diagnostic.WARNING, identifier= self._str_identifier)
				raise ng.InvalidRoundException()
			elif response[shared_am.GenericArchiveResponse.SUBTYPE] != shared_am.ArchiveResponseTypes.GENERIC_SUCCESS:
				raise ValueError(f"Invalid response type. Response: {response}")

			if self._allowed_to_store_weights_directly_in_archive_tmp_dir:
				self._store_model_on_disk(model_name, global_weights, new_optimizer_variables)
				self._logger.record(msg = f"New aggregated model written directly on disk. Model name: {model_name}. Round: {self.aggregation_round()}", logLevel = diagnostic.INFO, identifier= self._str_identifier)
			else:
				self._logger.record(msg = f"New aggregated model uploaded. Model name: {model_name}. Round: {self.aggregation_round()}", logLevel = diagnostic.INFO, identifier= self._str_identifier)

			if self._test_set_validation_of_global_model_after_each_round:
				acc_on_test_set = self._test(global_weights)

				self._logger.record(msg = f"Accuracy of the new aggregated model on the test set computed. Accuracy: {acc_on_test_set}. Round: {self.aggregation_round()}", logLevel = diagnostic.INFO, identifier= self._str_identifier)

			# Define the list of available nodes for the next round and define the list of nodes computing power factor
			# If a node is available but it has not broadcasted its computing power factor, it will not be considered available. The list of computing power factor will contain the computing power factor of all the existing and existed nodes
			list_of_available_nodes_for_next_round = list()
			list_of_nodes_computing_power_factor = list()
			tmp_dict_of_nodes_computing_power_factor = dict()
			
			with self._peers_lock:
				for peer_id in self._peers:
					if self._peers[peer_id][pg.PeersListElement.STATUS]:
						if peer_id not in self._nodes_computing_power_factor:
							self._logger.record(msg = f"Node computing power factor not found for an available node. Node will not be considered available. Node id: {peer_id}", logLevel = diagnostic.WARNING, identifier= self._str_identifier)
							continue

						list_of_available_nodes_for_next_round.append(peer_id)
							
			for node_id in self._nodes_computing_power_factor:
				if self._nodes_computing_power_factor[node_id] not in tmp_dict_of_nodes_computing_power_factor:
					tmp_dict_of_nodes_computing_power_factor[self._nodes_computing_power_factor[node_id]] = []
							
				tmp_dict_of_nodes_computing_power_factor[self._nodes_computing_power_factor[node_id]].append(node_id)
						
			for computing_power_factor in tmp_dict_of_nodes_computing_power_factor:
				list_of_nodes_computing_power_factor.append({NodesComputingPowerElemFields.COMPUTING_POWER_FACTOR: computing_power_factor, NodesComputingPowerElemFields.NODE_IDS: tmp_dict_of_nodes_computing_power_factor[computing_power_factor]})

			request = shared_am.GenericArchiveRequest.builder(shared_am.ArchiveRequestTypes.UPLOAD_BLOCKCHAIN_AGGREGATED_MODEL, am.UploadBlockchainAggregatedModelRequestBody.builder(self._peer_id, model_name, self.aggregation_round(), time.time(), updater_ids_update_names_aggregated_validation_scores, self._validators_list, list_of_available_nodes_for_next_round, list_of_nodes_computing_power_factor), self.aggregation_round())
			response = self._handle_request_with_archive(self._prepare_socket_msg_to_send(request))
			
			if response is None:
				raise Exception("Archive response is None")
			elif response[shared_am.GenericArchiveResponse.SUBTYPE] == shared_am.ArchiveResponseTypes.GENERIC_ERROR:
				raise Exception(f"Archive response is an error message. Response: {response}")
			elif response[shared_am.GenericArchiveResponse.SUBTYPE] == shared_am.ArchiveResponseTypes.INVALID_ROUND_NUMBER:
				self._logger.record(msg = f"Invalid round number. Request type: {shared_am.ArchiveRequestTypes.UPLOAD_BLOCKCHAIN_AGGREGATED_MODEL}. Message round: {response[shared_am.GenericArchiveResponse.ROUND]}. Node round: {self.aggregation_round()}", logLevel = diagnostic.WARNING, identifier= self._str_identifier)
				raise ng.InvalidRoundException()
			elif response[shared_am.GenericArchiveResponse.SUBTYPE] != shared_am.ArchiveResponseTypes.UPLOAD_BLOCKCHAIN_BLOCK:
				raise ValueError(f"Invalid response type. Response: {response}")

			self._latest_model_filename = model_name
			agg_block_hash = response[shared_am.GenericArchiveResponse.BODY][shared_am.UploadBlockchainBlockResponseBody.BLOCK_HASH]

			self._logger.record(msg = f"New aggregated model block uploaded. Block hash: {agg_block_hash}. Model filename: {model_name}", logLevel = diagnostic.INFO, identifier= self._str_identifier)
			self._logger.record(msg = f"New aggregated model block uploaded. List of available nodes for the next round: {list_of_available_nodes_for_next_round}", logLevel = diagnostic.DEBUG, identifier= self._str_identifier)
			self._logger.record(msg = f"New aggregated model block uploaded. List of nodes computing power factor: {list_of_nodes_computing_power_factor}", logLevel= diagnostic.DEBUG, identifier= self._str_identifier)

			print(f"\n{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')} New aggregated model block uploaded. Block hash: {agg_block_hash}. Model filename: {model_name}. Round: {self.aggregation_round()}\n", file= sys.stderr)

			self._send_message(shared_nm.BroadcastNewModelBlockAlert.builder(agg_block_hash, self._peer_id, self.aggregation_round()))

			self._prepare_node_for_next_round(aggregated_model_block_hash= agg_block_hash, aggregated_model_weights= global_weights, aggregated_model_optimizer_variables= new_optimizer_variables, available_nodes_for_next_round= list_of_available_nodes_for_next_round, updater_ids_update_names_aggregated_validation_scores= updater_ids_update_names_aggregated_validation_scores)
		
		except ng.InvalidRoundException as e:
			raise e
		except Exception as e:
			self._logger.record(msg = f"Error while aggregating the updates and uploading the model", exc = e, logLevel = diagnostic.ERROR, identifier= self._str_identifier)
			raise e

	def _create_and_upload_peer_update_thread(self):
		"""
		Train the model and upload the update to the archive in a iterative way
		"""
		while self._do_training_event.wait():

			try:
				self._do_training_event.clear()

				if self.is_peer_alive() is False:
					break
				
				with self._model_lock:
					if self._model_to_fit_when_not_validator is None:
						if self._model_aggregation_in_progress.is_set():
							if self._allowed_to_write_redudant_log_messages:
								self._logger.record(msg = f"Model aggregation is in progress. Trainer skips creating the next node update", logLevel = diagnostic.DEBUG, identifier= self._str_identifier)
							
							continue

						raise Exception("Model is None, but training is requested")

				weights = None
				num_samples_training_set = None
				current_round = self.aggregation_round()

				result = self._fit()

				if result is not None:
					weights, num_samples_training_set = result

				with self._model_lock:
					self._model_to_fit_when_not_validator = None
				
				gc.collect()

				if weights is not None and self._model_aggregation_in_progress.is_set() is False and self.is_peer_alive() and current_round == self.aggregation_round():
					name = self._create_weights_filename()

					# Node is allowed to store the weights directly in the archive temporary directory
					if self._allowed_to_store_weights_directly_in_archive_tmp_dir:
						request = shared_am.GenericArchiveRequest.builder(shared_am.ArchiveRequestTypes.AUTHORIZATION_TO_STORE_UPDATE_DIRECTLY, shared_am.AuthorizationToStoreUpdateDirectlyRequestBody.builder(self._peer_id, name, num_samples_training_set), current_round)
					# Node is not allowed to store the weights directly in the archive temporary directory and, so, the Archive will store the weights
					else:
						request = shared_am.GenericArchiveRequest.builder(shared_am.ArchiveRequestTypes.UPLOAD_PEER_UPDATE, shared_am.UploadPeerUpdateRequestBody.builder(self._peer_id, name, weights, num_samples_training_set), current_round)
					
					message_to_send = self._prepare_socket_msg_to_send(request)
					del request
					gc.collect()

					response = self._handle_request_with_archive(message_to_send)
					del message_to_send
					gc.collect()
					
					if response is None:
						raise Exception("Archive response is None")
					elif response[shared_am.GenericArchiveResponse.SUBTYPE] == shared_am.ArchiveResponseTypes.GENERIC_ERROR:
						raise Exception(f"Archive response is an error message. Response: {response}")
					elif response[shared_am.GenericArchiveResponse.SUBTYPE] == shared_am.ArchiveResponseTypes.INVALID_ROUND_NUMBER:
						self._logger.record(msg = f"Invalid round number. Request type: {shared_am.ArchiveRequestTypes.UPLOAD_PEER_UPDATE}. Message round: {response[shared_am.GenericArchiveResponse.ROUND]}. Node round: {current_round}", logLevel = diagnostic.WARNING, identifier= self._str_identifier)
						continue
					elif response[shared_am.GenericArchiveResponse.SUBTYPE] != shared_am.ArchiveResponseTypes.GENERIC_SUCCESS:
						raise ValueError(f"Invalid response type. Response: {response}")

					if self._allowed_to_store_weights_directly_in_archive_tmp_dir:
						self._store_update_on_disk(name, weights, num_samples_training_set)
						self._logger.record(msg = f"Update written directly on disk. Update name: {name}. Round: {current_round}", logLevel = diagnostic.INFO, identifier= self._str_identifier)
					else:
						self._logger.record(msg = f"Update uploaded. Update name: {name}. Round: {current_round}", logLevel = diagnostic.INFO, identifier= self._str_identifier)

					self._send_message(shared_nm.BroadcastUpdateUpload.builder(name, self._peer_id, current_round))
					
					# If the node is a validator, the update must be validated otherwise it must only be broadcasted
					if self.is_validator():
						self._validate_update_created_by_node(name, self._peer_id, weights)

				else:
					self._logger.record(msg = f"Update neither uploaded to the archive or broadcasted to other peers because model aggregation is in progress", logLevel = diagnostic.DEBUG, identifier= self._str_identifier)
					
				del weights
				gc.collect()

			except Exception as e:
				self._logger.record(msg = f"Error while creating the next node update", exc = e, logLevel = diagnostic.ERROR, identifier= self._str_identifier)

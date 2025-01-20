import os, json, time, numpy as np, sys, gc, setproctitle

from multiprocessing import Process, Event, current_process, BoundedSemaphore
from multiprocessing.managers import BaseManager
from multiprocessing.synchronize import Event as EventClass, BoundedSemaphore as BoundedSemaphoreClass

from ..shared import diagnostic, plot_utils as pu
from ..shared.enums import common as cm, validation_algos as va, peer_generic as pg
from ..shared.utils import configure_gpus, start_archive
from ..shared.dataset_utils import select_quanta_paths
from ..shared.constants import SEED

from .enums.common import AdditionalCommitteeSpecificConfigParams
from .constants import DEFAULT_COMMITTEE_SPECIFIC_CONFIG
from .block import GenesisBlock
from .archive import CommitteeBasedArchive
from .node import CommitteeBasedNode
from .attacker_nodes import CommitteeBasedLabelFlippingNode, CommitteeBasedAdditiveNoiseByzantineNode, CommitteeBasedRandomLabelByzantineNode, CommitteeBasedRandomNoiseByzantineNode, CommitteeBasedTargetedPoisoningNode
 
_MODULE = "Committee-Main"

class CustomManager(BaseManager):
	pass

def validate_committee_consensus_specific_config(stored_config: dict, run_time_only_params: dict) -> None:
	'''
	Define the committee specific running configuration from the stored configuration.
	'''

	try:
		if type(stored_config) != dict or type(run_time_only_params) != dict:
			raise TypeError("validate_committee_consensus_specific_config")
		elif cm.ConfigParams.CONSENSUS_PARAMS not in stored_config:
			raise KeyError("validate_committee_consensus_specific_config. Consensus params not in stored configuration")
		elif type(stored_config[cm.ConfigParams.CONSENSUS_PARAMS]) != dict:
			raise TypeError("validate_committee_consensus_specific_config. Consensus params is not a dictionary")
		elif cm.ConfigGenericConsensusAlgorithmParams.TYPE not in stored_config[cm.ConfigParams.CONSENSUS_PARAMS] or stored_config[cm.ConfigParams.CONSENSUS_PARAMS][cm.ConfigGenericConsensusAlgorithmParams.TYPE] != cm.ConsensusAlgorithmType.COMMITTEE:
			raise ValueError("validate_committee_consensus_specific_config. Consensus algorithm type is not committee specific")
		elif any(key not in run_time_only_params for key in cm.RunTimeOnlyParams.list()):
			raise KeyError("validate_committee_consensus_specific_config. Run time only params missing key")

		is_support_machine_config = not stored_config[cm.ConfigParams.IS_MAIN_SIMULATION]

		consensus_specific_config = stored_config[cm.ConfigParams.CONSENSUS_PARAMS]

		for key in AdditionalCommitteeSpecificConfigParams.list():
			if key not in consensus_specific_config:
				raise KeyError(f"validate_committee_consensus_specific_config. {key} not in stored configuration")

		# List of active trainers in the first round is required only for the support machine because if in the main simulation it is not specified then the main simulation will pick the active trainers randomly
		if is_support_machine_config is True:
			if consensus_specific_config[AdditionalCommitteeSpecificConfigParams.LIST_OF_ACTIVE_TRAINERS_IN_THE_FIRST_ROUND] is None:
				raise ValueError("validate_committee_consensus_specific_config. List of active trainers in the first round is None but it is required for the support machine")

		if consensus_specific_config[AdditionalCommitteeSpecificConfigParams.LIST_OF_ACTIVE_TRAINERS_IN_THE_FIRST_ROUND] is not None:
			if type(consensus_specific_config[AdditionalCommitteeSpecificConfigParams.LIST_OF_ACTIVE_TRAINERS_IN_THE_FIRST_ROUND]) != list:
				raise TypeError("validate_committee_consensus_specific_config. List of active trainers in the first round is not a list")
			elif any(type(elem) != int or elem < 0 for elem in consensus_specific_config[AdditionalCommitteeSpecificConfigParams.LIST_OF_ACTIVE_TRAINERS_IN_THE_FIRST_ROUND]):
				raise TypeError("validate_committee_consensus_specific_config. List of active trainers in the first round contains an element that is not an integer or is less than zero")
			
			for node_id in consensus_specific_config[AdditionalCommitteeSpecificConfigParams.LIST_OF_ACTIVE_TRAINERS_IN_THE_FIRST_ROUND]:
				if any(node_id == node[cm.NodeSpecificConfigParams.NODE_ID] for node in run_time_only_params[cm.RunTimeOnlyParams.NODES]) is False:
					raise ValueError("validate_committee_consensus_specific_config. List of active trainers in the first round contains a node id that is not part of the simulation")

		if is_support_machine_config is False:
			# Number of validators
			if consensus_specific_config[AdditionalCommitteeSpecificConfigParams.NUM_OF_VALIDATORS] is None:
				raise ValueError("validate_committee_consensus_specific_config. Number of validators is None")
			elif type(consensus_specific_config[AdditionalCommitteeSpecificConfigParams.NUM_OF_VALIDATORS]) != int:
				raise TypeError("validate_committee_consensus_specific_config. Number of validators is not an integer")
			elif consensus_specific_config[AdditionalCommitteeSpecificConfigParams.NUM_OF_VALIDATORS] < 0:
				raise ValueError("validate_committee_consensus_specific_config. Number of validators is less than zero")
			elif consensus_specific_config[AdditionalCommitteeSpecificConfigParams.NUM_OF_VALIDATORS] > len(run_time_only_params[cm.RunTimeOnlyParams.NODES]):
				raise ValueError("validate_committee_consensus_specific_config. Number of validators is greater than the number of nodes in the simulation")

			# List of node ids of first validators
			if consensus_specific_config[AdditionalCommitteeSpecificConfigParams.LIST_OF_NODE_IDS_OF_FIRST_VALIDATORS] is not None:
				if type(consensus_specific_config[AdditionalCommitteeSpecificConfigParams.LIST_OF_NODE_IDS_OF_FIRST_VALIDATORS]) != list:
					raise TypeError("validate_committee_consensus_specific_config. List of node ids of first validators is not a list")
				elif any(type(elem) != int or elem < 0 for elem in consensus_specific_config[AdditionalCommitteeSpecificConfigParams.LIST_OF_NODE_IDS_OF_FIRST_VALIDATORS]):
					raise TypeError("validate_committee_consensus_specific_config. List of node ids of first validators contains an element that is not an integer or is less than zero")
				elif len(consensus_specific_config[AdditionalCommitteeSpecificConfigParams.LIST_OF_NODE_IDS_OF_FIRST_VALIDATORS]) != consensus_specific_config[AdditionalCommitteeSpecificConfigParams.NUM_OF_VALIDATORS]:
					raise ValueError("validate_committee_consensus_specific_config. Number of node ids of first validators is different from the number of validators")
				
				for node_id in consensus_specific_config[AdditionalCommitteeSpecificConfigParams.LIST_OF_NODE_IDS_OF_FIRST_VALIDATORS]:
					if consensus_specific_config[AdditionalCommitteeSpecificConfigParams.LIST_OF_ACTIVE_TRAINERS_IN_THE_FIRST_ROUND] is not None:
						if node_id in consensus_specific_config[AdditionalCommitteeSpecificConfigParams.LIST_OF_ACTIVE_TRAINERS_IN_THE_FIRST_ROUND]:
							raise ValueError("validate_committee_consensus_specific_config. Node id of first validators is also part of the list of active trainers in the first round")
						
					if any(node_id == node[cm.NodeSpecificConfigParams.NODE_ID] for node in run_time_only_params[cm.RunTimeOnlyParams.NODES]) is False:
						raise ValueError("validate_committee_consensus_specific_config. List of node ids of first validators contains a node id that is not part of the simulation")

			# Threshold to pass validation
			if consensus_specific_config[AdditionalCommitteeSpecificConfigParams.PERC_THRESHOLD_TO_PASS_VALIDATION] is None:
				raise ValueError("validate_committee_consensus_specific_config. Threshold to pass validation is None")
			elif type(consensus_specific_config[AdditionalCommitteeSpecificConfigParams.PERC_THRESHOLD_TO_PASS_VALIDATION]) not in [float, int]:
				raise TypeError("validate_committee_consensus_specific_config. Threshold to pass validation is not a float or int")
			elif consensus_specific_config[AdditionalCommitteeSpecificConfigParams.PERC_THRESHOLD_TO_PASS_VALIDATION] < 0.0 or consensus_specific_config[AdditionalCommitteeSpecificConfigParams.PERC_THRESHOLD_TO_PASS_VALIDATION] > 1.0:
				raise ValueError("validate_committee_consensus_specific_config. Threshold to pass validation is not between 0 and 1")
		
			# Percentage of trainers active in each round
			if consensus_specific_config[AdditionalCommitteeSpecificConfigParams.PERC_OF_TRAINERS_ACTIVE_IN_EACH_ROUND] is None:
				raise ValueError("validate_committee_consensus_specific_config. Percentage of trainers active in each round is None")
			elif type(consensus_specific_config[AdditionalCommitteeSpecificConfigParams.PERC_OF_TRAINERS_ACTIVE_IN_EACH_ROUND]) not in [float, int]:
				raise TypeError("validate_committee_consensus_specific_config. Percentage of trainers active in each round is not a float or int")
			elif consensus_specific_config[AdditionalCommitteeSpecificConfigParams.PERC_OF_TRAINERS_ACTIVE_IN_EACH_ROUND] < 0.0 or consensus_specific_config[AdditionalCommitteeSpecificConfigParams.PERC_OF_TRAINERS_ACTIVE_IN_EACH_ROUND] > 1.0:
				raise ValueError("validate_committee_consensus_specific_config. Percentage of trainers active in each round is not between 0 and 1")
			
			# I don't check if 'num_of_trainers_in_each_round' is equal to the length of 'list_of_active_trainers_in_the_first_round' because it is possible that the number of active trainers in the first round is different from the expected number of active trainers in the first round. That may happen when there are support machines in the simulation
			num_of_trainers_in_each_round = round(consensus_specific_config[AdditionalCommitteeSpecificConfigParams.PERC_OF_TRAINERS_ACTIVE_IN_EACH_ROUND] * (len(run_time_only_params[cm.RunTimeOnlyParams.NODES]) - consensus_specific_config[AdditionalCommitteeSpecificConfigParams.NUM_OF_VALIDATORS]))

			#
			# Additional checks based on the specific validation algorithm
			#

			# Local dataset validation
			if stored_config[cm.ConfigParams.VALIDATION_PARAMS][cm.ConfigGenericConsensusAlgorithmParams.TYPE] == cm.ValidationAlgorithmType.LOCAL_DATASET_VALIDATION:
				if num_of_trainers_in_each_round < stored_config[cm.ConfigParams.VALIDATION_PARAMS][va.LocalDatasetUsedForValidationParams.MINIMUM_NUM_OF_UPDATES_BETWEEN_AGGREGATIONS]:
					raise ValueError("validate_committee_consensus_specific_config. Number of trainers in each round is less than the minimum number of updates between aggregations. Consequently, it is not possible to complete any round")
				elif stored_config[cm.ConfigParams.VALIDATION_PARAMS][va.LocalDatasetUsedForValidationParams.MINIMUM_NUM_OF_UPDATES_BETWEEN_AGGREGATIONS] < consensus_specific_config[AdditionalCommitteeSpecificConfigParams.NUM_OF_VALIDATORS]:
					raise ValueError("validate_committee_consensus_specific_config. Number of validators is higher than the minimum number of updates between aggregations. Consequently, it is not possible to elect a new committee")
		
			# Global dataset validation
			elif stored_config[cm.ConfigParams.VALIDATION_PARAMS][cm.ConfigGenericConsensusAlgorithmParams.TYPE] == cm.ValidationAlgorithmType.GLOBAL_DATASET_VALIDATION:
				if num_of_trainers_in_each_round < stored_config[cm.ConfigParams.VALIDATION_PARAMS][va.GlobalDatasetUsedForValidationParams.MINIMUM_NUM_OF_UPDATES_BETWEEN_AGGREGATIONS]:
					raise ValueError("validate_committee_consensus_specific_config. Number of trainers in each round is less than the minimum number of updates between aggregations. Consequently, it is not possible to complete any round")
				elif stored_config[cm.ConfigParams.VALIDATION_PARAMS][va.GlobalDatasetUsedForValidationParams.MINIMUM_NUM_OF_UPDATES_BETWEEN_AGGREGATIONS] < consensus_specific_config[AdditionalCommitteeSpecificConfigParams.NUM_OF_VALIDATORS]:
					raise ValueError("validate_committee_consensus_specific_config. Number of validators is higher than the minimum number of updates between aggregations. Consequently, it is not possible to elect a new committee")
			
			# Pass weights validation
			elif stored_config[cm.ConfigParams.VALIDATION_PARAMS][cm.ConfigGenericConsensusAlgorithmParams.TYPE] == cm.ValidationAlgorithmType.PASS_WEIGHTS:
				if num_of_trainers_in_each_round < stored_config[cm.ConfigParams.VALIDATION_PARAMS][va.PassWeightsValidationParams.MINIMUM_NUM_OF_UPDATES_BETWEEN_AGGREGATIONS]:
					raise ValueError("validate_committee_consensus_specific_config. Number of trainers in each round is less than the minimum number of updates between aggregations. Consequently, it is not possible to complete any round")
				elif stored_config[cm.ConfigParams.VALIDATION_PARAMS][va.PassWeightsValidationParams.MINIMUM_NUM_OF_UPDATES_BETWEEN_AGGREGATIONS] < consensus_specific_config[AdditionalCommitteeSpecificConfigParams.NUM_OF_VALIDATORS]:
					raise ValueError("validate_committee_consensus_specific_config. Number of validators is higher than the minimum number of updates between aggregations. Consequently, it is not possible to elect a new committee")

			# Krum validation
			elif stored_config[cm.ConfigParams.VALIDATION_PARAMS][cm.ConfigGenericConsensusAlgorithmParams.TYPE] == cm.ValidationAlgorithmType.KRUM:
				if num_of_trainers_in_each_round < stored_config[cm.ConfigParams.VALIDATION_PARAMS][va.KrumValidationParams.MINIMUM_NUM_OF_UPDATES_NEEDED_TO_START_VALIDATION]:
					raise ValueError("validate_committee_consensus_specific_config. Number of trainers in each round is less than the minimum number of updates needed to start the validation. Consequently, it is not possible to complete any round")
				elif stored_config[cm.ConfigParams.VALIDATION_PARAMS][va.KrumValidationParams.MINIMUM_NUM_OF_UPDATES_NEEDED_TO_START_VALIDATION] < consensus_specific_config[AdditionalCommitteeSpecificConfigParams.NUM_OF_VALIDATORS]:
					raise ValueError("validate_committee_consensus_specific_config. Number of validators is higher than the minimum number of updates needed to start the validation. Consequently, we are not sure to be able to elect a new committee")
			
			# Trimmed mean validation
			elif stored_config[cm.ConfigParams.VALIDATION_PARAMS][cm.ConfigGenericConsensusAlgorithmParams.TYPE] == cm.ValidationAlgorithmType.TRIMMED_MEAN:
				if num_of_trainers_in_each_round < stored_config[cm.ConfigParams.VALIDATION_PARAMS][va.TrimmedMeanValidationParams.MINIMUM_NUM_OF_UPDATES_NEEDED_TO_START_VALIDATION]:
					raise ValueError("validate_committee_consensus_specific_config. Number of trainers in each round is less than the minimum number of updates needed to start the validation. Consequently, it is not possible to complete any round")
				elif stored_config[cm.ConfigParams.VALIDATION_PARAMS][va.TrimmedMeanValidationParams.MINIMUM_NUM_OF_UPDATES_NEEDED_TO_START_VALIDATION] < consensus_specific_config[AdditionalCommitteeSpecificConfigParams.NUM_OF_VALIDATORS]:
					raise ValueError("validate_committee_consensus_specific_config. Number of validators is higher than the minimum number of updates needed to start the validation. Consequently, we are not sure to be able to elect a new committee")
			
			# Pass gradients validation
			elif stored_config[cm.ConfigParams.VALIDATION_PARAMS][cm.ConfigGenericConsensusAlgorithmParams.TYPE] == cm.ValidationAlgorithmType.PASS_GRADIENTS:
				if num_of_trainers_in_each_round < stored_config[cm.ConfigParams.VALIDATION_PARAMS][va.PassGradientsValidationParams.MINIMUM_NUM_OF_UPDATES_NEEDED_TO_START_VALIDATION]:
					raise ValueError("validate_committee_consensus_specific_config. Number of trainers in each round is less than the number of updates between aggregations. Consequently, it is not possible to complete any round")
				elif stored_config[cm.ConfigParams.VALIDATION_PARAMS][va.PassGradientsValidationParams.MINIMUM_NUM_OF_UPDATES_NEEDED_TO_START_VALIDATION] < consensus_specific_config[AdditionalCommitteeSpecificConfigParams.NUM_OF_VALIDATORS]:
					raise ValueError("validate_committee_consensus_specific_config. Number of validators is higher than the minimum number of updates needed to start the validation. Consequently, we are not sure to be able to elect a new committee")

			else:
				raise ValueError("validate_committee_consensus_specific_config. Invalid validation algorithm")

	except Exception as e:
		raise Exception(f"validate_committee_consensus_specific_config. Exception: {e}") from e

def start_nodes(process_index: int, logger_path: str, logger_level: int, kill_process_event: EventClass, nodes_specific_params: dict, quanta_paths: list, perc_of_iid_quanta: float, archive_host: str, archive_port: int, fit_epochs: int, batch_size: int, initial_peers_dict: dict, genesis_block, need_to_join_an_existing_network: bool, lazy_loading: bool, test_set_path: str, val_set_path: str, test_set_validation_of_global_model_after_each_round: bool, malicious_training_semaphore: BoundedSemaphoreClass, honest_training_semaphore: BoundedSemaphoreClass, validation_semaphore: BoundedSemaphoreClass, entry_point_nodes: list = [], store_weights_directly_in_archive_tmp_dir: bool = False, archive_tmp_dir: (str | None) = None) -> None:
	'''Start one or multiple nodes'''

	# Important: logger's type, quanta's type and test_set's type are not checked because they are not instances of Diagnostic, list and dict classes but of special classes provided by multiprocessing library

	if type(process_index) != int or type(logger_path) != str or type(logger_level) != int or isinstance(kill_process_event, EventClass) is False or type(nodes_specific_params) != dict or type(need_to_join_an_existing_network) != bool or type(quanta_paths) != list or (type(entry_point_nodes) != list or any(type(elem) != dict or any(key not in elem for key in cm.ConfigEntryPointNodesParams.list()) for elem in entry_point_nodes)) or type(lazy_loading) != bool or type(test_set_validation_of_global_model_after_each_round) != bool or isinstance(malicious_training_semaphore, BoundedSemaphoreClass) is False or isinstance(honest_training_semaphore, BoundedSemaphoreClass) is False or isinstance(validation_semaphore, BoundedSemaphoreClass) is False:
		raise TypeError("start_nodes method. Invalid parameters 1")
	elif kill_process_event.is_set() or type(perc_of_iid_quanta) not in [float, int] or type(archive_host) != str or type(archive_port) != int or type(fit_epochs) != int or type(batch_size) != int or type(initial_peers_dict) != dict or type(store_weights_directly_in_archive_tmp_dir) != bool or (type(archive_tmp_dir) != str and archive_tmp_dir is not None) or type(test_set_path) != str or type(val_set_path) != str:
		raise TypeError("start_nodes method. Invalid parameters 2")
	elif len(nodes_specific_params) == 0 or perc_of_iid_quanta < 0.0 or perc_of_iid_quanta > 1.0 or archive_port < 0 or archive_port > 65535 or fit_epochs < 1 or batch_size < 1 or logger_level not in [diagnostic.DEBUG, diagnostic.INFO, diagnostic.WARNING, diagnostic.ERROR]:
		raise ValueError("start_nodes method. Invalid parameters")
	elif need_to_join_an_existing_network is True and len(entry_point_nodes) == 0:
		raise ValueError("start_nodes method. Need to join an existing network is True but no entry point nodes are provided")
	elif store_weights_directly_in_archive_tmp_dir is True and archive_tmp_dir is None:
		raise ValueError("start_nodes method. Store weights directly in archive tmp dir is True but archive tmp dir is None")
	elif store_weights_directly_in_archive_tmp_dir is True and os.path.exists(archive_tmp_dir) is False:
		raise ValueError("start_nodes method. Store weights directly in archive tmp dir is True but archive tmp dir does not exist")
	elif any([type(path) != str or os.path.isfile(path) is False for path in quanta_paths]) or os.path.isfile(test_set_path) is False or os.path.isfile(val_set_path) is False:
		raise ValueError("start_nodes method. Invalid paths")
	
	setproctitle.setproctitle(f"{current_process().name}")

	gpus = configure_gpus()

	if len(gpus) > 0:
		print(f"Process is using GPU. GPUs detected: {gpus}")

	logger = diagnostic.Diagnostic(logger_path, logger_level, logger_name= f"Process {process_index}")

	try:
		nodes_threads = []

		for node_id in nodes_specific_params:
			if type(nodes_specific_params[node_id]) != dict:
				raise TypeError("start_nodes method. Invalid node config")
			elif any(key not in nodes_specific_params[node_id] for key in cm.NodeSpecificConfigParams.list()):
				raise KeyError("start_nodes method. Invalid node config")

			if type(nodes_specific_params[node_id][cm.NodeSpecificConfigParams.NODE_ID]) != int or type(nodes_specific_params[node_id][cm.NodeSpecificConfigParams.HOST]) != str or type(nodes_specific_params[node_id][cm.NodeSpecificConfigParams.PORT]) != int or type(nodes_specific_params[node_id][cm.NodeSpecificConfigParams.ALLOWED_TO_PRODUCE_DEBUG_LOG_MESSAGES]) != bool or (type(nodes_specific_params[node_id][cm.NodeSpecificConfigParams.ACTIVE_TRAINER_IN_FIRST_ROUND]) != bool and nodes_specific_params[node_id][cm.NodeSpecificConfigParams.ACTIVE_TRAINER_IN_FIRST_ROUND] is not None) or isinstance(nodes_specific_params[node_id][cm.NodeSpecificConfigParams.PAUSE_RESUME_EVENT], EventClass) is False:
				raise TypeError("start_nodes method. Invalid node config param")
			elif nodes_specific_params[node_id][cm.NodeSpecificConfigParams.PORT] < 0 or nodes_specific_params[node_id][cm.NodeSpecificConfigParams.PORT] > 65535 or nodes_specific_params[node_id][cm.NodeSpecificConfigParams.NUM_OF_IID_QUANTA_TO_USE] < 0 or nodes_specific_params[node_id][cm.NodeSpecificConfigParams.NUM_OF_QUANTA_TO_USE] < 0 or nodes_specific_params[node_id][cm.NodeSpecificConfigParams.PAUSE_RESUME_EVENT].is_set():
				raise ValueError("start_nodes method. Invalid node config param")
			
			# create the list of quanta paths
			selected_quanta_indexes = select_quanta_paths(quanta_paths, perc_of_iid_quanta, nodes_specific_params[node_id][cm.NodeSpecificConfigParams.NUM_OF_IID_QUANTA_TO_USE], nodes_specific_params[node_id][cm.NodeSpecificConfigParams.NUM_OF_QUANTA_TO_USE], SEED + nodes_specific_params[node_id][cm.NodeSpecificConfigParams.NODE_ID])
			selected_quanta_paths = [quanta_paths[index] for index in selected_quanta_indexes]
			
			if not nodes_specific_params[node_id][cm.NodeSpecificConfigParams.MALICIOUS]:
				nodes_threads.append(
					CommitteeBasedNode(
						logger,
						nodes_specific_params[node_id][cm.NodeSpecificConfigParams.NODE_ID],
						nodes_specific_params[node_id][cm.NodeSpecificConfigParams.HOST],
						nodes_specific_params[node_id][cm.NodeSpecificConfigParams.PORT],
						selected_quanta_paths,
						archive_host,
						archive_port,
						lazy_loading,
						test_set_path,
						val_set_path,
						malicious_training_semaphore,
						honest_training_semaphore,
						validation_semaphore,
						fit_epochs,
						batch_size,
						initial_peers_dict,
						nodes_specific_params[node_id][cm.NodeSpecificConfigParams.ALLOWED_TO_PRODUCE_DEBUG_LOG_MESSAGES],
						test_set_validation_of_global_model_after_each_round,
						genesis_block,
						nodes_specific_params[node_id][cm.NodeSpecificConfigParams.ACTIVE_TRAINER_IN_FIRST_ROUND],
						store_weights_directly_in_archive_tmp_dir,
						archive_tmp_dir
					)
				)
			
			else:
				if nodes_specific_params[node_id][cm.AttackerSpecificConfigParams.TYPE] == cm.MaliciousNodeBehaviourType.LABEL_FLIPPING:
					nodes_threads.append(
						CommitteeBasedLabelFlippingNode(
							logger,
							nodes_specific_params[node_id][cm.AttackerSpecificConfigParams.NODE_ID],
							nodes_specific_params[node_id][cm.AttackerSpecificConfigParams.HOST],
							nodes_specific_params[node_id][cm.AttackerSpecificConfigParams.PORT],
							selected_quanta_paths,
							archive_host,
							archive_port,
							lazy_loading,
							test_set_path,
							val_set_path,
							malicious_training_semaphore,
							honest_training_semaphore,
							validation_semaphore,
							fit_epochs,
							batch_size,
							initial_peers_dict,
							nodes_specific_params[node_id][cm.AttackerSpecificConfigParams.ALLOWED_TO_PRODUCE_DEBUG_LOG_MESSAGES],
							test_set_validation_of_global_model_after_each_round,
							genesis_block,
							nodes_specific_params[node_id][cm.AttackerSpecificConfigParams.ACTIVE_TRAINER_IN_FIRST_ROUND],
							store_weights_directly_in_archive_tmp_dir,
							archive_tmp_dir,
							nodes_specific_params[node_id][cm.AttackerSpecificConfigParams.STARTING_ROUND_FOR_MALICIOUS_BEHAVIOUR],
							nodes_specific_params[node_id][cm.AttackerSpecificConfigParams.SELECTED_CLASSES],
							nodes_specific_params[node_id][cm.AttackerSpecificConfigParams.TARGET_CLASSES],
							nodes_specific_params[node_id][cm.AttackerSpecificConfigParams.NUM_OF_SAMPLES],
							len(nodes_specific_params[node_id][cm.AttackerSpecificConfigParams.COLLUSION_PEERS]) > 0,
							nodes_specific_params[node_id][cm.AttackerSpecificConfigParams.COLLUSION_PEERS]
						)
					)
				
				elif nodes_specific_params[node_id][cm.AttackerSpecificConfigParams.TYPE] == cm.MaliciousNodeBehaviourType.TARGETED_POISONING:
					nodes_threads.append(
						CommitteeBasedTargetedPoisoningNode(
							logger,
							nodes_specific_params[node_id][cm.AttackerSpecificConfigParams.NODE_ID],
							nodes_specific_params[node_id][cm.AttackerSpecificConfigParams.HOST],
							nodes_specific_params[node_id][cm.AttackerSpecificConfigParams.PORT],
							selected_quanta_paths,
							archive_host,
							archive_port,
							lazy_loading,
							test_set_path,
							val_set_path,
							malicious_training_semaphore,
							honest_training_semaphore,
							validation_semaphore,
							fit_epochs,
							batch_size,
							initial_peers_dict,
							nodes_specific_params[node_id][cm.AttackerSpecificConfigParams.ALLOWED_TO_PRODUCE_DEBUG_LOG_MESSAGES],
							test_set_validation_of_global_model_after_each_round,
							genesis_block,
							nodes_specific_params[node_id][cm.AttackerSpecificConfigParams.ACTIVE_TRAINER_IN_FIRST_ROUND],
							store_weights_directly_in_archive_tmp_dir,
							archive_tmp_dir,
							nodes_specific_params[node_id][cm.AttackerSpecificConfigParams.STARTING_ROUND_FOR_MALICIOUS_BEHAVIOUR],
							nodes_specific_params[node_id][cm.AttackerSpecificConfigParams.TARGET_CLASS],
							nodes_specific_params[node_id][cm.AttackerSpecificConfigParams.SQUARE_SIZE],
							nodes_specific_params[node_id][cm.AttackerSpecificConfigParams.NUM_OF_SAMPLES],
							len(nodes_specific_params[node_id][cm.AttackerSpecificConfigParams.COLLUSION_PEERS]) > 0,
							nodes_specific_params[node_id][cm.AttackerSpecificConfigParams.COLLUSION_PEERS]
						)
					)
				
				elif nodes_specific_params[node_id][cm.AttackerSpecificConfigParams.TYPE] == cm.MaliciousNodeBehaviourType.RANDOM_LABEL:
					nodes_threads.append(
						CommitteeBasedRandomLabelByzantineNode(
							logger,
							nodes_specific_params[node_id][cm.AttackerSpecificConfigParams.NODE_ID],
							nodes_specific_params[node_id][cm.AttackerSpecificConfigParams.HOST],
							nodes_specific_params[node_id][cm.AttackerSpecificConfigParams.PORT],
							selected_quanta_paths,
							archive_host,
							archive_port,
							lazy_loading,
							test_set_path,
							val_set_path,
							malicious_training_semaphore,
							honest_training_semaphore,
							validation_semaphore,
							fit_epochs,
							batch_size,
							initial_peers_dict,
							nodes_specific_params[node_id][cm.AttackerSpecificConfigParams.ALLOWED_TO_PRODUCE_DEBUG_LOG_MESSAGES],
							test_set_validation_of_global_model_after_each_round,
							genesis_block,
							nodes_specific_params[node_id][cm.AttackerSpecificConfigParams.ACTIVE_TRAINER_IN_FIRST_ROUND],
							store_weights_directly_in_archive_tmp_dir,
							archive_tmp_dir,
							nodes_specific_params[node_id][cm.AttackerSpecificConfigParams.STARTING_ROUND_FOR_MALICIOUS_BEHAVIOUR],
							nodes_specific_params[node_id][cm.AttackerSpecificConfigParams.NUM_OF_SAMPLES],
							len(nodes_specific_params[node_id][cm.AttackerSpecificConfigParams.COLLUSION_PEERS]) > 0,
							nodes_specific_params[node_id][cm.AttackerSpecificConfigParams.COLLUSION_PEERS]
						)
					)
				
				elif nodes_specific_params[node_id][cm.AttackerSpecificConfigParams.TYPE] == cm.MaliciousNodeBehaviourType.ADDITIVE_NOISE:
					nodes_threads.append(
						CommitteeBasedAdditiveNoiseByzantineNode(
							logger,
							nodes_specific_params[node_id][cm.AttackerSpecificConfigParams.NODE_ID],
							nodes_specific_params[node_id][cm.AttackerSpecificConfigParams.HOST],
							nodes_specific_params[node_id][cm.AttackerSpecificConfigParams.PORT],
							selected_quanta_paths,
							archive_host,
							archive_port,
							lazy_loading,
							test_set_path,
							val_set_path,
							malicious_training_semaphore,
							honest_training_semaphore,
							validation_semaphore,
							fit_epochs,
							batch_size,
							initial_peers_dict,
							nodes_specific_params[node_id][cm.AttackerSpecificConfigParams.ALLOWED_TO_PRODUCE_DEBUG_LOG_MESSAGES],
							test_set_validation_of_global_model_after_each_round,
							genesis_block,
							nodes_specific_params[node_id][cm.AttackerSpecificConfigParams.ACTIVE_TRAINER_IN_FIRST_ROUND],
							store_weights_directly_in_archive_tmp_dir,
							archive_tmp_dir,
							nodes_specific_params[node_id][cm.AttackerSpecificConfigParams.STARTING_ROUND_FOR_MALICIOUS_BEHAVIOUR],
							nodes_specific_params[node_id][cm.AttackerSpecificConfigParams.SIGMA],
							nodes_specific_params[node_id][cm.AttackerSpecificConfigParams.NUM_OF_SAMPLES],
							len(nodes_specific_params[node_id][cm.AttackerSpecificConfigParams.COLLUSION_PEERS]) > 0,
							nodes_specific_params[node_id][cm.AttackerSpecificConfigParams.COLLUSION_PEERS]
						)
					)
				
				elif nodes_specific_params[node_id][cm.AttackerSpecificConfigParams.TYPE] == cm.MaliciousNodeBehaviourType.RANDOM_NOISE:
					nodes_threads.append(
						CommitteeBasedRandomNoiseByzantineNode(
							logger,
							nodes_specific_params[node_id][cm.AttackerSpecificConfigParams.NODE_ID],
							nodes_specific_params[node_id][cm.AttackerSpecificConfigParams.HOST],
							nodes_specific_params[node_id][cm.AttackerSpecificConfigParams.PORT],
							selected_quanta_paths,
							archive_host,
							archive_port,
							lazy_loading,
							test_set_path,
							val_set_path,
							malicious_training_semaphore,
							honest_training_semaphore,
							validation_semaphore,
							fit_epochs,
							batch_size,
							initial_peers_dict,
							nodes_specific_params[node_id][cm.AttackerSpecificConfigParams.ALLOWED_TO_PRODUCE_DEBUG_LOG_MESSAGES],
							test_set_validation_of_global_model_after_each_round,
							genesis_block,
							nodes_specific_params[node_id][cm.AttackerSpecificConfigParams.ACTIVE_TRAINER_IN_FIRST_ROUND],
							store_weights_directly_in_archive_tmp_dir,
							archive_tmp_dir,
							nodes_specific_params[node_id][cm.AttackerSpecificConfigParams.STARTING_ROUND_FOR_MALICIOUS_BEHAVIOUR],
							nodes_specific_params[node_id][cm.AttackerSpecificConfigParams.NUM_OF_SAMPLES],
							len(nodes_specific_params[node_id][cm.AttackerSpecificConfigParams.COLLUSION_PEERS]) > 0,
							nodes_specific_params[node_id][cm.AttackerSpecificConfigParams.COLLUSION_PEERS]
						)
					)
				
				else:
					raise ValueError(f"start_nodes method. Invalid malicious node behaviour. Behaviour: {nodes_specific_params[node_id][cm.AttackerSpecificConfigParams.TYPE]}")
					
			# The node has already been initialized
			nodes_specific_params[node_id][cm.NodeSpecificConfigParams.PAUSE_RESUME_EVENT].set()
			print(f"Node {node_id} initialized")

		# Wait for the main process to start the node
		while True:
			if all(nodes_specific_params[node_id][cm.NodeSpecificConfigParams.PAUSE_RESUME_EVENT].is_set() is False for node_id in nodes_specific_params) or kill_process_event.is_set():
				break

			time.sleep(0.2)

		if kill_process_event.is_set():
			return

		for node in nodes_threads:
			node.start()

		len_entry_point_nodes = len(entry_point_nodes)

		if need_to_join_an_existing_network:
			for index, node in enumerate(nodes_threads):
				node.connect_to_peer(entry_point_nodes[index % len_entry_point_nodes][cm.ConfigEntryPointNodesParams.HOST], entry_point_nodes[index % len_entry_point_nodes][cm.ConfigEntryPointNodesParams.PORT])

		while True: 
			if kill_process_event.is_set():
				break
			elif any(node.is_alive() is False for node in nodes_threads):
				break

			time.sleep(1)

		if kill_process_event.is_set():
			print(f"Process killed")

		for node in nodes_threads:
			if node.is_alive():
				node.stop()
		
		for node in nodes_threads:
			node.join()

		print(f"Process joined")
	
	except Exception as e:
		logger.record(f"Exception occurred", diagnostic.ERROR, f"Process {process_index}", exc= e)
		print(f"Exception occurred in process {process_index}. Exception: {e}")
	else:
		logger.record(f"Process exited correctly", diagnostic.DEBUG, f"Process {process_index}")
		print(f"Process {process_index} exited correctly")
	finally:
		logger.shutdown()

def start_committee_simulation_main(logger: diagnostic.Diagnostic, quanta_paths: list, testset_path: str, valset_path: str, config_file_path: str, preloaded_config: dict, run_time_only_params: dict) -> None:
	'''Start the committee-based simulation'''

	if isinstance(logger, diagnostic.Diagnostic) is False or type(quanta_paths) != list or type(testset_path) != str or type(valset_path) != str or type(config_file_path) != str or type(preloaded_config) != dict or type(run_time_only_params) != dict:
		raise TypeError("start_committee_simulation_main")
	elif os.path.exists(config_file_path) is False or any(os.path.isfile(path) is False for path in quanta_paths) or os.path.isfile(testset_path) is False or os.path.isfile(valset_path) is False:
		raise FileNotFoundError("start_committee_simulation_main")
	elif any(key not in preloaded_config for key in cm.ConfigParams.list()):
		raise KeyError("start_committee_simulation_main. cm.ConfigParams missing key")
	elif any(key not in run_time_only_params for key in cm.RunTimeOnlyParams.list()):
		raise KeyError("start_committee_simulation_main. Run time only params missing key")

	is_support_simulation = not preloaded_config[cm.ConfigParams.IS_MAIN_SIMULATION]
	
	# Load the configuration file
	preloaded_config[cm.ConfigParams.CONSENSUS_PARAMS] = {**DEFAULT_COMMITTEE_SPECIFIC_CONFIG, **preloaded_config[cm.ConfigParams.CONSENSUS_PARAMS]}

	validate_committee_consensus_specific_config(preloaded_config, run_time_only_params)
	
	#
	# If the simulation is the main one, we need to define the validators of the first round
	#

	nodes_ids_of_first_validators = list()

	if is_support_simulation is False:

		# First validators
		if preloaded_config[cm.ConfigParams.CONSENSUS_PARAMS][AdditionalCommitteeSpecificConfigParams.LIST_OF_NODE_IDS_OF_FIRST_VALIDATORS] is not None:
			nodes_ids_of_first_validators = preloaded_config[cm.ConfigParams.CONSENSUS_PARAMS][AdditionalCommitteeSpecificConfigParams.LIST_OF_NODE_IDS_OF_FIRST_VALIDATORS]
		
		else:
			nodes_ids_ready_for_being_validators_list = list()
			nodes_ids_not_available = list()

			if preloaded_config[cm.ConfigParams.CONSENSUS_PARAMS][AdditionalCommitteeSpecificConfigParams.LIST_OF_ACTIVE_TRAINERS_IN_THE_FIRST_ROUND] is not None:
				nodes_ids_not_available = preloaded_config[cm.ConfigParams.CONSENSUS_PARAMS][AdditionalCommitteeSpecificConfigParams.LIST_OF_ACTIVE_TRAINERS_IN_THE_FIRST_ROUND]

			for node in run_time_only_params[cm.RunTimeOnlyParams.NODES]:
				if node[cm.NodeSpecificConfigParams.NODE_ID] not in nodes_ids_not_available:
					nodes_ids_ready_for_being_validators_list.append(node[cm.NodeSpecificConfigParams.NODE_ID])
		
			if len(nodes_ids_ready_for_being_validators_list) < preloaded_config[cm.ConfigParams.CONSENSUS_PARAMS][AdditionalCommitteeSpecificConfigParams.NUM_OF_VALIDATORS]:
				raise ValueError("start_committee_simulation_main. Number of nodes that can be validators is less than the required number of validators")
		
			nodes_ids_of_first_validators = np.random.choice(nodes_ids_ready_for_being_validators_list, preloaded_config[cm.ConfigParams.CONSENSUS_PARAMS][AdditionalCommitteeSpecificConfigParams.NUM_OF_VALIDATORS], replace=False).tolist()

	#
	# Print the running configuration
	#

	print(f"Preloaded configuration:\n{json.dumps(preloaded_config, indent=4)}")

	if is_support_simulation is False:
		print(f"First validators: {nodes_ids_of_first_validators}")
		logger.record(f"First validators: {nodes_ids_of_first_validators}", diagnostic.INFO, _MODULE)

	# Create all the variables that will be shared among the node processes
	CustomManager.register('GenesisBlock', GenesisBlock)
	processes_shared_variables_manager = CustomManager()
	processes_shared_variables_manager.start()

	print("Config file loaded successfully")
	logger.record("Config file loaded successfully", diagnostic.DEBUG, _MODULE)

	pu.ram_usage(preloaded_config[cm.ConfigParams.RAM_USAGE_LOG_PATH], True)

	gc.collect()

	print("Starting the simulation...")
	logger.record("Starting the simulation...", diagnostic.DEBUG, _MODULE)

	print("Creating the archive...")
	logger.record("Creating the archive...", diagnostic.DEBUG, _MODULE)

	# Creation of the Archive
	genesis_block = None
	archive_process = None
	archive_kill_process_event = None

	# If the network is not already created, the genesis block must be created and the archive must be started
	if is_support_simulation is False:
		threshold_to_pass_validation = round(preloaded_config[cm.ConfigParams.CONSENSUS_PARAMS][AdditionalCommitteeSpecificConfigParams.PERC_THRESHOLD_TO_PASS_VALIDATION] * len(nodes_ids_of_first_validators))

		genesis_block = processes_shared_variables_manager.GenesisBlock(
			# Global params
			run_time_only_params[cm.RunTimeOnlyParams.MODEL_ARCHITECTURE],
			run_time_only_params[cm.RunTimeOnlyParams.STARTING_WEIGHTS],
			run_time_only_params[cm.RunTimeOnlyParams.STARTING_OPTIMIZER_STATE],
			preloaded_config[cm.ConfigParams.MAX_NUM_OF_ROUNDS],
			preloaded_config[cm.ConfigParams.CONSENSUS_PARAMS][AdditionalCommitteeSpecificConfigParams.NUM_OF_VALIDATORS],
			preloaded_config[cm.ConfigParams.CONSENSUS_PARAMS][AdditionalCommitteeSpecificConfigParams.PERC_OF_TRAINERS_ACTIVE_IN_EACH_ROUND],
			threshold_to_pass_validation,
			nodes_ids_of_first_validators,
			preloaded_config[cm.ConfigParams.VALIDATION_PARAMS],
			preloaded_config[cm.ConfigParams.AGGREGATION_PARAMS]
		)

		if preloaded_config[cm.ConfigParams.ARCHIVE_PARAMS][cm.ConfigArchiveParams.ARCHIVE_MUST_BE_CREATED]:
			archive_kill_process_event = Event()
			process_ready_event = Event()

			archive_process = Process(
				target=start_archive,
				args=(
					CommitteeBasedArchive,
					archive_kill_process_event,
					process_ready_event,
					preloaded_config[cm.ConfigParams.ARCHIVE_PARAMS][cm.ConfigArchiveParams.HOST],
					preloaded_config[cm.ConfigParams.ARCHIVE_PARAMS][cm.ConfigArchiveParams.PORT],
					preloaded_config[cm.ConfigParams.ARCHIVE_PARAMS][cm.ConfigArchiveParams.TMP_DIR],
					preloaded_config[cm.ConfigParams.ARCHIVE_PARAMS][cm.ConfigArchiveParams.PERSISTENT_MODE],
					preloaded_config[cm.ConfigParams.ARCHIVE_PARAMS][cm.ConfigArchiveParams.LOGGER_PATH],
					preloaded_config[cm.ConfigParams.ARCHIVE_PARAMS][cm.ConfigArchiveParams.LOGGER_LEVEL],
					genesis_block
				), 
				name="FedBlockSimulator - start_archive process - committee",
				daemon=True
			)

			archive_process.start()

			if process_ready_event.wait(240) is False:
				raise TimeoutError("The archive process did not start after 240 seconds of waiting")

	# Creation of the nodes
	print("Creating the nodes...")
	logger.record("Creating the nodes...", diagnostic.DEBUG, _MODULE)

	# Creation of the semaphores to limit the number of training and validation subprocesses executed at the same time
	honest_training_semaphore = BoundedSemaphore(preloaded_config[cm.ConfigParams.MAXIMUM_NUMBER_OF_PARALLEL_HONEST_TRAININGS])
	validation_semaphore = BoundedSemaphore(preloaded_config[cm.ConfigParams.MAXIMUM_NUMBER_OF_PARALLEL_VALIDATIONS])
	malicious_training_semaphore = BoundedSemaphore(preloaded_config[cm.ConfigParams.MAXIMUM_NUMBER_OF_PARALLEL_MALICIOUS_TRAININGS])

	nodes_processes = []
	initial_peers_dict = {}
	pause_and_resume_events = []
	nodes_configs_in_each_process = {}

	if is_support_simulation is False:
		current_timestamp = time.time()

		for node in run_time_only_params[cm.RunTimeOnlyParams.NODES]:
			initial_peers_dict[node[cm.RunTimeNodeParams.NODE_ID]] = {pg.PeersListElement.HOST: node[cm.RunTimeNodeParams.HOST], pg.PeersListElement.PORT: node[cm.RunTimeNodeParams.PORT], pg.PeersListElement.STATUS: True, pg.PeersListElement.LAST_UPDATE_TIMESTAMP: current_timestamp}

	for index, node in enumerate(run_time_only_params[cm.RunTimeOnlyParams.NODES]):
		try:
			# Create a composite dataset
			node_num_of_quanta_to_use = None
			node_num_of_iid_quanta_to_use = None

			for composite_dataset in run_time_only_params[cm.RunTimeOnlyParams.NODES_COMPOSITE_DATASETS]:
				if node[cm.RunTimeNodeParams.NODE_ID] in composite_dataset[cm.RunTimeNodeCompositeDatasetParams.RELATED_NODES_IDS]:
					node_num_of_quanta_to_use = composite_dataset[cm.RunTimeNodeCompositeDatasetParams.NUM_OF_QUANTA_TO_USE]
					node_num_of_iid_quanta_to_use = composite_dataset[cm.RunTimeNodeCompositeDatasetParams.IID_QUANTA_TO_USE]
					break
			else:
				raise ValueError(f"Node ID {node[cm.RunTimeNodeParams.NODE_ID]} is not in any composite dataset")
	
			# Decide the type of node
			is_attacker = False
			behaviour = ""
			selected_classes = []
			target_classes = []
			target_class = None
			square_size = None
			sigma = None
			num_of_samples = None
			starting_round_for_malicious_behaviour = None

			for node_behaviour in preloaded_config[cm.ConfigParams.MALICIOUS_NODES_PARAMS][cm.ConfigMaliciousNodesParams.NODE_BEHAVIOURS]:
				if node[cm.RunTimeNodeParams.NODE_ID] in node_behaviour[cm.ConfigNodeBehaviorParams.NODES_ID]:
					is_attacker = True

					num_of_samples = node_behaviour[cm.ConfigNodeBehaviorParams.NUM_OF_SAMPLES]
					behaviour = node_behaviour[cm.ConfigNodeBehaviorParams.TYPE]
					starting_round_for_malicious_behaviour = node_behaviour[cm.ConfigNodeBehaviorParams.STARTING_ROUND_FOR_MALICIOUS_BEHAVIOUR]
					
					if node_behaviour[cm.ConfigNodeBehaviorParams.TYPE] == cm.MaliciousNodeBehaviourType.LABEL_FLIPPING:
						selected_classes = node_behaviour[cm.ConfigNodeBehaviorParams.SELECTED_CLASSES]
						target_classes = node_behaviour[cm.ConfigNodeBehaviorParams.TARGET_CLASSES]
					
					elif node_behaviour[cm.ConfigNodeBehaviorParams.TYPE] == cm.MaliciousNodeBehaviourType.TARGETED_POISONING:
						square_size = node_behaviour[cm.ConfigNodeBehaviorParams.SQUARE_SIZE]
						target_class = node_behaviour[cm.ConfigNodeBehaviorParams.TARGET_CLASS]
					
					elif node_behaviour[cm.ConfigNodeBehaviorParams.TYPE] == cm.MaliciousNodeBehaviourType.RANDOM_LABEL:
						pass

					elif node_behaviour[cm.ConfigNodeBehaviorParams.TYPE] == cm.MaliciousNodeBehaviourType.ADDITIVE_NOISE:
						sigma = node_behaviour[cm.ConfigNodeBehaviorParams.SIGMA]
					
					elif node_behaviour[cm.ConfigNodeBehaviorParams.TYPE] == cm.MaliciousNodeBehaviourType.RANDOM_NOISE:
						pass
					
					else:
						raise ValueError(f"Unknown malicious behaviour. Behaviour: {node_behaviour[cm.ConfigNodeBehaviorParams.TYPE]}")
					
					break

			# Node is active in the first round if it is in the list of active trainers in the first round
			node_active_in_first_round = None

			if preloaded_config[cm.ConfigParams.CONSENSUS_PARAMS][AdditionalCommitteeSpecificConfigParams.LIST_OF_ACTIVE_TRAINERS_IN_THE_FIRST_ROUND] is not None:
				if node[cm.RunTimeNodeParams.NODE_ID] in preloaded_config[cm.ConfigParams.CONSENSUS_PARAMS][AdditionalCommitteeSpecificConfigParams.LIST_OF_ACTIVE_TRAINERS_IN_THE_FIRST_ROUND]:
					node_active_in_first_round = True
				else:
					node_active_in_first_round = False

			pause_and_resume_event = Event()

			process_index = index % preloaded_config[cm.ConfigParams.NUM_OF_PROCESSES_TO_USE_TO_MANAGE_NODES]

			if process_index not in nodes_configs_in_each_process:
				nodes_configs_in_each_process[process_index] = {}

			if not is_attacker:
				nodes_configs_in_each_process[process_index][node[cm.RunTimeNodeParams.NODE_ID]] = {
					cm.NodeSpecificConfigParams.NODE_ID: node[cm.RunTimeNodeParams.NODE_ID],
					cm.NodeSpecificConfigParams.HOST: node[cm.RunTimeNodeParams.HOST],
					cm.NodeSpecificConfigParams.PORT: node[cm.RunTimeNodeParams.PORT],
					cm.NodeSpecificConfigParams.NUM_OF_QUANTA_TO_USE: node_num_of_quanta_to_use,
					cm.NodeSpecificConfigParams.NUM_OF_IID_QUANTA_TO_USE: node_num_of_iid_quanta_to_use,
					cm.NodeSpecificConfigParams.ACTIVE_TRAINER_IN_FIRST_ROUND: node_active_in_first_round,
					cm.NodeSpecificConfigParams.ALLOWED_TO_PRODUCE_DEBUG_LOG_MESSAGES: node[cm.RunTimeNodeParams.ALLOWED_TO_PRODUCE_DEBUG_LOG_MESSAGES],
					cm.NodeSpecificConfigParams.PAUSE_RESUME_EVENT: pause_and_resume_event,
					cm.NodeSpecificConfigParams.MALICIOUS: is_attacker,
				}
			else:
				nodes_configs_in_each_process[process_index][node[cm.RunTimeNodeParams.NODE_ID]] = {
					cm.AttackerSpecificConfigParams.NODE_ID: node[cm.RunTimeNodeParams.NODE_ID],
					cm.AttackerSpecificConfigParams.HOST: node[cm.RunTimeNodeParams.HOST],
					cm.AttackerSpecificConfigParams.PORT: node[cm.RunTimeNodeParams.PORT],
					cm.AttackerSpecificConfigParams.NUM_OF_QUANTA_TO_USE: node_num_of_quanta_to_use,
					cm.AttackerSpecificConfigParams.NUM_OF_IID_QUANTA_TO_USE: node_num_of_iid_quanta_to_use,
					cm.AttackerSpecificConfigParams.ACTIVE_TRAINER_IN_FIRST_ROUND: node_active_in_first_round,
					cm.AttackerSpecificConfigParams.ALLOWED_TO_PRODUCE_DEBUG_LOG_MESSAGES: node[cm.RunTimeNodeParams.ALLOWED_TO_PRODUCE_DEBUG_LOG_MESSAGES],
					cm.AttackerSpecificConfigParams.PAUSE_RESUME_EVENT: pause_and_resume_event,
					cm.AttackerSpecificConfigParams.MALICIOUS: is_attacker,
					cm.AttackerSpecificConfigParams.TYPE: behaviour,
					cm.AttackerSpecificConfigParams.SELECTED_CLASSES: selected_classes,
					cm.AttackerSpecificConfigParams.TARGET_CLASSES: target_classes,
					cm.AttackerSpecificConfigParams.TARGET_CLASS: target_class,
					cm.AttackerSpecificConfigParams.SQUARE_SIZE: square_size,
					cm.AttackerSpecificConfigParams.SIGMA: sigma,
					cm.AttackerSpecificConfigParams.STARTING_ROUND_FOR_MALICIOUS_BEHAVIOUR: starting_round_for_malicious_behaviour,
					cm.AttackerSpecificConfigParams.NUM_OF_SAMPLES: num_of_samples,
					cm.AttackerSpecificConfigParams.COLLUSION_PEERS: preloaded_config[cm.ConfigParams.MALICIOUS_NODES_PARAMS][cm.ConfigMaliciousNodesParams.LIST_OF_COLLUSION_PEER_IDS]
				}

			pause_and_resume_events.append(pause_and_resume_event)

		except Exception as e:
			logger.record(f"Exception while creating the node config. Node id: {node[cm.RunTimeNodeParams.NODE_ID]}", diagnostic.ERROR, _MODULE, exc= e)
			print(f"Exception while creating the node config. Node id: {node[cm.RunTimeNodeParams.NODE_ID]}. Exception: {e}")
			sys.exit(1)

	if len(nodes_configs_in_each_process) > preloaded_config[cm.ConfigParams.NUM_OF_PROCESSES_TO_USE_TO_MANAGE_NODES]:
		raise ValueError("The number of processes to use to manage nodes specified in the config is lower than the number of processes that have been created")
	
	elif len(nodes_configs_in_each_process) < preloaded_config[cm.ConfigParams.NUM_OF_PROCESSES_TO_USE_TO_MANAGE_NODES]:
		logger.record(f"The number of processes to use to manage nodes specified in the config is higher than the number of processes that have been created", diagnostic.INFO, _MODULE)
		print("The number of processes to use to manage nodes specified in the config is higher than the number of processes that have been created")
	
	try:
		for process_id in nodes_configs_in_each_process:

			kill_process_event = Event()

			nodes_processes.append(
				(
					Process(
						target=start_nodes,
						args=(
							process_id,
							preloaded_config[cm.ConfigParams.LOGGER_PATH],
							preloaded_config[cm.ConfigParams.LOGGER_LEVEL],
							kill_process_event,
							nodes_configs_in_each_process[process_id],
							quanta_paths,
							preloaded_config[cm.ConfigParams.DATASET_PARAMS][cm.ConfigDatasetParams.PERC_OF_IID_QUANTA],
							preloaded_config[cm.ConfigParams.ARCHIVE_PARAMS][cm.ConfigArchiveParams.HOST],
							preloaded_config[cm.ConfigParams.ARCHIVE_PARAMS][cm.ConfigArchiveParams.PORT],
							preloaded_config[cm.ConfigParams.FIT_EPOCHS],
							preloaded_config[cm.ConfigParams.BATCH_SIZE],
							initial_peers_dict,
							genesis_block,
							preloaded_config[cm.ConfigParams.NEED_TO_JOIN_AN_EXISTING_NETWORK],
							preloaded_config[cm.ConfigParams.DATASET_PARAMS][cm.ConfigDatasetParams.LAZY_LOADING],
							testset_path,
							valset_path,
							preloaded_config[cm.ConfigParams.VALIDATION_WITH_TEST_SET_AFTER_MODEL_AGGREGATION],
							malicious_training_semaphore,
							honest_training_semaphore,
							validation_semaphore,
							preloaded_config[cm.ConfigParams.ENTRY_POINT_NODES],
							preloaded_config[cm.ConfigParams.STORE_WEIGHTS_DIRECTLY_IN_ARCHIVE_TMP_DIR],
							preloaded_config[cm.ConfigParams.ARCHIVE_PARAMS][cm.ConfigArchiveParams.TMP_DIR] if preloaded_config[cm.ConfigParams.STORE_WEIGHTS_DIRECTLY_IN_ARCHIVE_TMP_DIR] is True else None 
						),
						name=f"FedBlockSimulator - start_nodes process {process_id} - committee"
						# daemon=True												# Cannot be daemon because it will create new subprocesses
					),
					kill_process_event
				)
			)

			nodes_processes[-1][0].start()

	except Exception as e:
		logger.record(f"Exception while starting the nodes processes", diagnostic.ERROR, _MODULE, exc= e)
		print(f"Exception while starting the nodes processes. Exception: {e}")
		sys.exit(1)

	for node_id in preloaded_config[cm.ConfigParams.NODES_PARAMS][cm.ConfigNodesParams.LIST_OF_NODES_ALLOWED_TO_PRODUCE_DEBUG_LOG_MESSAGES]:
		if any(node_id == node[cm.RunTimeNodeParams.NODE_ID] for node in run_time_only_params[cm.RunTimeOnlyParams.NODES]) is False:
			logger.record(f"Node ID {node_id} is not a valid node ID. Its debug messages will not be printed", diagnostic.WARNING, _MODULE)
			print(f"Node ID {node_id} is not a valid node ID. Its debug messages will not be printed")

	for node_id in preloaded_config[cm.ConfigParams.MALICIOUS_NODES_PARAMS][cm.ConfigMaliciousNodesParams.LIST_OF_COLLUSION_PEER_IDS]:
		if any(node_id == node[cm.RunTimeNodeParams.NODE_ID] for node in run_time_only_params[cm.RunTimeOnlyParams.NODES]) is False:
			logger.record(f"Collusion peer ID {node_id} is not valid. The node doesn't exist at the moment. It may be part of an auxiliary simulation", diagnostic.WARNING, _MODULE)
			print(f"Collusion peer ID {node_id} is not valid. The node doesn't exist at the moment. It may be part of an auxiliary simulation")

	# Start the simulation

	# Wait for all the nodes to be ready
	while True:
		if all(pause_and_resume_event.is_set() for pause_and_resume_event in pause_and_resume_events):
			break

		time.sleep(0.5)

		if round(time.time()) % 10 == 0:
			print("Waiting for all the nodes to be ready...")
		
	print("All the nodes are ready")

	# Resume all the nodes
	for pause_and_resume_event in pause_and_resume_events:
		pause_and_resume_event.clear()

	del pause_and_resume_events
	gc.collect()

	print("Simulation started")
	logger.record("Simulation started", diagnostic.INFO, _MODULE)

	minute_counter = 0

	try:
		while len(nodes_processes) > 0:
			time.sleep(5)
			if minute_counter % 12 == 0:
				pu.ram_usage(file_path=preloaded_config[cm.ConfigParams.RAM_USAGE_LOG_PATH], store=True)

			nodes_to_remove = []
			for node in nodes_processes:
				if node[0].is_alive() is False:
					nodes_to_remove.append(node)
					logger.record(f"Process has stopped. Proc: {node[0]}. Is alive: {node[0].is_alive()}  Exit code: {node[0].exitcode}", diagnostic.INFO, _MODULE)
					print(f"Process has stopped. Proc: {node[0]}. Is alive: {node[0].is_alive()}  Exit code: {node[0].exitcode}")

			for node in nodes_to_remove:
				nodes_processes.remove(node)

			minute_counter += 1

	except KeyboardInterrupt:
		print("Keyboard interrupt. Stopping the simulation...")
		
		for node in nodes_processes:
			if node[0].is_alive():
				node[1].set()

		for node in nodes_processes:
			node[0].join()

	except Exception as e:
		raise e
	else:
		logger.record("All the nodes have stopped", diagnostic.INFO, _MODULE)
		print("All the nodes have stopped")
	finally:
		if is_support_simulation is False:
			if archive_process is not None and archive_process.is_alive():
				archive_kill_process_event.set()
				archive_process.join()
			
			processes_shared_variables_manager.shutdown()

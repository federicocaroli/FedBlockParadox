import hashlib, json, socket, time, numpy as np, gc, os, os.path, setproctitle, string	

from tensorflow import convert_to_tensor, GradientTape
from tensorflow.keras.callbacks import Callback

from multiprocessing import Process, current_process
from multiprocessing.synchronize import BoundedSemaphore as BoundedSemaphoreClass

from threading import Lock, Event, Thread

from . import diagnostic, utils
from .peer import Peer
from .constants import VERBOSE, TESTING_OPS, SEED, DIRECTORY_WHERE_TO_STORE_TMP_FILES
from .validation_algos import KrumValidation, LocalDatasetValidation, TrimmedMeanValidation, PassWeightsValidation, GlobalDatasetValidation, PassGradientsValidation
from .aggregation_algos import FlowerFedAvg, MeanAgg, MedianAgg, TrimmedMeanAgg
from .block import BlockType

from .enums.node_generic import InvalidRoundException, ValidationResultsListElemFields
from .enums.archive_generic import PeerUpdate, AggregatedModel
from .enums import peer_messages as pm, archive_messages as am, node_messages as nm, common as cm, validation_algos as va, aggregation_algos as ag, node_generic as ng

class SampleCounterCallback(Callback):
	def __init__(self, batch_size):
		super().__init__()
		self.total_samples = 0
		self.result = 0
		self.batch_size = batch_size

	def on_train_batch_end(self, batch, logs=None):		
		self.total_samples += self.batch_size
	
	def on_epoch_end(self, epoch, logs=None):
		self.result = self.total_samples 
		self.total_samples = 0

class GenericNode(Peer):
	
	# IMPORTANT: It must be overridden by the subclasses
	def __init__(self, logger: diagnostic.Diagnostic, node_id: int, host: str, port: int, dataset_quanta_paths: list, archive_host: str, archive_port: int, lazy_loading: bool, test_set_path: str, val_set_path: str, malicious_training_semaphore: BoundedSemaphoreClass, honest_training_semaphore: BoundedSemaphoreClass, validation_semaphore: BoundedSemaphoreClass, fit_epochs: int = 3, batch_size: int = 32, already_available_peers: dict = {}, allowed_to_write_redudant_log_messages: bool = False, test_set_validation_of_global_model_after_each_round: bool = False, is_active_trainer_in_first_round: (bool | None) = None, store_weights_directly_in_archive_tmp_dir: bool = False, archive_tmp_dir: (str | None) = None):
		
		if type(node_id) != int or type(host) != str or type(port) != int or type(dataset_quanta_paths) != list or type(archive_host) != str or type(archive_port) != int or type(fit_epochs) != int or type(batch_size) != int or type(already_available_peers) != dict or type(allowed_to_write_redudant_log_messages) != bool or (is_active_trainer_in_first_round is not None and type(is_active_trainer_in_first_round) != bool) or type(test_set_path) != str or type(val_set_path) != str or type(lazy_loading) != bool or type(store_weights_directly_in_archive_tmp_dir) != bool or (type(archive_tmp_dir) != str and archive_tmp_dir is not None) or isinstance(logger, diagnostic.Diagnostic) is False or type(test_set_validation_of_global_model_after_each_round) != bool or isinstance(malicious_training_semaphore, BoundedSemaphoreClass) is False or isinstance(honest_training_semaphore, BoundedSemaphoreClass) is False or isinstance(validation_semaphore, BoundedSemaphoreClass) is False:
			raise TypeError("Node constructor")
		elif fit_epochs <= 0 or batch_size <= 0 or node_id < 0 or port < 0 or port > 65535 or len(dataset_quanta_paths) == 0 or (store_weights_directly_in_archive_tmp_dir is True and archive_tmp_dir is None) or (store_weights_directly_in_archive_tmp_dir is True and os.path.isdir(archive_tmp_dir) is False):
			raise ValueError("Node constructor")
		elif any([type(path) != str or os.path.isfile(path) is False for path in dataset_quanta_paths]) or os.path.isfile(test_set_path) is False or os.path.isfile(val_set_path) is False:
			raise ValueError("Node constructor 2")

		#
		# Initialize common variables
		#

		super().__init__(logger, node_id, host, port, already_available_peers, allowed_to_write_redudant_log_messages)
		
		# Semaphores to limit the number of parallel trainings and validations
		self._malicious_training_semaphore = malicious_training_semaphore
		self._honest_training_semaphore = honest_training_semaphore
		self._validation_semaphore = validation_semaphore

		# Dataset informations
		self._dataset_quanta_paths = dataset_quanta_paths

		# Validation informations
		self._num_of_validation_in_progress = 0
		self._num_of_validation_in_progress_lock = Lock()
		self._validation_mechanism = None
		self._validation_mechanism_type = None
		self._num_of_validators = 1
		self._validators_list = []
		self._validators_list_lock = Lock()
		self._positive_threshold_to_pass_validation = 1

		# Aggregation informations
		self._latest_model_filename = None
		self._aggregation_strategy_type = None
		self._aggregation_strategy = None

		# Training informations
		self._lazy_loading = lazy_loading
		self._batch_size = batch_size
		self._fit_epochs = fit_epochs
		self._do_training_event = Event()

		# Validation set informations
		self._val_set_path = val_set_path

		# Test set informations
		self._test_set_path = test_set_path
		self._test_set_validation_of_global_model_after_each_round = test_set_validation_of_global_model_after_each_round

		# Genesis informations
		self._model_architecture = None
		self._model_to_fit_when_not_validator = None
		self._model_lock = Lock()

		# Node informations
		self._is_training = False
		self._training_lock = Lock()
		self._is_validator = False
		self._is_active_trainer_in_first_round = is_active_trainer_in_first_round
		self._model_aggregation_in_progress = Event()

		# Generic informations
		self._archive_host = archive_host
		self._archive_port = archive_port
		self._allowed_to_store_weights_directly_in_archive_tmp_dir = store_weights_directly_in_archive_tmp_dir
		self._archive_tmp_dir = archive_tmp_dir
		self._perc_of_nodes_active_in_a_round = 1.0
		self._max_num_of_aggregation_rounds = 1
		self._aggregation_round = 1
		self._aggregation_round_lock = Lock()
		self._aggregation_round_during_aggregation = 1
		self._num_of_messages_being_handled = 0
		self._num_of_messages_being_handled_lock = Lock()

	# IMPORTANT: It must be overridden by the subclasses
	def run(self):
		super().run()

	def is_validator(self):
		with self._validators_list_lock:
			return self._is_validator
	
	def aggregation_round(self):
		with self._aggregation_round_lock:
			return self._aggregation_round

	def start(self):
		Thread(target= self._create_and_upload_peer_update_thread, daemon=True, name=f"{self._node_name} fit thread").start()
		super().start()

	def stop(self):

		super().stop()

		self._clear_training_and_validation_variables()
		
		self._do_training_event.set()

		if self._allowed_to_write_redudant_log_messages:
			self._logger.record(msg = f"Node correctly stopped", logLevel = diagnostic.DEBUG, identifier= self._str_identifier)

	#
	# Methods to interact with the archive
	#

	def _handle_request_with_archive(self, message_to_send: bytes, max_number_of_connection_attempts: int = 3):

		if type(message_to_send) != bytes or type(max_number_of_connection_attempts) != int:
			raise TypeError("Node _handle_request_with_archive method")

		s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
		socket_is_connected = False
		response = None

		try:
			for _ in range(max_number_of_connection_attempts):
				try:
					s.connect((self._archive_host, self._archive_port))

				except ConnectionRefusedError as e:
					pass
				
				except Exception as e:
					self._logger.record(msg = f"Error while connecting to the archive", exc = e, logLevel = diagnostic.ERROR, identifier= self._str_identifier)
					raise e
				else:
					socket_is_connected = True
					break

				time.sleep(1)
			else:
				raise Exception(f"Error while connecting to the archive. Archive is unreachable. Maximum number of retries reached. Max number of retries: {max_number_of_connection_attempts}")

			s.sendall(message_to_send)

			response = self._handle_socket_recv(s)
			if response is not None:
				response = json.loads(response)
			else:
				raise Exception("Error while receiving a message from the Archive. Response is None")

		except Exception as e:
			self._logger.record(msg = f"Error while handling a request with the archive", exc = e, logLevel = diagnostic.ERROR, identifier= self._str_identifier)
			raise e

		finally:
			if socket_is_connected:
				s.shutdown(socket.SHUT_RDWR)

			s.close()

		return response

	def _download_blockchain_block_from_archive(self, block_hash: (str | None) = None):
		'''
		Dowload a blockchain block from the archive. If the block hash is None, the last block is downloaded
		'''

		if type(block_hash) != str and block_hash is not None:
			raise TypeError("Node _download_blockchain_block_from_archive method")
		
		try:
			if block_hash is None:
				request = am.GenericArchiveRequest.builder(am.ArchiveRequestTypes.DOWNLOAD_LAST_BLOCKCHAIN_BLOCK, am.DownloadLastBlockchainBlockRequestBody.builder(self._peer_id), self.aggregation_round())
			else:
				request = am.GenericArchiveRequest.builder(am.ArchiveRequestTypes.DOWNLOAD_BLOCKCHAIN_BLOCK, am.DownloadBlockchainBlockRequestBody.builder(self._peer_id, block_hash), self.aggregation_round())

			response = self._handle_request_with_archive(self._prepare_socket_msg_to_send(request))
			
			if response is None:
				raise Exception("Archive response is None")
			elif response[am.GenericArchiveResponse.SUBTYPE] == am.ArchiveResponseTypes.GENERIC_ERROR:
				raise Exception(f"Archive response is an error message. Response: {response}")
			elif response[am.GenericArchiveResponse.SUBTYPE] == am.ArchiveResponseTypes.INVALID_ROUND_NUMBER:
				raise InvalidRoundException()
			elif response[am.GenericArchiveResponse.SUBTYPE] != am.ArchiveResponseTypes.DOWNLOAD_BLOCKCHAIN_BLOCK:
				raise ValueError(f"Invalid response type. Response: {response}")
			
			block_json = response[am.GenericArchiveResponse.BODY][am.DownloadBlockchainBlockResponseBody.BLOCK]
			block_type = response[am.GenericArchiveResponse.BODY][am.DownloadBlockchainBlockResponseBody.BLOCK_TYPE]

			return block_json, block_type
		except InvalidRoundException as e:
			self._logger.record(msg = f"Invalid round number. Request type: {am.ArchiveRequestTypes.DOWNLOAD_BLOCKCHAIN_BLOCK}. Block hash: {block_hash}. Message round: {response[am.GenericArchiveResponse.ROUND]}. Node round: {self.aggregation_round()}", logLevel = diagnostic.WARNING, identifier= self._str_identifier)
			raise e
		except Exception as e:
			self._logger.record(msg = f"Error while downloading a blockchain block from the archive. Block hash: {block_hash}", exc = e, logLevel = diagnostic.ERROR, identifier= self._str_identifier)
			raise e

	def _download_genesis_block_from_archive(self, genesis_block_from_json_method):
		"""
		Download the genesis block from the archive and parse the informations
		"""
		
		try:
			request = am.GenericArchiveRequest.builder(am.ArchiveRequestTypes.DOWNLOAD_GENESIS, am.DownloadGenesisRequestBody.builder(self._peer_id), self.aggregation_round())			
			response = self._handle_request_with_archive(self._prepare_socket_msg_to_send(request))

			if response is None:
				raise Exception("Archive response is None")
			elif response[am.GenericArchiveResponse.SUBTYPE] == am.ArchiveResponseTypes.GENERIC_ERROR:
				raise Exception(f"Archive response is an error message. Response: {response}")
			elif response[am.GenericArchiveResponse.SUBTYPE] == am.ArchiveResponseTypes.INVALID_ROUND_NUMBER:
				raise InvalidRoundException()
			elif response[am.GenericArchiveResponse.SUBTYPE] != am.ArchiveResponseTypes.DOWNLOAD_BLOCKCHAIN_BLOCK:
				raise ValueError(f"Invalid response type. Response: {response}")

			block_json = response[am.GenericArchiveResponse.BODY][am.DownloadBlockchainBlockResponseBody.BLOCK]
			block_type = response[am.GenericArchiveResponse.BODY][am.DownloadBlockchainBlockResponseBody.BLOCK_TYPE]

			if block_type != BlockType.GENESIS:
				raise Exception(f"Invalid block type for genesis block. Received block type: {block_type}")

			block = genesis_block_from_json_method(block_json)
			return block

		except InvalidRoundException as e:
			self._logger.record(msg = f"Invalid round number. Request type: {am.ArchiveRequestTypes.DOWNLOAD_GENESIS}. Message round: {response[am.GenericArchiveResponse.ROUND]}. Node round: {self.aggregation_round()}", logLevel = diagnostic.WARNING, identifier= self._str_identifier)
			raise e
		except Exception as e:
			self._logger.record(msg = f"Error while downloading the genesis block from the archive", exc = e, logLevel = diagnostic.ERROR, identifier= self._str_identifier)
			raise e

	def _get_authorization_to_read_model_weights_directly_from_file(self, model_name: str):
		"""
		Get the authorization to read the model weights directly from the file
		"""

		try:
			if type(model_name) != str:
				raise TypeError("Node _get_authorization_to_read_model_weights_directly_from_file method")

			request = am.GenericArchiveRequest.builder(am.ArchiveRequestTypes.AUTHORIZATION_TO_READ_AGGREGATED_MODEL_DIRECTLY, am.AuthorizationToReadAggregatedModelDirectlyRequestBody.builder(self._peer_id, model_name), self.aggregation_round())			
			response = self._handle_request_with_archive(self._prepare_socket_msg_to_send(request))
			
			if response is None:
				raise Exception("Archive response is None")
			elif response[am.GenericArchiveResponse.SUBTYPE] == am.ArchiveResponseTypes.GENERIC_ERROR:
				raise Exception(f"Archive response is an error message. Response: {response}")
			elif response[am.GenericArchiveResponse.SUBTYPE] == am.ArchiveResponseTypes.INVALID_ROUND_NUMBER:
				raise InvalidRoundException()
			elif response[am.GenericArchiveResponse.SUBTYPE] != am.ArchiveResponseTypes.GENERIC_SUCCESS:
				raise ValueError(f"Invalid response type. Response: {response}")

		except InvalidRoundException as e:
			self._logger.record(msg = f"Invalid round number. Request type: {am.ArchiveRequestTypes.AUTHORIZATION_TO_READ_AGGREGATED_MODEL_DIRECTLY}. Model name: {model_name}. Message round: {response[am.GenericArchiveResponse.ROUND]}. Node round: {self.aggregation_round()}", logLevel = diagnostic.WARNING, identifier= self._str_identifier)
			raise e
		except Exception as e:
			self._logger.record(msg = f"Error while getting the authorization to read the model weights directly from the file", exc = e, logLevel = diagnostic.ERROR, identifier= self._str_identifier)
			raise e

	def _download_model_from_archive(self, model_name: str):
		"""
		Download the aggregated model from the archive
		"""
		try:
			if type(model_name) != str:
				raise TypeError("Node _download_model_from_archive method")

			request = am.GenericArchiveRequest.builder(am.ArchiveRequestTypes.DOWNLOAD_AGGREGATED_MODEL, am.DownloadAggregatedModelRequestBody.builder(self._peer_id, model_name), self.aggregation_round())
			response = self._handle_request_with_archive(self._prepare_socket_msg_to_send(request))
			
			if response is None:
				raise Exception("Archive response is None")
			elif response[am.GenericArchiveResponse.SUBTYPE] == am.ArchiveResponseTypes.GENERIC_ERROR:
				raise Exception(f"Archive response is an error message. Response: {response}")
			elif response[am.GenericArchiveResponse.SUBTYPE] == am.ArchiveResponseTypes.INVALID_ROUND_NUMBER:
				raise InvalidRoundException()
			elif response[am.GenericArchiveResponse.SUBTYPE] != am.ArchiveResponseTypes.DOWNLOAD_AGGREGATED_MODEL:
				raise ValueError(f"Invalid response type. Response: {response}")
			
			return response[am.GenericArchiveResponse.BODY]

		except InvalidRoundException as e:
			self._logger.record(msg = f"Invalid round number. Request type: {am.ArchiveRequestTypes.DOWNLOAD_AGGREGATED_MODEL}. Model name: {model_name}. Message round: {response[am.GenericArchiveResponse.ROUND]}. Node round: {self.aggregation_round()}", logLevel = diagnostic.WARNING, identifier= self._str_identifier)
			raise e
		except Exception as e:
			self._logger.record(msg = f"Error while downloading the aggregated model from the archive", exc = e, logLevel = diagnostic.ERROR, identifier= self._str_identifier)
			raise e

	def _get_authorization_to_read_update_directly_from_file(self, update_name: str):
		"""
		Get the authorization to read the update directly from the file
		"""

		try:
			if type(update_name) != str:
				raise TypeError("Node get_authorization_to_read_update_directly_from_file method")

			request = am.GenericArchiveRequest.builder(am.ArchiveRequestTypes.AUTHORIZATION_TO_READ_UPDATE_DIRECTLY, am.AuthorizationToReadUpdateDirectlyRequestBody.builder(self._peer_id, update_name), self.aggregation_round())
			response = self._handle_request_with_archive(self._prepare_socket_msg_to_send(request))
			
			if response is None:
				raise Exception("Archive response is None")
			elif response[am.GenericArchiveResponse.SUBTYPE] == am.ArchiveResponseTypes.GENERIC_ERROR:
				raise Exception(f"Archive response is an error message. Response: {response}")
			elif response[am.GenericArchiveResponse.SUBTYPE] == am.ArchiveResponseTypes.INVALID_ROUND_NUMBER:
				raise InvalidRoundException()
			elif response[am.GenericArchiveResponse.SUBTYPE] != am.ArchiveResponseTypes.GENERIC_SUCCESS:
				raise ValueError(f"Invalid response type. Response: {response}")

		except InvalidRoundException as e:
			self._logger.record(msg = f"Invalid round number. Request type: {am.ArchiveRequestTypes.AUTHORIZATION_TO_READ_UPDATE_DIRECTLY}. Update name: {update_name}. Message round: {response[am.GenericArchiveResponse.ROUND]}. Node round: {self.aggregation_round()}", logLevel = diagnostic.WARNING, identifier= self._str_identifier)
			raise e
		except Exception as e:
			self._logger.record(msg = f"Error while getting the authorization to read the update directly from the file", exc = e, logLevel = diagnostic.ERROR, identifier= self._str_identifier)
			raise e

	def _download_update_from_archive(self, update_name: str):
		"""
		Download the update corresponding to the update_name from the archive and return the weights and the number of samples
		"""

		try:
			if type(update_name) != str:
				raise TypeError("Node _download_update_from_archive method")

			request = am.GenericArchiveRequest.builder(am.ArchiveRequestTypes.DOWNLOAD_PEER_UPDATE, am.DownloadPeerUpdateRequestBody.builder(self._peer_id, update_name), self.aggregation_round())
			response = self._handle_request_with_archive(self._prepare_socket_msg_to_send(request))
			
			if response is None:
				raise Exception("Archive response is None")
			elif response[am.GenericArchiveResponse.SUBTYPE] == am.ArchiveResponseTypes.GENERIC_ERROR:
				raise Exception(f"Archive response is an error message. Response: {response}")
			elif response[am.GenericArchiveResponse.SUBTYPE] == am.ArchiveResponseTypes.INVALID_ROUND_NUMBER:
				raise InvalidRoundException()
			elif response[am.GenericArchiveResponse.SUBTYPE] != am.ArchiveResponseTypes.DOWNLOAD_PEER_UPDATE:
				raise ValueError(f"Invalid response type. Response: {response}")

			body = response[am.GenericArchiveResponse.BODY]
			updater_id = body[am.DownloadPeerUpdateResponseBody.PEER_ID]
			update_weights = body[am.DownloadPeerUpdateResponseBody.UPDATE_JSON_WEIGHTS]
			num_of_samples = body[am.DownloadPeerUpdateResponseBody.NUM_SAMPLES]

			return updater_id, update_weights, num_of_samples
		
		except InvalidRoundException as e:
			self._logger.record(msg = f"Invalid round number. Request type: {am.ArchiveRequestTypes.DOWNLOAD_PEER_UPDATE}. Update name: {update_name}. Message round: {response[am.GenericArchiveResponse.ROUND]}. Node round: {self.aggregation_round()}", logLevel = diagnostic.WARNING, identifier= self._str_identifier)
			raise e
		except Exception as e:
			self._logger.record(msg = f"Error while downloading the update from the archive", exc = e, logLevel = diagnostic.ERROR, identifier= self._str_identifier)
			raise e

	#
	# Methods to interact with the storage
	#

	def _read_update_from_disk(self, file_name: str) -> tuple:
		
		if type(file_name) != str:
			raise TypeError("Node _read_update_from_disk method")
		elif self._allowed_to_store_weights_directly_in_archive_tmp_dir is False:
			raise Exception("Node _read_update_from_disk method. Node is not allowed to store weights directly in the archive temporary directory")
		elif self._archive_tmp_dir is None:
			raise Exception("Node _read_update_from_disk method. Archive temporary directory is None")
		elif os.path.isdir(self._archive_tmp_dir) is False:
			raise Exception("Node _read_update_from_disk method. Archive temporary directory does not exist")
		elif os.path.isfile(os.path.join(self._archive_tmp_dir, file_name)) is False:
			raise Exception("Node _read_update_from_disk method. File does not exist")
		
		with open(os.path.join(self._archive_tmp_dir, file_name), "r") as f:
			update = json.load(f)
		
		return update[PeerUpdate.PEER_ID], update[PeerUpdate.WEIGHTS], update[PeerUpdate.NUM_SAMPLES]

	def _read_model_from_disk(self, file_name: str) -> tuple:
		
		if type(file_name) != str:
			raise TypeError("Node _read_model_from_disk method")
		elif self._allowed_to_store_weights_directly_in_archive_tmp_dir is False:
			raise Exception("Node _read_model_from_disk method. Node is not allowed to store weights directly in the archive temporary directory")
		elif self._archive_tmp_dir is None:
			raise Exception("Node _read_model_from_disk method. Archive temporary directory is None")
		elif os.path.isdir(self._archive_tmp_dir) is False:
			raise Exception("Node _read_model_from_disk method. Archive temporary directory does not exist")
		elif os.path.isfile(os.path.join(self._archive_tmp_dir, file_name)) is False:
			raise Exception("Node _read_model_from_disk method. File does not exist")
		
		with open(os.path.join(self._archive_tmp_dir, file_name), "r") as f:
			model_info = json.load(f)
		
		return model_info

	def _store_update_on_disk(self, file_name: str, weights: list, num_samples_training_set: (int | None) = None):
		"""
		Store the node update on disk inside the archive temporary directory
		"""
		
		if self._allowed_to_store_weights_directly_in_archive_tmp_dir is False:
			raise Exception("Node _store_update_on_disk method. Node is not allowed to store weights directly in the archive temporary directory")
		elif self._archive_tmp_dir is None:
			raise Exception("Node _store_update_on_disk method. Archive temporary directory is None")
		elif os.path.isdir(self._archive_tmp_dir) is False:
			raise Exception("Node _store_update_on_disk method. Archive temporary directory does not exist")
		elif type(file_name) != str or type(weights) != list or (num_samples_training_set is not None and type(num_samples_training_set) != int):
			raise TypeError("Node _store_update_on_disk method")
		
		with open(os.path.join(self._archive_tmp_dir, file_name), "w") as f:
			json.dump({PeerUpdate.WEIGHTS: weights, PeerUpdate.NUM_SAMPLES: num_samples_training_set, PeerUpdate.PEER_ID: self._peer_id, PeerUpdate.NAME: file_name}, f)

	def _store_model_on_disk(self, file_name: str, weights: list, optimizer: dict | None = None):
		"""
		Store the model on disk inside the archive temporary directory
		"""
		
		if self._allowed_to_store_weights_directly_in_archive_tmp_dir is False:
			raise Exception("Node _store_model_on_disk method. Node is not allowed to store weights directly in the archive temporary directory")
		elif self._archive_tmp_dir is None:
			raise Exception("Node _store_model_on_disk method. Archive temporary directory is None")
		elif os.path.isdir(self._archive_tmp_dir) is False:
			raise Exception("Node _store_model_on_disk method. Archive temporary directory does not exist")
		elif type(file_name) != str or type(weights) != list or (optimizer is not None and type(optimizer) != dict):
			raise TypeError("Node _store_model_on_disk method")
		
		with open(os.path.join(self._archive_tmp_dir, file_name), "w") as f:
			json.dump({AggregatedModel.WEIGHTS: weights, AggregatedModel.NAME: file_name, AggregatedModel.OPTIMIZER: optimizer}, f)

	#
	# Utils methods
	#

	def _get_val_and_agg_mechanism(self, validation_params: dict, aggregation_params: dict):
		"""
		Set the validation and aggregation mechanisms
		"""

		try:
			if type(validation_params) != dict or type(aggregation_params) != dict:
				raise TypeError("Node _get_val_and_agg_mechanism method")

			validation_mechanism_type = validation_params[cm.ConfigGenericValidationAlgorithmParams.TYPE]

			# Define the validation mechanism
			if validation_mechanism_type == cm.ValidationAlgorithmType.LOCAL_DATASET_VALIDATION:
				# Minimum score is, at first, defined by the genesis block then it will be changed according to the scores of the updates
				validation_mechanism = LocalDatasetValidation(self._validation_semaphore, self._dataset_quanta_paths, validation_params[va.LocalDatasetUsedForValidationParams.MIN_UPDATE_VALIDATION_SCORE_FIRST_ROUND], validation_params[va.LocalDatasetUsedForValidationParams.MINIMUM_NUM_OF_UPDATES_BETWEEN_AGGREGATIONS], validation_params[va.LocalDatasetUsedForValidationParams.MAXIMUM_NUM_OF_UPDATES_BETWEEN_AGGREGATIONS], validation_params[va.LocalDatasetUsedForValidationParams.COUNT_DOWN_TIMER_TO_START_AGGREGATION], self._validators_list, self._positive_threshold_to_pass_validation, self._lazy_loading, self._batch_size, VERBOSE, TESTING_OPS)
			elif validation_mechanism_type == cm.ValidationAlgorithmType.KRUM:
				validation_mechanism = KrumValidation(self._validation_semaphore, self._peer_id, validation_params[va.KrumValidationParams.MINIMUM_NUM_OF_UPDATES_NEEDED_TO_START_VALIDATION], validation_params[va.KrumValidationParams.MAXIMUM_NUM_OF_UPDATES_NEEDED_TO_START_VALIDATION], validation_params[va.KrumValidationParams.NUM_OF_UPDATES_TO_VALIDATE_NEGATIVELY], validation_params[va.KrumValidationParams.COUNT_DOWN_TIMER_TO_START_VALIDATION], validation_params[va.KrumValidationParams.DISTANCE_FUNCTION], self._validators_list)
			elif validation_mechanism_type == cm.ValidationAlgorithmType.TRIMMED_MEAN:
				validation_mechanism = TrimmedMeanValidation(self._validation_semaphore, self._peer_id, validation_params[va.TrimmedMeanValidationParams.MINIMUM_NUM_OF_UPDATES_NEEDED_TO_START_VALIDATION], validation_params[va.TrimmedMeanValidationParams.MAXIMUM_NUM_OF_UPDATES_NEEDED_TO_START_VALIDATION], validation_params[va.TrimmedMeanValidationParams.TRIMMING_PERCENTAGE], validation_params[va.TrimmedMeanValidationParams.COUNT_DOWN_TIMER_TO_START_VALIDATION], validation_params[va.TrimmedMeanValidationParams.DISTANCE_FUNCTION], self._validators_list)
			elif validation_mechanism_type == cm.ValidationAlgorithmType.PASS_WEIGHTS:
				validation_mechanism = PassWeightsValidation(self._validation_semaphore, validation_params[va.PassWeightsValidationParams.MINIMUM_NUM_OF_UPDATES_BETWEEN_AGGREGATIONS], validation_params[va.PassWeightsValidationParams.MAXIMUM_NUM_OF_UPDATES_BETWEEN_AGGREGATIONS], validation_params[va.PassWeightsValidationParams.COUNT_DOWN_TIMER_TO_START_AGGREGATION], self._validators_list, VERBOSE)
			elif validation_mechanism_type == cm.ValidationAlgorithmType.PASS_GRADIENTS:
				validation_mechanism = PassGradientsValidation(self._validation_semaphore, self._peer_id, validation_params[va.PassGradientsValidationParams.MINIMUM_NUM_OF_UPDATES_NEEDED_TO_START_VALIDATION], validation_params[va.PassGradientsValidationParams.MAXIMUM_NUM_OF_UPDATES_NEEDED_TO_START_VALIDATION], validation_params[va.PassGradientsValidationParams.COUNT_DOWN_TIMER_TO_START_VALIDATION], self._validators_list)
			elif validation_mechanism_type == cm.ValidationAlgorithmType.GLOBAL_DATASET_VALIDATION:
				validation_mechanism = GlobalDatasetValidation(self._validation_semaphore, self._val_set_path, validation_params[va.GlobalDatasetUsedForValidationParams.MIN_UPDATE_VALIDATION_SCORE_FIRST_ROUND], validation_params[va.GlobalDatasetUsedForValidationParams.MINIMUM_NUM_OF_UPDATES_BETWEEN_AGGREGATIONS], validation_params[va.GlobalDatasetUsedForValidationParams.MAXIMUM_NUM_OF_UPDATES_BETWEEN_AGGREGATIONS], validation_params[va.GlobalDatasetUsedForValidationParams.COUNT_DOWN_TIMER_TO_START_AGGREGATION], self._validators_list, self._positive_threshold_to_pass_validation, self._lazy_loading, self._batch_size, VERBOSE, TESTING_OPS)
			else:
				raise ValueError("Node _get_val_and_agg_mechanism method. Invalid validation algorithm type in the genesis block")
			
			aggregation_strategy_type = aggregation_params[cm.ConfigGenericValidationAlgorithmParams.TYPE]
			if aggregation_strategy_type == cm.AggregationAlgorithmType.FEDAVG:
				aggregation_strategy = FlowerFedAvg()
			elif aggregation_strategy_type == cm.AggregationAlgorithmType.MEAN:
				aggregation_strategy = MeanAgg()
			elif aggregation_strategy_type == cm.AggregationAlgorithmType.MEDIAN:
				aggregation_strategy = MedianAgg()
			elif aggregation_strategy_type == cm.AggregationAlgorithmType.TRIMMED_MEAN:
				aggregation_strategy = TrimmedMeanAgg(aggregation_params[ag.TrimmedMeanAggParams.TRIMMING_PERCENTAGE])
			else:
				raise ValueError("Node _get_val_and_agg_mechanism method. Invalid aggregation algorithm type in the genesis block")
			
			return validation_mechanism_type, validation_mechanism, aggregation_strategy_type, aggregation_strategy

		except Exception as e:
			self._logger.record(msg = f"Error while setting the validation and aggregation mechanisms", exc = e, logLevel = diagnostic.ERROR, identifier= self._str_identifier)
			raise e

	def _get_random_elements_from_list(self, elements: list, num_of_elements: int, seed_num: int) -> list:
		"""
		Return a list of random elements from the list
		"""
		if type(elements) != list or type(num_of_elements) != int or type(seed_num) != int:
			raise TypeError("Node _get_random_elements_from_list method")
		
		numpy_random_generator = np.random.default_rng(seed= seed_num)
		random_choice = numpy_random_generator.choice(elements, num_of_elements, replace=False).tolist()
		del numpy_random_generator
		return random_choice
	
	def _create_weights_filename(self):
		"""
		Use the peer_id and the timestamp to create a unique name for the model.
		"""

		combined_string = f"{self._peer_id}{time.time()}"
		hash_object = hashlib.sha256(combined_string.encode())
		name = hash_object.hexdigest()

		return name
	
	#
	# Methods to handle the update validation
	#

	# Multiple updates validation mechanism
	def _validate_update_by_means_of_gradients(self, update_name: str, updater_id: int, update_weights: list) -> None:
		try:
			if type(update_name) != str or type(updater_id) != int or type(update_weights) != list:
				raise TypeError("Node _validate_update_by_means_of_gradients method")
			elif self._validation_mechanism_type not in cm.GradientsBasedValidationAlgorithmType.list():
				raise Exception("Node _validate_update_by_means_of_gradients method. Validation mechanism is not in the list of the gradients based validation mechanisms")

			min_num_of_updates_reached = self._validation_mechanism.add_update_name_to_validate(update_name)
			if min_num_of_updates_reached is None:
				if self._validation_mechanism.must_stop_accepting_new_updates() is True:
					if self._allowed_to_write_redudant_log_messages:
						self._logger.record(msg = f"Update not considered in the validation because the validation mechanism is not accepting new updates. Update name: {update_name}", logLevel = diagnostic.DEBUG, identifier= self._str_identifier)
				else:
					raise Exception("Node _validate_update_by_means_of_gradients method. Update not considered in the validation but we don't know why")
				
			else:
				if min_num_of_updates_reached is True:
					# Start the validation thread used to check when the validation is ready to start and then perform the validation
					Thread(target= self._perform_validation_by_means_of_gradients, daemon=True, name=f"{self._node_name} {self._validation_mechanism_type} validation thread").start()
					
					if self._allowed_to_write_redudant_log_messages:
						self._logger.record(msg = f"{self._validation_mechanism_type} validation thread started", logLevel = diagnostic.DEBUG, identifier= self._str_identifier)

				if self._allowed_to_write_redudant_log_messages:
					self._logger.record(msg = f"Update added to the validation mechanism. Update name: {update_name}", logLevel = diagnostic.DEBUG, identifier= self._str_identifier)

		except Exception as e:
			self._logger.record(msg = f"Error while validating the update by means of {self._validation_mechanism_type} validation mechanism", exc = e, logLevel = diagnostic.ERROR, identifier= self._str_identifier)
			raise e

	def _perform_validation_by_means_of_gradients(self):
		try:
			current_round = self.aggregation_round()

			while self._validation_mechanism.is_validation_ready_to_start() is False:
				time.sleep(1)
			
			update_names_to_validate = self._validation_mechanism.get_list_of_update_names_to_validate()
			update_weights_to_validate = list()
			update_uploader_ids = list()
			tmp_update_weights = None

			for name in update_names_to_validate:				
				if self._allowed_to_store_weights_directly_in_archive_tmp_dir:
					self._get_authorization_to_read_update_directly_from_file(name)	# Archive will confirm the authorization to read the update directly from the file
					tmp_updater_id, tmp_update_weights, _ = self._read_update_from_disk(name)

				else:
					# Validate the update
					tmp_updater_id, tmp_update_weights, _ = self._download_update_from_archive(name)
				
				update_weights_to_validate.append(tmp_update_weights)
				update_uploader_ids.append(tmp_updater_id)

			update_scores, good_updates_indexes = self._validation_mechanism.perform_validation(update_weights_to_validate)
			
			validation_results = list()
			good_updates_names_scores = [(update_names_to_validate[index], update_scores[index]) for index in good_updates_indexes]

			for index, name in enumerate(update_names_to_validate):
				if index in good_updates_indexes:
					validation_results.append({ValidationResultsListElemFields.UPDATE_NAME: name, ValidationResultsListElemFields.VALIDATION_RESULT: True, ValidationResultsListElemFields.VALIDATION_SCORE: update_scores[index]})
				else:
					validation_results.append({ValidationResultsListElemFields.UPDATE_NAME: name, ValidationResultsListElemFields.VALIDATION_RESULT: False, ValidationResultsListElemFields.VALIDATION_SCORE: update_scores[index]})
			
			if self._allowed_to_write_redudant_log_messages:
				self._logger.record(msg = f"Updates to validate: {update_names_to_validate}", logLevel = diagnostic.DEBUG, identifier= self._str_identifier, skipLengthTruncation= True)
				self._logger.record(msg = f"{self._validation_mechanism_type} validation scores: {update_scores}. Good updates indexes: {good_updates_indexes}", logLevel = diagnostic.DEBUG, identifier= self._str_identifier, skipLengthTruncation= True)
				self._logger.record(msg = f"Good updates names and scores: {good_updates_names_scores}", logLevel = diagnostic.DEBUG, identifier= self._str_identifier, skipLengthTruncation= True)

			# I want to force the garbage collector to free the memory before sending the validation results and, consequently, let another validator start the validation process
			del update_weights_to_validate, tmp_update_weights
			gc.collect()

			if self._model_aggregation_in_progress.is_set() is False and self.is_peer_alive() and current_round == self.aggregation_round():
				self._send_message(nm.BroadcastMultipleValidationResults.builder(validation_results, self._peer_id, current_round))
				self._handle_multiple_validation_results(validation_results, self._peer_id)
			
			else:
				self._logger.record(msg = f"Validation results not sent because the model aggregation is in progress or the node is not alive", logLevel = diagnostic.DEBUG, identifier= self._str_identifier)

		except Exception as e:
			self._logger.record(msg = f"Error while performing the validation by means of the gradients based validation mechanism", exc = e, logLevel = diagnostic.ERROR, identifier= self._str_identifier)

	def _handle_multiple_validation_results(self, results: list, validator_id: int):
		"""
		Handle the multiple validation scores received from the validators
		"""
		try:
			if type(results) != list or type(validator_id) != int:
				raise TypeError("Node _handle_multiple_validation_results method")
			
			elif any(key not in elem for key in ValidationResultsListElemFields.list() for elem in results):
				raise ValueError("Node _handle_multiple_validation_results method. Invalid validation results list")

			elif self._validation_mechanism_type not in cm.GradientsBasedValidationAlgorithmType.list():
				raise Exception("Node _handle_multiple_validation_results method. Validation mechanism is not in the list of the gradients based validation mechanisms")
			
			elif self.is_validator() is False:
				raise Exception("Node _handle_multiple_validation_results method. Node is not a validator")

			# Convert results in dictionary
			dict_results = dict()

			for elem in results:
				dict_results[elem[ValidationResultsListElemFields.UPDATE_NAME]] = (elem[ValidationResultsListElemFields.VALIDATION_RESULT], elem[ValidationResultsListElemFields.VALIDATION_SCORE])
			
			self._validation_mechanism.handle_new_validation_results(dict_results, validator_id)

			# Aggregation start only when a message from each validator has been received
			if self._validation_mechanism.is_aggregation_ready():
				if self._model_aggregation_in_progress.is_set():
					raise Exception(f"Node _handle_multiple_validation_results method. Model aggregation is already in progress. This should not happen")
				
				# We need to avoid problems with concurrency caused by the _clear_training_and_validation_variables method, so, we need to get the variables before calling the method
				aggregation_round = self.aggregation_round()
				with self._validators_list_lock:
					validators_list = self._validators_list.copy()
				
				seed = sum(validators_list) + aggregation_round

				self._aggregation_round_during_aggregation = aggregation_round
				self._model_aggregation_in_progress.set()

				if self._allowed_to_write_redudant_log_messages:
					self._logger.record(msg = f"Node has reached the required number of validation results", logLevel = diagnostic.DEBUG, identifier= self._str_identifier)

				self._clear_training_and_validation_variables()

				perform_aggregation = False
				
				node_responsible_of_model_aggregation = (self._get_random_elements_from_list(validators_list, 1, seed))[0]

				if self._allowed_to_write_redudant_log_messages:
					self._logger.record(msg = f"Node responsible of model aggregation elected. Node id: {node_responsible_of_model_aggregation}", logLevel = diagnostic.DEBUG, identifier= self._str_identifier)

				if node_responsible_of_model_aggregation == self._peer_id:
					self._logger.record(msg = f"Node is responsible of model aggregation. Round: {aggregation_round}", logLevel = diagnostic.DEBUG, identifier= self._str_identifier)
					perform_aggregation = True

				if perform_aggregation:
					validation_results = self._validation_mechanism.get_validation_results()

					update_positive_votes_counts_aggregated_validation_score = dict()								# Update name -> (positive votes count, aggregated validation score)

					for validator_id in validation_results:
						for update_name in validation_results[validator_id]:
							if update_name not in update_positive_votes_counts_aggregated_validation_score:
								update_positive_votes_counts_aggregated_validation_score[update_name] = (0, 0)

							# Check if the validator has voted positively
							if validation_results[validator_id][update_name][0] is True:
								if validation_results[validator_id][update_name][1] == 0:
									raise Exception(f"Node _handle_multiple_validation_results method. Validation score is 0. Validator id: {validator_id}. Update name: {update_name}")

								aggregated_validation_score = update_positive_votes_counts_aggregated_validation_score[update_name][1] + min([1 / validation_results[validator_id][update_name][1], 100])		# 100 is the maximum score that a validator can give to an update (these scores are used to elect the new committee)

								update_positive_votes_counts_aggregated_validation_score[update_name] = (update_positive_votes_counts_aggregated_validation_score[update_name][0] + 1, aggregated_validation_score)
				
					update_names_and_aggregated_scores_to_aggregate = dict()

					for update_name in update_positive_votes_counts_aggregated_validation_score:
						if update_positive_votes_counts_aggregated_validation_score[update_name][0] >= self._positive_threshold_to_pass_validation:
							update_names_and_aggregated_scores_to_aggregate[update_name] = update_positive_votes_counts_aggregated_validation_score[update_name][1]
							self._logger.record(msg = f"Update {update_name} passed the validation. Positive votes: {update_positive_votes_counts_aggregated_validation_score[update_name][0]}. Aggregated score assigned: {update_positive_votes_counts_aggregated_validation_score[update_name][1]}", logLevel = diagnostic.DEBUG, identifier= self._str_identifier)
						else:
							self._logger.record(msg = f"Update {update_name} did not pass the validation. Positive votes: {update_positive_votes_counts_aggregated_validation_score[update_name][0]}", logLevel = diagnostic.DEBUG, identifier= self._str_identifier)

					# Aggregate the updates
					self._create_and_upload_aggregated_model_block(update_names_and_aggregated_scores_to_aggregate)

		except Exception as e:
			self._logger.record(msg = f"Error while handling the multiple validation scores", exc = e, logLevel = diagnostic.ERROR, identifier= self._str_identifier)
			raise e
	
	# Local dataset validation mechanism
	def _validate_update_by_means_of_weights(self, update_name: str, updater_id: int, update_weights: list):
		if type(update_weights) != list or type(update_name) != str or type(updater_id) != int:
			raise TypeError("Node _validate_update_by_means_of_weights method")
		elif self._validation_mechanism_type not in  cm.WeightsBasedValidationAlgorithmType.list():
			raise Exception("Node _validate_update_by_means_of_weights method. Validation mechanism is not in the list of the weights based validation mechanisms")
		
		result = None
		current_round = self.aggregation_round()

		with self._num_of_validation_in_progress_lock:
			self._num_of_validation_in_progress += 1
		try:
			result = self._validation_mechanism.validate_update(self._model_architecture, update_weights)

			if result is None:
				if self._validation_mechanism.must_stop_accepting_new_updates():
					if self._allowed_to_write_redudant_log_messages:
						self._logger.record(msg = f"Update not validated. Validation mechanism is not accepting new updates. Update name: {update_name}", logLevel = diagnostic.DEBUG, identifier= self._str_identifier)
				else:
					self._logger.record(msg = f"Update not validated. Result is None. Update name: {update_name}", logLevel = diagnostic.ERROR, identifier= self._str_identifier)

			# Memory optimization
			update_weights.clear()

		except Exception as e:
			self._logger.record(msg = f"Node _validate_update_by_means_of_weights method. Error while validating the update. Update name: {update_name}", exc = e, logLevel = diagnostic.ERROR, identifier= self._str_identifier)
			raise e
		
		finally:
			with self._num_of_validation_in_progress_lock:
				self._num_of_validation_in_progress -= 1

				if self._num_of_validation_in_progress < 0:
					raise Exception(f"Invalid number of validation in progress. Number is negative. Number of validation in progress: {self._num_of_validation_in_progress}")

		try:
			if result is not None:
				accuracy, positive_validation = result

				if self._validation_mechanism.must_stop_accepting_new_updates() is False and self.is_peer_alive() and current_round == self.aggregation_round():
					if self._allowed_to_write_redudant_log_messages:
						self._logger.record(msg = f"Update validated. Update name: {update_name}. Positive validation: {positive_validation}. Score: {accuracy}", logLevel = diagnostic.DEBUG, identifier= self._str_identifier)
				
					self._send_message(nm.BroadcastValidationResult.builder(positive_validation, accuracy, update_name, updater_id, self._peer_id, current_round))
					self._handle_single_validation_result(positive_validation, accuracy, update_name, updater_id, self._peer_id)
			
				else:
					if self._allowed_to_write_redudant_log_messages:
						self._logger.record(msg = f"Update validated but ignored because model aggregation is in progress or node is not alive. Update name: {update_name}", logLevel = diagnostic.DEBUG, identifier= self._str_identifier)
			
		except Exception as e:
			self._logger.record(msg = f"Node _validate_update_by_means_of_weights method. Error while sending the validation score. Update name: {update_name}", exc = e, logLevel = diagnostic.ERROR, identifier= self._str_identifier)
			raise e

	def _handle_single_validation_result(self, positive_validation: bool, accuracy: float, update_name: str, updater_id: int, validator_id: int):
		"""
		Handle the validation score received from a peer
		"""
		try:
			if type(positive_validation) != bool or type(accuracy) not in [float, int] or type(update_name) != str or type(updater_id) != int or type(validator_id) != int:
				raise TypeError("Node _handle_single_validation_result method")
			
			elif self._validation_mechanism_type not in cm.WeightsBasedValidationAlgorithmType.list():
				raise Exception("Node _handle_single_validation_result method. Validation mechanism is not in the list of the weights based validation mechanisms")
			
			# I used this lock to serialize the management of the various validation results, otherwise we could have problems with the start of the aggregation		
			with self._validators_list_lock:
				if validator_id not in self._validators_list:
					raise Exception("Node _handle_single_validation_result method. Validator id not in the validators list")		
				elif self._peer_id not in self._validators_list:
					raise Exception("Node _handle_single_validation_result method. Peer id not in the validators list")

				result = self._validation_mechanism.handle_new_validation_result(positive_validation, accuracy, update_name, updater_id, validator_id)

				if result is not None:
					if self._allowed_to_write_redudant_log_messages:
						if result[0] is True:
							self._logger.record(msg = f"Update has enough positive scores. Updante name: {update_name}. Num of positive scores: {result[1]}. Updater id: {updater_id}. Aggregated score: {result[2]}", logLevel = diagnostic.INFO, identifier= self._str_identifier)
						elif result[0] is False:
							self._logger.record(msg = f"Update doesn't have enough positive scores. Updante name: {update_name}. Num of positive scores: {result[1]}. Updater id: {updater_id}", logLevel = diagnostic.WARNING, identifier= self._str_identifier)

					# It is equal to True if the update has reached the min required number of positive scores
					if result[3] is True:
						seed = sum(self._validators_list) + self.aggregation_round()
						
						node_responsible_of_model_aggregation = (self._get_random_elements_from_list(self._validators_list, 1, seed))[0]
						
						if self._allowed_to_write_redudant_log_messages:
							self._logger.record(msg = f"Node responsible of model aggregation elected. Node id: {node_responsible_of_model_aggregation}", logLevel = diagnostic.DEBUG, identifier= self._str_identifier)

						if node_responsible_of_model_aggregation == self._peer_id:
							Thread(target= self._perform_aggretation_after_handling_enough_single_validation_results, daemon=True, name=f"{self._node_name} {self._aggregation_strategy_type} aggregation thread").start()
					
							if self._allowed_to_write_redudant_log_messages:
								self._logger.record(msg = f"Node has reached the minimum number of updates validated positively and it is the aggregator", logLevel = diagnostic.DEBUG, identifier= self._str_identifier)
								self._logger.record(msg = f"{self._aggregation_strategy_type} aggregation thread started", logLevel = diagnostic.DEBUG, identifier= self._str_identifier)				

						else:
							if self._allowed_to_write_redudant_log_messages:
								self._logger.record(msg = f"Node has reached the minimum number of updates validated positively, it is a validator but not the aggregator", logLevel = diagnostic.DEBUG, identifier= self._str_identifier)

				elif self._validation_mechanism.must_stop_accepting_new_updates():
					if self._allowed_to_write_redudant_log_messages:
						self._logger.record(msg = f"Update validation score not handle. Validation mechanism is not accepting new validation scores. Update name: {update_name}", logLevel = diagnostic.DEBUG, identifier= self._str_identifier)

		except Exception as e:
			self._logger.record(msg = f"Error while handling the validation score", exc = e, logLevel = diagnostic.ERROR, identifier= self._str_identifier)
			raise e

	def _perform_aggretation_after_handling_enough_single_validation_results(self):
		"""
		Perform the aggregation after handling enough single validation results
		"""
		try:			
			with self._validators_list_lock:
				if self._is_validator is False:
					raise Exception("Node _perform_aggretation_after_handling_enough_single_validation_results method. Peer id not in the validators list")

			while self._validation_mechanism.is_aggregation_ready() is False:				
				time.sleep(5)

			if self._model_aggregation_in_progress.is_set():
				raise Exception("Node _perform_aggretation_after_handling_enough_single_validation_results method. Model aggregation is already in progress")
			elif self._validation_mechanism.must_stop_accepting_new_updates() is False:
				raise Exception("Node _perform_aggretation_after_handling_enough_single_validation_results method. Validation mechanism is still accepting new updates")

			self._aggregation_round_during_aggregation = self.aggregation_round()
			self._model_aggregation_in_progress.set()

			if self._allowed_to_write_redudant_log_messages:
				self._logger.record(msg = f"Aggregator has reached the required number of updates validated positively", logLevel = diagnostic.DEBUG, identifier= self._str_identifier)

			self._clear_training_and_validation_variables()

			update_names_and_aggregated_scores_to_aggregate = self._validation_mechanism.get_honest_updates_and_aggregated_scores()

			for update_name in update_names_and_aggregated_scores_to_aggregate:
				self._logger.record(msg = f"Update is ready for the aggregation. Update name: {update_name}. Aggregated validation score: {update_names_and_aggregated_scores_to_aggregate[update_name]}", logLevel = diagnostic.INFO, identifier= self._str_identifier)

			# Aggregate the updates
			self._create_and_upload_aggregated_model_block(update_names_and_aggregated_scores_to_aggregate)
		
		except Exception as e:
			self._logger.record(msg = f"Error while performing the aggregation after handling enough single validation results", exc = e, logLevel = diagnostic.ERROR, identifier= self._str_identifier)
			raise e

	# Entry-point method for the update validation
	def _validate_update_created_by_node(self, update_name: str, updater_id: int, update_weights: list):
		"""
		Validate the update created by the node and send the score to the all the other peers
		"""

		try:
			if type(updater_id) != int or type(update_weights) != list or type(update_name) != str:
				raise TypeError("Node _validate_update_created_by_node method")
			
			if self._validation_mechanism_type in cm.GradientsBasedValidationAlgorithmType.list():
				self._validate_update_by_means_of_gradients(update_name, updater_id, update_weights)

			elif self._validation_mechanism_type in cm.WeightsBasedValidationAlgorithmType.list():
				self._validate_update_by_means_of_weights(update_name, updater_id, update_weights)

			else:
				raise NotImplementedError("Node _validate_update_created_by_node method. Validation mechanism not implemented")

		except Exception as e:
			self._logger.record(msg = f"Node _validate_update_created_by_node method. Error while downloading and validating the update from the archive", exc = e, logLevel = diagnostic.ERROR, identifier= self._str_identifier)
			raise e
		finally:
			gc.collect()

	#
	# Generic methods to handle the node state
	#

	def _wait_to_start_next_round(self):
		"""
		Wait for the event to start the training
		"""
		request = am.GenericArchiveRequest.builder(am.ArchiveRequestTypes.READY_FOR_NEXT_ROUND, am.ReadyForNextRoundRequestBody.builder(self._peer_id), self.aggregation_round())
		response = self._handle_request_with_archive(self._prepare_socket_msg_to_send(request))

		if response is None:
			raise Exception("Node _wait_to_start_next_round method. Response is None")
		elif response[am.GenericArchiveResponse.SUBTYPE] != am.ArchiveResponseTypes.GENERIC_SUCCESS:
			raise Exception(f"Node _wait_to_start_next_round method. Archive response is not success. Response: {response}")
		
		while True:
			time.sleep(5)

			if self.is_peer_alive() is False:
				break

			request = am.GenericArchiveRequest.builder(am.ArchiveRequestTypes.START_NEXT_ROUND, am.StartNextRoundRequestBody.builder(self._peer_id), self.aggregation_round())
			response = self._handle_request_with_archive(self._prepare_socket_msg_to_send(request))

			if response is None:
				raise Exception("Node _wait_to_start_next_round method. Response is None")
			elif response[am.GenericArchiveResponse.SUBTYPE] == am.ArchiveResponseTypes.WAIT_FOR_NEXT_ROUND:
				if self._allowed_to_write_redudant_log_messages:
					self._logger.record(msg = f"Node is waiting for the beginning of the next round. Round: {self.aggregation_round()}", logLevel = diagnostic.DEBUG, identifier= self._str_identifier)
			elif response[am.GenericArchiveResponse.SUBTYPE] == am.ArchiveResponseTypes.GENERIC_SUCCESS:
				if self._allowed_to_write_redudant_log_messages:
					self._logger.record(msg = f"Node is starting the next round. Round: {self.aggregation_round()}", logLevel = diagnostic.DEBUG, identifier= self._str_identifier)
				break
			else:
				raise Exception(f"Node _wait_to_start_next_round method. Archive response is unexpected. Response: {response}")

	def _define_and_start_nodes_active_in_the_next_round(self, seed: int = SEED, base_list_of_nodes: (list | None) = None):
		if type(seed) != int or (base_list_of_nodes is not None and type(base_list_of_nodes) != list):
			raise TypeError("Node _define_nodes_active_in_the_next_round method")

		list_of_nodes_active_in_the_next_round = self._define_nodes_active_in_the_next_round(seed, base_list_of_nodes)
		
		if self._allowed_to_write_redudant_log_messages:
			if base_list_of_nodes is not None:
				self._logger.record(msg = f"Nodes for creating updates in the next round selected. List of nodes available included in the latest block: {base_list_of_nodes}. Nodes ids: {list_of_nodes_active_in_the_next_round}. Round: {self.aggregation_round()}", logLevel = diagnostic.DEBUG, identifier= self._str_identifier)
			else:
				self._logger.record(msg = f"Nodes for creating updates in the next round selected. Peers list used. Nodes ids: {list_of_nodes_active_in_the_next_round}. Round: {self.aggregation_round()}", logLevel = diagnostic.DEBUG, identifier= self._str_identifier)

		if self._peer_id in list_of_nodes_active_in_the_next_round:	

			self._logger.record(msg = f"Node is an active trainer in the next round. Round: {self.aggregation_round()}. Starting fitting...", logLevel = diagnostic.INFO, identifier= self._str_identifier)

			# Start training if not part of the commitee
			self._do_training_event.set()

	def _handle_custom_msg(self, msg: dict) -> (dict | None):

		reply = None

		try:
			if type(msg) != dict:
				raise TypeError("Invalid message")

			sender_peer_id = msg[pm.GenericPeerMessage.PEER_ID]

			if msg[nm.GenericNodeMessage.ROUND] < self.aggregation_round():
				if self._allowed_to_write_redudant_log_messages:
					self._logger.record(msg = f"Message received from an old round. It is going to be discarded. Sender peer id: {sender_peer_id}. Message type: {msg[nm.GenericNodeMessage.TYPE]}. Message round: {msg[nm.GenericNodeMessage.ROUND]}. Node round: {self.aggregation_round()}. Message body: {msg[nm.GenericNodeMessage.BODY]}", logLevel = diagnostic.DEBUG, identifier= self._str_identifier)
				
				return None

			# Any message received while an aggregation is in progress is delayed until the end of the aggregation (except for the new model block alert message)
			if msg[nm.GenericNodeMessage.TYPE] != nm.NodeMessageTypes.BROADCAST_NEW_MODEL_BLOCK_ALERT:
				# If an aggregation is in progress, discard messages from current round
				if self._model_aggregation_in_progress.is_set() and msg[nm.GenericNodeMessage.ROUND] == self._aggregation_round_during_aggregation:
					if self._allowed_to_write_redudant_log_messages:
						self._logger.record(msg = f"Message received from current round but aggregation is in progress. It is going to be discarded. Sender peer id: {sender_peer_id}. Message type: {msg[nm.GenericNodeMessage.TYPE]}. Message round: {msg[nm.GenericNodeMessage.ROUND]}. Node round: {self.aggregation_round()}. Message body: {msg[nm.GenericNodeMessage.BODY]}", logLevel = diagnostic.DEBUG, identifier= self._str_identifier)
					
					return None
			
			# If the message is from a peer that is already in the next round, the message is delayed until the next round
			while self.is_peer_alive() and (msg[nm.GenericNodeMessage.ROUND] > self.aggregation_round() or (msg[nm.GenericNodeMessage.ROUND] > self._aggregation_round_during_aggregation and self._model_aggregation_in_progress.is_set())):				
				time.sleep(0.5)	

			if self.is_peer_alive() is False:
				return None

			with self._num_of_messages_being_handled_lock:
				self._num_of_messages_being_handled += 1
			
			try:
				if msg[nm.GenericNodeMessage.TYPE] == nm.NodeMessageTypes.BROADCAST_UPDATE_UPLOAD:
					# A peer has uploaded a new update in the archive

					if self.is_validator():
						try:
							if self._allowed_to_store_weights_directly_in_archive_tmp_dir:
								self._get_authorization_to_read_update_directly_from_file(msg[nm.GenericNodeMessage.BODY][nm.BroadcastUpdateUpload.UPDATE_NAME])	# Archive will confirm the authorization to read the update directly from the file
								updater_id, update_weights, _ = self._read_update_from_disk(msg[nm.GenericNodeMessage.BODY][nm.BroadcastUpdateUpload.UPDATE_NAME])

							else:
								# Validate the update
								updater_id, update_weights, _ = self._download_update_from_archive(msg[nm.GenericNodeMessage.BODY][nm.BroadcastUpdateUpload.UPDATE_NAME])
						except InvalidRoundException:
							return None

						self._validate_update_created_by_node(msg[nm.GenericNodeMessage.BODY][nm.BroadcastUpdateUpload.UPDATE_NAME], updater_id, update_weights)
			
				elif msg[nm.GenericNodeMessage.TYPE] == nm.NodeMessageTypes.BROADCAST_VALIDATION_RESULT:
					# A Validator validated an update by means of the local dataset validation mechanism
					if self._validation_mechanism_type not in cm.WeightsBasedValidationAlgorithmType.list():
						raise NotImplementedError("Validation mechanism is not in the list of the weights based validation mechanisms")

					if self.is_validator():
						body = msg[nm.GenericNodeMessage.BODY]
						name = body[nm.BroadcastValidationResult.UPDATE_NAME]
						score = body[nm.BroadcastValidationResult.SCORE]
						updater_id = body[nm.BroadcastValidationResult.UPDATER_ID]
						result = body[nm.BroadcastValidationResult.RESULT]
						# Save the score and aggregate the scores if the criteria are met
						self._handle_single_validation_result(result, score, name, updater_id, sender_peer_id)
				
				elif msg[nm.GenericNodeMessage.TYPE] == nm.NodeMessageTypes.BROADCAST_MULTIPLE_VALIDATION_RESULTS:
					# A Validator validated an update by means of the Krum validation mechanism
					if self._validation_mechanism_type not in cm.GradientsBasedValidationAlgorithmType.list():
						raise NotImplementedError("Validation mechanism is not in the list of the gradients based validation mechanisms")

					if self.is_validator():
						body = msg[nm.GenericNodeMessage.BODY]
						results = body[nm.BroadcastMultipleValidationResults.RESULTS]
						validator_id = msg[nm.BroadcastMultipleValidationResults.PEER_ID]

						# Save the score and aggregate the scores if the criteria are met
						self._handle_multiple_validation_results(results, validator_id)			
				
				elif msg[nm.GenericNodeMessage.TYPE] == nm.NodeMessageTypes.BROADCAST_NEW_MODEL_BLOCK_ALERT:

					block_hash = msg[nm.GenericNodeMessage.BODY][nm.BroadcastNewModelBlockAlert.BLOCK_HASH]
					self._prepare_node_for_next_round(aggregated_model_block_hash= block_hash)
				
				else:
					raise ValueError("Invalid message type")
			
			except Exception as e:
				raise e
			
			finally:
				with self._num_of_messages_being_handled_lock:
					self._num_of_messages_being_handled -= 1

		except Exception as e:
			self._logger.record(msg = f"Error while handling a custom message", exc = e, logLevel = diagnostic.ERROR, identifier= self._str_identifier)
			raise e

		return reply

	def _download_and_load_info_from_latest_aggregated_model_block(self, model_block_from_json_method, load_model_weights: bool = False) -> bool:
		"""
		Download the last aggregated model block (if any) from the archive and load the informations
		"""
		try:
			if type(load_model_weights) != bool:
				raise TypeError("Node _download_and_load_info_from_latest_aggregated_model_block method")

			recent_aggregated_model_block_found = False
			request = am.GenericArchiveRequest.builder(am.ArchiveRequestTypes.DOWNLOAD_LAST_AGGREGATED_MODEL_BLOCK, am.DownloadLastAggregatedModelRequestBody.builder(self._peer_id), self.aggregation_round())
			response = self._handle_request_with_archive(self._prepare_socket_msg_to_send(request))
			
			if response is None:
				raise Exception("Archive response is None")
			elif response[am.GenericArchiveResponse.SUBTYPE] == am.ArchiveResponseTypes.GENERIC_ERROR:
				raise Exception(f"Archive response is an error message. Response: {response}")
			elif response[am.GenericArchiveResponse.SUBTYPE] != am.ArchiveResponseTypes.DOWNLOAD_LAST_AGGREGATED_MODEL_BLOCK:
				raise ValueError(f"Invalid response type. Response: {response}")
			
			with self._aggregation_round_lock:
				with self._model_lock:
					recent_aggregated_model_block_found = response[am.GenericArchiveResponse.BODY][am.DownloadLastAggregatedModelResponseBody.MODEL_FOUND]
					if recent_aggregated_model_block_found:
						model_block_json = response[am.GenericArchiveResponse.BODY][am.DownloadLastAggregatedModelResponseBody.BLOCK]
						block = model_block_from_json_method(model_block_json)
						model_name = block.get_global_model_name()

						self._aggregation_round = block.get_aggregation_round()
						self._aggregation_round += 1									# To obtain the current round number
						
						if self._allowed_to_write_redudant_log_messages:
							self._logger.record(msg = f"Aggregated model block found and downloaded from the archive. Model name: {model_name}. Round: {self._aggregation_round}", logLevel = diagnostic.DEBUG, identifier= self._str_identifier)

						if load_model_weights:

							if self._allowed_to_store_weights_directly_in_archive_tmp_dir:
								self._get_authorization_to_read_model_weights_directly_from_file(model_name)
								model_info = self._read_model_from_disk(model_name)

							else:
								model_info = self._download_model_from_archive(model_name)
							
							model_weights = model_info[AggregatedModel.WEIGHTS]
							model_optimizer = model_info[AggregatedModel.OPTIMIZER]
							
							del model_info

							self._model_to_fit_when_not_validator = {ng.ModelToFitDictFields.WEIGHTS: model_weights, ng.ModelToFitDictFields.OPTIMIZER: model_optimizer}
							
							del model_weights

							if self._allowed_to_write_redudant_log_messages:
								self._logger.record(msg = f"Latest weights downloaded and loaded from most recent aggregated model block. Model name: {model_name}. Round: {self._aggregation_round}", logLevel = diagnostic.DEBUG, identifier= self._str_identifier)

						else:
							if self._allowed_to_write_redudant_log_messages:
								self._logger.record(msg = f"Weights from most recent aggregated model block ignored. Model name: {model_name}. Round: {self._aggregation_round}", logLevel = diagnostic.DEBUG, identifier= self._str_identifier)
						
					else:
						if self._allowed_to_write_redudant_log_messages:
							self._logger.record(msg = f"No previous model found in the archive. Genesis block's model and weights considered", logLevel = diagnostic.DEBUG, identifier= self._str_identifier)

			return recent_aggregated_model_block_found

		except Exception as e:
			self._logger.record(msg = f"Error while downloading the last aggregated model from the archive", exc = e, logLevel = diagnostic.ERROR, identifier= self._str_identifier)
			raise e
		finally:
			gc.collect()

	def _clear_training_and_validation_variables(self):

		with self._training_lock:
			with self._model_lock:
				if self._is_training:
					if self._model_to_fit_when_not_validator is None:
						raise Exception("Model is None, but training should be in progress")

					self._is_training = False

					if self._allowed_to_write_redudant_log_messages:
						self._logger.record(msg = f"Node is training. Stopping the training...", logLevel = diagnostic.DEBUG, identifier= self._str_identifier)
				
				self._model_to_fit_when_not_validator = None
				
		# If the node is a validator, we need to wait for the validation of the updates before going on
		while True:
			with self._num_of_validation_in_progress_lock:
				if self._num_of_validation_in_progress == 0:
					break

				elif self._num_of_validation_in_progress < 0:
					raise Exception(f"Invalid number of validation in progress. Number is negative. Number of validation in progress: {self._num_of_validation_in_progress}")
				else:
					if self._allowed_to_write_redudant_log_messages:
						self._logger.record(msg = f"Waiting for the validation of the updates to be completed. Number of validation in progress: {self._num_of_validation_in_progress}", logLevel = diagnostic.DEBUG, identifier= self._str_identifier)

			time.sleep(5)
		
		gc.collect()

	def _fit(self):

		with self._training_lock:
			if self._is_training == True:
				raise Exception("Training is already in progress")
			
			self._is_training = True

		with self._model_lock:
			if self._model_to_fit_when_not_validator is None:
				if self._model_aggregation_in_progress.is_set():
					if self._allowed_to_write_redudant_log_messages:
						self._logger.record(msg = f"Model aggregation is in progress. Trainer skips creating the next node update", logLevel = diagnostic.DEBUG, identifier= self._str_identifier)

					return None

				raise Exception("Model is None, but training is requested")
		
		result = None

		if self._validation_mechanism_type in cm.WeightsBasedValidationAlgorithmType.list():
			if TESTING_OPS:
				num_of_samples = 2
				weights = self._model_to_fit_when_not_validator[ng.ModelToFitDictFields.WEIGHTS]
				result = ([arr.tolist() for arr in weights], num_of_samples)

			else:
				result = self._weight_fit()
		
		elif self._validation_mechanism_type in cm.GradientsBasedValidationAlgorithmType.list():
			if TESTING_OPS:
				raise NotImplementedError("Testing operations not implemented")

			result = self._gradient_fit()

		else:
			raise NotImplementedError("Validation mechanism not implemented")
	
		return result
	
	@staticmethod
	def _weight_fit_operations(is_subprocess: bool, lazy_loading: bool, model_architecture: dict, model_info: dict, dataset_quanta_paths: list, fit_epochs: int, batch_size: int, filename: str | None = None):
		result = None
		
		try:
			if type(is_subprocess) != bool or type(lazy_loading) != bool or type(model_architecture) != dict or type(model_info) != dict or type(fit_epochs) != int or type(batch_size) != int or (is_subprocess and (filename is None or type(filename) != str)) or type(dataset_quanta_paths) != list:
				raise TypeError("Node _weight_fit_operations method")
			elif fit_epochs <= 0 or batch_size <= 0:
				raise ValueError("Node _weight_fit_operations method")
			
			if is_subprocess:
				setproctitle.setproctitle(f"{current_process().name}")
			
			training_set = utils.get_training_set(dataset_quanta_paths, lazy_loading, batch_size)

			model = utils.build_model_from_architecture_and_weights(model_architecture, model_info[ng.ModelToFitDictFields.WEIGHTS], optimizer_variables= model_info[ng.ModelToFitDictFields.OPTIMIZER])

			if lazy_loading:
				sample_counter_callback = SampleCounterCallback(batch_size)
				history = model.fit(training_set, epochs=fit_epochs, callbacks=[sample_counter_callback], verbose=VERBOSE)
				num_of_samples = sample_counter_callback.result
			else:
				history = model.fit(training_set['img'], training_set['label'], epochs=fit_epochs, batch_size=batch_size, verbose=VERBOSE)
				num_of_samples = len(training_set['img'])
			
			weights = model.get_weights()
			weights = [arr.tolist() for arr in weights]

			result = {"weights": weights, "history": history.history, "num_of_samples": num_of_samples}

		except Exception as e:
			if is_subprocess:
				result = {"error": f"{type(e)}:{str(e)}"}
			else:
				raise e
		
		if is_subprocess:
			with open(filename, 'w') as f:
				json.dump(result, f)
		else:
			return result

	def _weight_fit(self):
		"""
		Perform model fitting for one or more epochs and return the weights
		"""
		try:
			with self._honest_training_semaphore:
				
				filename = os.path.join(DIRECTORY_WHERE_TO_STORE_TMP_FILES, ''.join(np.random.choice(list(string.ascii_lowercase + string.digits), size=24)) + ".json")

				training_process = Process(target = GenericNode._weight_fit_operations, args = (True, self._lazy_loading, self._model_architecture, self._model_to_fit_when_not_validator, self._dataset_quanta_paths, self._fit_epochs, self._batch_size, filename), name= f"FedBlockSimulator - trainer_{self._peer_id}_honest_weight_train", daemon= True)
				training_process.start()
				training_process.join()

			with open(filename, 'r') as f:
				result = json.load(f)
			
			os.remove(filename)

			if type(result) != dict:
				raise Exception("Result from subprocess is not dict")
			elif "error" in result:
				raise Exception(f"Error while performing honest weights-based training. Error: {result['error']}")
			
			history = result["history"]
			num_of_samples = result["num_of_samples"]
			weights = result["weights"]

			if self._allowed_to_write_redudant_log_messages:
				self._logger.record(msg = f"lazy: {self._lazy_loading} - global round: {self._aggregation_round} - peer {self._peer_id} - History of the training: {history}", logLevel = diagnostic.DEBUG, identifier= self._str_identifier)
			
			if any(np.isnan(history["loss"])):
				self._logger.record(msg = f"Loss has NaN values!", logLevel = diagnostic.ERROR, identifier= self._str_identifier)

			with self._training_lock:
				with self._model_lock:

					if self._is_training == False:
						if self._allowed_to_write_redudant_log_messages:
							self._logger.record(msg = f"Node {self._peer_id} honest weights-based training process discarded because training is not necessary", logLevel = diagnostic.DEBUG, identifier= self._str_identifier)

						return None
					
					self._is_training = False
			
			if self._allowed_to_write_redudant_log_messages:
				self._logger.record(msg = f"Node {self._peer_id} performed an honest weights-based training process. Number of samples: {num_of_samples}", logLevel = diagnostic.DEBUG, identifier= self._str_identifier)

			return (weights, num_of_samples)
		
		except Exception as e:
			self._logger.record(msg = f"Error while performing model fitting to obtain new weights", exc = e, logLevel = diagnostic.ERROR, identifier= self._str_identifier)
			raise e

	@staticmethod
	def _gradient_fit_operations(is_subprocess: bool, lazy_loading: bool, model_architecture: dict, model_info: dict, dataset_quanta_paths: list, current_round: int, batch_size: int, filename: str | None = None):
		result = None

		try:
			if type(is_subprocess) != bool or type(batch_size) != int or type(model_architecture) != dict or type(model_info) != dict or type(dataset_quanta_paths) != list or type(current_round) != int or (is_subprocess and (filename is None or type(filename) != str)) or type(lazy_loading) != bool:
				raise TypeError("Node _gradient_fit_operations method")
			elif batch_size <= 0:
				raise ValueError("Node _gradient_fit_operations method")
			
			if is_subprocess:
				setproctitle.setproctitle(f"{current_process().name}")
			
			training_set = utils.get_training_set(dataset_quanta_paths, lazy_loading, batch_size)

			model = utils.build_model_from_architecture_and_weights(model_architecture, model_info[ng.ModelToFitDictFields.WEIGHTS], optimizer_variables= model_info[ng.ModelToFitDictFields.OPTIMIZER])

			if not lazy_loading:
				batch_idx = current_round % (len(training_set["img"]) // batch_size)

				batch_start = batch_idx * batch_size
				batch_end = batch_start + batch_size
				batch_data = training_set["img"][batch_start:batch_end]
				batch_labels = training_set["label"][batch_start:batch_end]
			else:
				training_set = iter(training_set)

				for _ in range(current_round):
					try:
						batch_data, batch_labels = next(training_set)
					except StopIteration:
						training_set = iter(utils.get_training_set(dataset_quanta_paths, lazy_loading, batch_size))
						batch_data, batch_labels = next(training_set)

			if len(batch_data) != batch_size or len(batch_labels) != batch_size:
				raise Exception(f"Batch data and/or labels have unexpected lengths. Batch data length: {len(batch_data)}, batch labels length: {len(batch_labels)}, batch size: {batch_size}")

			batch_data = convert_to_tensor(batch_data)
			batch_labels = convert_to_tensor(batch_labels)

			# Compute the loss and the gradients
			with GradientTape() as tape:
				predictions = model(batch_data, training=True)
				loss = model.compiled_loss(batch_labels, predictions)
			
			gradients = tape.gradient(loss, model.trainable_variables)

			gradients = [arr.numpy().tolist() for arr in gradients]

			result = {"gradients": gradients, "loss": loss.numpy().tolist()}

		except Exception as e:
			if is_subprocess:
				result = {"error": f"{type(e)}:{str(e)}"}
			else:
				raise e
		
		if is_subprocess:
			with open(filename, 'w') as f:
				json.dump(result, f)
		else:
			return result

	def _gradient_fit(self):
		"""
		Perform one step of model fitting and return the gradients
		"""
		try:
			if self._allowed_to_write_redudant_log_messages:
				self._logger.record(msg = f"Performing gradient fit... Batch size: {self._batch_size}", logLevel = diagnostic.DEBUG, identifier= self._str_identifier)

			with self._honest_training_semaphore:
				filename = os.path.join(DIRECTORY_WHERE_TO_STORE_TMP_FILES, ''.join(np.random.choice(list(string.ascii_lowercase + string.digits), size=24)) + ".json")

				training_process = Process(target = GenericNode._gradient_fit_operations, args = (True, self._lazy_loading, self._model_architecture, self._model_to_fit_when_not_validator, self._dataset_quanta_paths, self.aggregation_round(), self._batch_size, filename), name= f"FedBlockSimulator - trainer_{self._peer_id}_honest_gradient_train", daemon= True)
				training_process.start()
				training_process.join()

			with open(filename, 'r') as f:
				result = json.load(f)
			
			os.remove(filename)

			if type(result) != dict:
				raise Exception("Result from subprocess is not dict")
			elif "error" in result:
				raise Exception(f"Error while performing honest gradients-based training. Error: {result['error']}")
			
			gradients = result["gradients"]
			loss = result["loss"]

			with self._training_lock:
				with self._model_lock:

					if self._is_training == False:
						if self._allowed_to_write_redudant_log_messages:
							self._logger.record(msg = f"Node {self._peer_id} honest gradients-based training process discarded because training is not necessary", logLevel = diagnostic.DEBUG, identifier= self._str_identifier)
						
						return None
				
					self._is_training = False

			if self._allowed_to_write_redudant_log_messages:
				self._logger.record(msg = f"Node {self._peer_id} performed an honest gradients-based training process. Gradient loss computed: {loss}", logLevel= diagnostic.DEBUG, identifier= self._str_identifier)

			return (gradients, None)
		
		except Exception as e:
			self._logger.record(msg = f"Error while performing one step of model fitting to obtain new gradients", exc = e, logLevel = diagnostic.ERROR, identifier= self._str_identifier)
			raise e

	@staticmethod
	def _test_operations(is_subprocess: bool, lazy_loading: bool, test_set_path: str, model_architecture: dict, weights: list, batch_size: int, filename: str | None = None):
		result = None

		try:
			if type(is_subprocess) != bool or type(model_architecture) != dict or type(weights) != list or (is_subprocess and (filename is None or type(filename) != str)) or type(lazy_loading) != bool or type(test_set_path) != str or type(batch_size) != int:
				raise TypeError("Node _test_operations method")
			elif batch_size <= 0:
				raise ValueError("Node _test_operations method")
			
			if is_subprocess:
				setproctitle.setproctitle(f"{current_process().name}")
			
			accuracy = None

			test_set = utils.get_dataset(batch_size, test_set_path, lazy_loading)

			model = utils.build_model_from_architecture_and_weights(model_architecture, weights)
			
			if lazy_loading:
				_, accuracy = model.evaluate(test_set, verbose=VERBOSE)
			
			else:				
				_, accuracy = model.evaluate(test_set["img"], test_set["label"], verbose=VERBOSE)

			result = {"accuracy": accuracy}

		except Exception as e:
			if is_subprocess:
				result = {"error": f"{type(e)}:{str(e)}"}
			else:
				raise e
		
		if is_subprocess:
			with open(filename, 'w') as f:
				json.dump(result, f)
		else:
			return result

	def _test(self, weights: list):

		try:
			if type(weights) != list:
				raise TypeError("Node _test method")

			accuracy = None

			if TESTING_OPS:
				accuracy = np.random.uniform(0.75, 1.0)
			else:	
				
				with self._validation_semaphore:
					filename = os.path.join(DIRECTORY_WHERE_TO_STORE_TMP_FILES, ''.join(np.random.choice(list(string.ascii_lowercase + string.digits), size=24)) + ".json")

					test_process = Process(target = GenericNode._test_operations, args = (True, self._lazy_loading, self._test_set_path, self._model_architecture, weights, self._batch_size, filename), name= f"FedBlockSimulator - {self._peer_id}_test", daemon= True)
					test_process.start()
					test_process.join()

				with open(filename, 'r') as f:
					result = json.load(f)

				os.remove(filename)
				
				if type(result) != dict:
					raise Exception("Result from subprocess is not dict")
				elif "error" in result:
					raise Exception(f"Error while performing test. Error: {result['error']}")
				
				accuracy = result["accuracy"]

			return accuracy

		except Exception as e:
			self._logger.record(msg = f"Error while testing the model", exc = e, logLevel = diagnostic.ERROR, identifier= self._str_identifier)
			raise e

	# IMPORTANT: It MAY be necessary to override this method in the subclasses
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

				current_round = self.aggregation_round()

				weights = None
				num_samples_training_set = None

				result = self._fit()

				if result is not None:
					weights, num_samples_training_set = result

				with self._model_lock:
					self._model_to_fit_when_not_validator = None
				
				gc.collect()

				if weights is not None and self._model_aggregation_in_progress.is_set() is False and self.is_peer_alive() and current_round == self.aggregation_round():
					try:
						name = self._create_weights_filename()

						# Node is allowed to store the weights directly in the archive temporary directory
						if self._allowed_to_store_weights_directly_in_archive_tmp_dir:
							request = am.GenericArchiveRequest.builder(am.ArchiveRequestTypes.AUTHORIZATION_TO_STORE_UPDATE_DIRECTLY, am.AuthorizationToStoreUpdateDirectlyRequestBody.builder(self._peer_id, name, num_samples_training_set), current_round)
						# Node is not allowed to store the weights directly in the archive temporary directory and, so, the Archive will store the weights
						else:
							request = am.GenericArchiveRequest.builder(am.ArchiveRequestTypes.UPLOAD_PEER_UPDATE, am.UploadPeerUpdateRequestBody.builder(self._peer_id, name, weights, num_samples_training_set), current_round)
						
						message_to_send = self._prepare_socket_msg_to_send(request)
						del request
						gc.collect()

						response = self._handle_request_with_archive(message_to_send)
						del message_to_send
						gc.collect()
						
						if response is None:
							raise Exception("Archive response is None")
						elif response[am.GenericArchiveResponse.SUBTYPE] == am.ArchiveResponseTypes.GENERIC_ERROR:
							raise Exception(f"Archive response is an error message. Response: {response}")
						elif response[am.GenericArchiveResponse.SUBTYPE] == am.ArchiveResponseTypes.INVALID_ROUND_NUMBER:
							self._logger.record(msg = f"Invalid round number. Request type: {am.ArchiveRequestTypes.UPLOAD_PEER_UPDATE}. Message round: {response[am.GenericArchiveResponse.ROUND]}. Node round: {current_round}", logLevel = diagnostic.WARNING, identifier= self._str_identifier)
							continue
						elif response[am.GenericArchiveResponse.SUBTYPE] != am.ArchiveResponseTypes.GENERIC_SUCCESS:
							raise ValueError(f"Invalid response type. Response: {response}")

						if self._allowed_to_store_weights_directly_in_archive_tmp_dir:
							self._store_update_on_disk(name, weights, num_samples_training_set)
							self._logger.record(msg = f"Update written directly on disk. Update name: {name}. Round: {current_round}", logLevel = diagnostic.INFO, identifier= self._str_identifier)
						else:
							self._logger.record(msg = f"Update uploaded. Update name: {name}. Round: {current_round}", logLevel = diagnostic.INFO, identifier= self._str_identifier)

						self._send_message(nm.BroadcastUpdateUpload.builder(name, self._peer_id, current_round))
					
					except InvalidRoundException:
						self._logger.record(msg = f"Trainer skips uploading the update to the archive because his current round is lower than the current round of the archive", exc = e, logLevel = diagnostic.DEBUG, identifier= self._str_identifier)

				else:
					if self._allowed_to_write_redudant_log_messages:
						self._logger.record(msg = f"Update neither uploaded to the archive or broadcasted to other peers because model aggregation is in progress or node is not alive", logLevel = diagnostic.DEBUG, identifier= self._str_identifier)
					
				del weights
				gc.collect()

			except Exception as e:
				self._logger.record(msg = f"Error while creating the next node update", exc = e, logLevel = diagnostic.ERROR, identifier= self._str_identifier)

	# IMPORTANT: It must be overridden in the subclasses
	def _define_nodes_active_in_the_next_round(self):
		raise NotImplementedError("Node _define_nodes_active_in_the_next_round method")

	# IMPORTANT: It must be overridden in the subclasses
	def _elect_new_validators(self):
		"""
		Elect the new validators based on the scores of the last aggregation round.
		"""
		raise NotImplementedError("Node _elect_new_validators method")
	
	# IMPORTANT: It must be overridden in the subclasses
	def _prepare_node_for_next_round(self):
		"""
		Prepare the node for the next round
		"""
		raise NotImplementedError("Node _prepare_node_for_next_round method")

	# IMPORTANT: It must be overridden in the subclasses
	def _create_and_upload_aggregated_model_block(self):
		raise NotImplementedError("Node _create_and_upload_aggregated_model_block method")
	

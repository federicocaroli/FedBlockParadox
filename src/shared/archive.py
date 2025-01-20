import os, os.path, shutil, time, json, gc, copy
from typing import Optional
from threading import Lock, Event

from . import diagnostic
from .socket_node import SocketNode
from .enums.archive_generic import FileWrittenByNode, PeerUpdate, AggregatedModel, FileStoredType, FileStored, InvalidRoundNumber
from .enums import archive_messages as am
from .block import BlockType

class AbstractArchive(SocketNode):
	def __init__(self, host: str, port: int, persist: bool = False, used_path: str = "./tmp_archive", logger_path: str = "./archive_logger.log", logger_level: int = diagnostic.INFO):

		if type(host) != str or type(port) != int or type(persist) != bool or type(used_path) != str or type(logger_path) != str or type(logger_level) != int:
			raise TypeError("Archive __init__ method")

		super().__init__("Archive", host, port)

		self._logger = diagnostic.Diagnostic(logger_path, logger_level, logger_name="Archive")

		if not os.path.exists(used_path):
			os.makedirs(used_path)

		self._used_path = used_path
		self._persist = persist
		self._blockchain = []
		self._block_index_list = {}
		self._blockchain_lock = Lock()
		self._files_stored = {}
		self._files_lock = Lock()
		self._threads_accessing_files = 0								# Number of threads accessing the file system (reading / writing files). Used to prevent deletion of files while they are being accessed
		self._nodes_writing_files = {}									# Nodes that are currently writing files to the archive
		self._threads_accessing_files_lock = Lock()			# Lock to mantain the list of available files thread safe
		self._active_connections = 0
		self._active_connections_lock = Lock()
		self._stop_archive = None
		self._delete_old_files = None									# Event to signal that the files stored in the archive should be deleted (used only when persist is False to delete the files related to previous rounds)
		self._current_round_number = 1
		self._current_round_number_lock = Lock()
		self._available_nodes_at_the_end_of_the_round = []
		self._available_nodes_at_the_end_of_the_round_lock = Lock()
		self._ready_message_received_from_available_nodes = []

	def current_round_number(self) -> int:
		with self._current_round_number_lock:
			return self._current_round_number

	def stop(self, remove_tmp_dir: bool = True):

		if type(remove_tmp_dir) != bool:
			raise TypeError("Archive stop method")
		elif self._delete_old_files is None:
			raise Exception("Event variable _delete_old_files is None")
		elif self._stop_archive is None:
			raise Exception("Event variable _stop_archive is None")
		elif self._stop_archive.is_set():
			self._logger.record(f"Archive already stopped", logLevel = diagnostic.WARNING, identifier = self._node_name)
			return

		self._stop_archive.set()

		super().stop()

		# Wait for all active connections to finish
		while True:
			with self._active_connections_lock:
				if self._active_connections == 0:
					break
			
			time.sleep(0.5)

		# Clear the blockchain and files stored
		with self._blockchain_lock:
			self._blockchain = []
			self._block_index_list = {}

		if os.path.exists(self._used_path) and remove_tmp_dir:

			self._logger.record(f"Removing temporary directory: {self._used_path} ...", logLevel = diagnostic.INFO, identifier = self._node_name)

			if self._delete_old_files is None:
				raise Exception("Event variable _delete_old_files is None")
			
			while self._delete_old_files.is_set():
				pass

			self._delete_old_files.set()

			self._wait_until_everyone_stops_writing_and_reading_files()

			shutil.rmtree(self._used_path, ignore_errors=True)

			# Clear the files stored
			with self._files_lock:
				self._files_stored = {}
		
			self._logger.record(f"Temporary directory removed", logLevel = diagnostic.INFO, identifier = self._node_name)

		self._stop_archive = None
		self._delete_old_files = None

		self._logger.record(f"Archive stopped", logLevel = diagnostic.INFO, identifier = self._node_name)

	def start(self, genesis_block):

		if self._stop_archive is not None:
			raise Exception("Archive already started")

		self._active_connections = 0
		self._stop_archive = Event()

		self._threads_accessing_files = 0
		self._delete_old_files = Event()

		with self._blockchain_lock:
			self._blockchain = [genesis_block]
			self._block_index_list = {genesis_block.get_block_hash(): self._blockchain[0]}

		super().start()

		self._logger.record(f"Archive started", logLevel = diagnostic.INFO, identifier = self._node_name)

	def _wait_until_everyone_stops_writing_and_reading_files(self):
		try:
			while True:
				with self._threads_accessing_files_lock:

					if len(self._nodes_writing_files) > 0:
						node_ids_to_remove = []

						for node_id in self._nodes_writing_files:

							node_can_be_removed = True
							for file_elem in self._nodes_writing_files[node_id]:
								# If the file is already written by the node, remove the node from the list
								if os.path.exists(os.path.join(self._used_path, file_elem[FileWrittenByNode.FILE_NAME])):
									pass
								
								# If the node is writing the file for more than 30 seconds, remove the node from the list
								elif time.time() - file_elem[FileWrittenByNode.TIMESTAMP_REQUEST] > 30:
									self._logger.record(f"Node has been writing the file for too much time! Node id: {node_id}. File name: {file_elem[FileWrittenByNode.FILE_NAME]}. Time delta: {time.time() - file_elem[FileWrittenByNode.TIMESTAMP_REQUEST]}", logLevel = diagnostic.ERROR, identifier = self._node_name)
								
								else:
									self._logger.record(f"Node is still writing the file. Node id: {node_id}. File name: {file_elem[FileWrittenByNode.FILE_NAME]}", logLevel = diagnostic.DEBUG, identifier = self._node_name)
									node_can_be_removed = False
									break
							
							if node_can_be_removed:
								node_ids_to_remove.append(node_id)
							
						for node_id in node_ids_to_remove:
							self._nodes_writing_files.pop(node_id)

					if self._threads_accessing_files == 0 and len(self._nodes_writing_files) == 0:
						break
				
				time.sleep(0.5)

		except Exception as e:
			self._logger.record(f"Error while waiting for everyone to stop writing and reading files", logLevel = diagnostic.ERROR, identifier = self._node_name, exc= e)
			raise e

	def store_blockchain_in_file(self, file_name: str):

		try:
			if type(file_name) != str:
				raise TypeError("store_blockchain_in_file method")
		
			blockchain_list = []

			with self._blockchain_lock:
				for block in self._blockchain:
					blockchain_list.append(
						{
							"block_type": block.get_block_type(),
							"block_content": block.to_json()
						}
					)
			
			with open(file_name, "w") as file:
				json.dump(blockchain_list, file, indent=4)

		except Exception as e:
			self._logger.record(f"Error while storing blockchain in file", logLevel = diagnostic.ERROR, identifier = self._node_name, exc= e)

	def _handle_msg(self, msg: dict) -> Optional[str]:
		raise NotImplementedError

class GenericArchive(AbstractArchive):
	def __init__(self, host: str, port: int, persist: bool = False, used_path: str = "./tmp_archive", logger_path: str = "./archive_logger.log", logger_level: int = diagnostic.INFO):
		super().__init__(host, port, persist, used_path, logger_path, logger_level)

	def start(self, genesis_block):
		
		super().start(genesis_block)

	def _handle_msg(self, msg: dict) -> Optional[str]:				
		
		if type(msg) != dict:
			raise TypeError("Archive _handle_msg method")
		elif self._stop_archive is None or self._stop_archive.is_set():
			self._logger.record(f"Requested received while archive is stopped", logLevel = diagnostic.ERROR, identifier = self._node_name)
			return am.GenericArchiveRequest.builder(am.ArchiveResponseTypes.GENERIC_ERROR, body=am.ErrorResponseBody.builder("Archive is stopped"), round=self.current_round_number())

		reply = None

		with self._active_connections_lock:
			self._active_connections += 1

		try:
			if any(key not in msg.keys() for key in am.GenericArchiveMessage.list()):
				raise Exception(f"Invalid syntax in GenericArchiveMessage. Message keys: {msg.keys()}")
			elif msg[am.GenericArchiveMessage.TYPE] != am.ArchiveMessageTypes.REQUEST:
				raise Exception(f"Received message is not a request message. Message type: {msg[am.GenericArchiveMessage.TYPE]}")
			elif any(key not in msg.keys() for key in am.GenericArchiveRequest.list()):
				raise Exception(f"Invalid syntax in GenericArchiveRequest. Message keys: {msg.keys()}")
			elif msg[am.GenericArchiveRequest.ROUND] != self.current_round_number():

				# Nodes can download the last aggregated model block to resume the training process (in case of a crash or disconnection)
				if msg[am.GenericArchiveRequest.SUBTYPE] not in [am.ArchiveRequestTypes.DOWNLOAD_LAST_AGGREGATED_MODEL_BLOCK, am.ArchiveRequestTypes.DOWNLOAD_GENESIS]:
					raise InvalidRoundNumber()

			msg_body = msg[am.GenericArchiveRequest.BODY]

			if msg[am.GenericArchiveRequest.SUBTYPE] == am.ArchiveRequestTypes.READY_FOR_NEXT_ROUND:
				with self._available_nodes_at_the_end_of_the_round_lock:
					if self._available_nodes_at_the_end_of_the_round is None:
						raise Exception("Available nodes at the end of the round is None")

					if msg_body[am.GenericArchiveRequestBody.PEER_ID] in self._ready_message_received_from_available_nodes:
						raise Exception(f"Node with id: {msg_body[am.GenericArchiveRequestBody.PEER_ID]} already sent the ready message")
					
					self._ready_message_received_from_available_nodes.append(msg_body[am.GenericArchiveRequestBody.PEER_ID])

					self._logger.record(f"Ready nodes: {self._ready_message_received_from_available_nodes}. Available nodes: {self._available_nodes_at_the_end_of_the_round}. Round: {self.current_round_number()}", logLevel = diagnostic.DEBUG, identifier = self._node_name)

					# These operations must be done only 1 time per round
					if all(peer_id in self._ready_message_received_from_available_nodes for peer_id in self._available_nodes_at_the_end_of_the_round):
						self._delete_files_from_previous_round()			
			
				reply = am.GenericArchiveResponse.builder(am.ArchiveResponseTypes.GENERIC_SUCCESS, body=am.SuccessResponseBody.builder(), round=self.current_round_number())
			
			elif msg[am.GenericArchiveRequest.SUBTYPE] == am.ArchiveRequestTypes.START_NEXT_ROUND:
				with self._available_nodes_at_the_end_of_the_round_lock:
					if self._available_nodes_at_the_end_of_the_round is None:
						raise Exception("Available nodes at the end of the round is None")
					
					if any(node_id not in self._ready_message_received_from_available_nodes for node_id in self._available_nodes_at_the_end_of_the_round):
						reply = am.GenericArchiveResponse.builder(am.ArchiveResponseTypes.WAIT_FOR_NEXT_ROUND, body=am.WaitResponseBody.builder(), round=self.current_round_number())
					else:
						reply = am.GenericArchiveResponse.builder(am.ArchiveResponseTypes.GENERIC_SUCCESS, body=am.SuccessResponseBody.builder(), round=self.current_round_number())

			elif msg[am.GenericArchiveRequest.SUBTYPE] == am.ArchiveRequestTypes.UPLOAD_PEER_UPDATE:
				self._handle_upload_peer_update_request(msg_body)

			elif msg[am.GenericArchiveRequest.SUBTYPE] == am.ArchiveRequestTypes.AUTHORIZATION_TO_STORE_UPDATE_DIRECTLY:
				reply = self._handle_upload_peer_update_request(msg_body)

			elif msg[am.GenericArchiveRequest.SUBTYPE] == am.ArchiveRequestTypes.DOWNLOAD_PEER_UPDATE:
				reply = self._handle_download_peer_update_request(msg_body, True)
			
			elif msg[am.GenericArchiveRequest.SUBTYPE] == am.ArchiveRequestTypes.AUTHORIZATION_TO_READ_UPDATE_DIRECTLY:
				reply = self._handle_download_peer_update_request(msg_body, False)
			
			elif msg[am.GenericArchiveRequest.SUBTYPE] == am.ArchiveRequestTypes.UPLOAD_AGGREGATED_MODEL:
				self._handle_upload_aggregated_model_request(msg_body)

			elif msg[am.GenericArchiveRequest.SUBTYPE] == am.ArchiveRequestTypes.AUTHORIZATION_TO_STORE_AGGREGATED_MODEL_DIRECTLY:
				self._handle_upload_aggregated_model_request(msg_body)

			elif msg[am.GenericArchiveRequest.SUBTYPE] == am.ArchiveRequestTypes.DOWNLOAD_AGGREGATED_MODEL:
				reply = self._handle_download_aggregated_model_request(msg_body, True)

			elif msg[am.GenericArchiveRequest.SUBTYPE] == am.ArchiveRequestTypes.AUTHORIZATION_TO_READ_AGGREGATED_MODEL_DIRECTLY:
				reply = self._handle_download_aggregated_model_request(msg_body, False)
			
			elif msg[am.GenericArchiveRequest.SUBTYPE] == am.ArchiveRequestTypes.DOWNLOAD_LAST_AGGREGATED_MODEL_BLOCK:
				reply = self._handle_download_last_aggregated_model_request(msg_body)

			elif msg[am.GenericArchiveRequest.SUBTYPE] == am.ArchiveRequestTypes.UPLOAD_BLOCKCHAIN_AGGREGATED_MODEL:
				reply = self._handle_upload_blockchain_aggregated_model_request(msg_body)

			elif msg[am.GenericArchiveRequest.SUBTYPE] == am.ArchiveRequestTypes.DOWNLOAD_LAST_BLOCKCHAIN_BLOCK:
				reply = self._handle_download_last_blockchain_block_request(msg_body)

			elif msg[am.GenericArchiveRequest.SUBTYPE] == am.ArchiveRequestTypes.DOWNLOAD_BLOCKCHAIN_BLOCK:
				reply = self._handle_download_blockchain_block_request(msg_body)

			elif msg[am.GenericArchiveRequest.SUBTYPE] == am.ArchiveRequestTypes.DOWNLOAD_GENESIS:
				reply = self._handle_download_genesis_request(msg_body)

			else:
				raise Exception(f"Invalid request type. Request type: {msg[am.GenericArchiveRequest.TYPE]}")

			# Double checking if the round is correct
			if msg[am.GenericArchiveRequest.ROUND] != self.current_round_number():

				# Nodes can download the last aggregated model block to resume the training process (in case of a crash or disconnection), The uploader of the new aggregated model block has the wrong round number at this point of the code for sure
				if msg[am.GenericArchiveRequest.SUBTYPE] not in [am.ArchiveRequestTypes.DOWNLOAD_LAST_AGGREGATED_MODEL_BLOCK, am.ArchiveRequestTypes.DOWNLOAD_GENESIS, am.ArchiveRequestTypes.UPLOAD_BLOCKCHAIN_AGGREGATED_MODEL]:
					raise InvalidRoundNumber()

		except InvalidRoundNumber as e:
			reply = am.GenericArchiveResponse.builder(am.ArchiveResponseTypes.INVALID_ROUND_NUMBER, body=am.InvalidRoundNumberResponseBody.builder(), round=self.current_round_number())
			self._logger.record(f"Invalid round number. Message round number: {msg[am.GenericArchiveRequest.ROUND]}. Current round number: {self.current_round_number()}. Msg: {msg}", logLevel = diagnostic.WARNING, identifier = self._node_name)
		except Exception as e:
			reply = am.GenericArchiveResponse.builder(am.ArchiveResponseTypes.GENERIC_ERROR, body=am.ErrorResponseBody.builder(e.args), round=self.current_round_number())
			self._logger.record(f"Exception while handling socket message.  Msg: {msg}", exc = e, logLevel = diagnostic.ERROR, identifier = self._node_name)
		finally:
			with self._active_connections_lock:
				self._active_connections -= 1
			
		if reply is None:
			reply = am.GenericArchiveResponse.builder(am.ArchiveResponseTypes.GENERIC_SUCCESS, body=am.SuccessResponseBody.builder(), round=self.current_round_number())

		gc.collect()

		return reply

	def _delete_files_from_previous_round(self):
		try:
			if self._delete_old_files is None:
				raise Exception("Event variable _delete_old_files is None")
			elif self._delete_old_files.is_set():
				self._logger.record(f"Event variable _delete_old_files is already set", logLevel = diagnostic.WARNING, identifier = self._node_name)
				return

			self._delete_old_files.set()

			self._wait_until_everyone_stops_writing_and_reading_files()
			
			with self._blockchain_lock:
				model_name = self._blockchain[-1].get_global_model_name()

			with self._files_lock:
				if model_name not in self._files_stored:
					raise Exception(f"Model with name: '{model_name}' does not exist")
				
				for file_name in copy.deepcopy(list(self._files_stored.keys())):
					# If persist is False, delete the old files related to the previous rounds
					if self._files_stored[file_name][FileStored.ROUND] < self.current_round_number() and file_name != model_name and self._persist is False:
						del self._files_stored[file_name]
						os.remove(os.path.join(self._used_path, file_name))
					
					# Used to store a copy of global models in a separate directory
					if file_name == model_name:
						if not os.path.exists(os.path.join(self._used_path, "global_models")):
							os.makedirs(os.path.join(self._used_path, "global_models"))

						global_model_copy = f"{self.current_round_number() - 1}_{model_name}"
						if os.path.exists(os.path.join(os.path.join(self._used_path, "global_models"), global_model_copy)):
							raise Exception(f"Global model with name: '{global_model_copy}' already exists")

						shutil.copyfile(os.path.join(self._used_path, model_name), os.path.join(os.path.join(self._used_path, "global_models"), global_model_copy))

				if self._persist is False:
					self._logger.record(f"Old files related to previous rounds deleted", logLevel = diagnostic.DEBUG, identifier = self._node_name)

			self._delete_old_files.clear()
		
		except Exception as e:
			self._logger.record(f"Error while deleting files from previous rounds", logLevel = diagnostic.ERROR, identifier = self._node_name, exc= e)
			raise e

	def _handle_upload_peer_update_request(self, msg: dict):

		if type(msg) != dict:
			raise TypeError("Archive _handle_upload_peer_update_request method")
		elif any(key not in msg.keys() for key in am.AuthorizationToStoreUpdateDirectlyRequestBody.list()):
			raise Exception(f"Invalid syntax in AuthorizationToStoreUpdateDirectlyRequestBody. Message keys: {msg.keys()}")
		
		update_name = msg[am.AuthorizationToStoreUpdateDirectlyRequestBody.UPDATE_NAME]
		num_samples = msg[am.AuthorizationToStoreUpdateDirectlyRequestBody.NUM_SAMPLES]
		peer_id = msg[am.AuthorizationToStoreUpdateDirectlyRequestBody.PEER_ID]

		while self._delete_old_files is not None and self._delete_old_files.is_set():
			time.sleep(0.5)

		with self._threads_accessing_files_lock:
			self._threads_accessing_files += 1

		try:
			with self._files_lock:
				if update_name in self._files_stored:
					raise Exception(f"Update with name: '{update_name}' already exists")
				
				self._files_stored[update_name] = {FileStored.TYPE: FileStoredType.UPDATE, FileStored.ROUND: self.current_round_number()}

			# Optionally the update weights can be stored by the archive
			if am.UploadPeerUpdateRequestBody.UPDATE_WEIGHTS in msg:
				
				if any(key not in msg.keys() for key in am.UploadPeerUpdateRequestBody.list()):
					raise Exception(f"Invalid syntax in UploadPeerUpdateRequestBody. Message keys: {msg.keys()}")
				
				with open(os.path.join(self._used_path, update_name), "w") as f:
					json.dump({PeerUpdate.WEIGHTS: msg[am.UploadPeerUpdateRequestBody.UPDATE_WEIGHTS], PeerUpdate.NUM_SAMPLES: num_samples, PeerUpdate.PEER_ID: peer_id, PeerUpdate.NAME: update_name}, f)

			# If the update weights are not stored, we keep track of the node that is currently writing the file
			else:
				with self._threads_accessing_files_lock:
					if peer_id in self._nodes_writing_files:
						if any(file_elem[FileWrittenByNode.FILE_NAME] == update_name for file_elem in self._nodes_writing_files[peer_id]):
							raise Exception(f"Node with id: '{peer_id}' is already writing that update. File name: {update_name}")
					
					else:
						self._nodes_writing_files[peer_id] = []

					# We keep track of the nodes that are currently writing files to the archive
					self._nodes_writing_files[peer_id].append({FileWrittenByNode.FILE_NAME: update_name, FileWrittenByNode.TIMESTAMP_REQUEST: time.time()})

		except Exception as e:
			self._logger.record(f"Exception while uploading peer update. Peer id: {peer_id}", exc = e, logLevel = diagnostic.ERROR, identifier = self._node_name)
			raise e
		else:
			self._logger.record(f"Peer update uploaded. Peer id: {peer_id}. Update name: {update_name}. Num samples: {num_samples}", logLevel = diagnostic.DEBUG, identifier = self._node_name)
		finally:
			with self._threads_accessing_files_lock:
				self._threads_accessing_files -= 1
	
	def _handle_download_peer_update_request(self, msg: dict, return_weights: bool) -> dict:

		if type(msg) != dict or type(return_weights) != bool:
			raise TypeError("Archive _handle_download_peer_update_request method")
		elif any(key not in msg.keys() for key in am.AuthorizationToReadUpdateDirectlyRequestBody.list()):
			raise Exception(f"Invalid syntax in AuthorizationToReadUpdateDirectlyRequestBody. Message keys: {msg.keys()}")
		
		elif any(key not in msg.keys() for key in am.DownloadPeerUpdateRequestBody.list()):
			raise Exception(f"Invalid syntax in DownloadPeerUpdateRequestBody. Message keys: {msg.keys()}")

		response = None
		update = None
		update_name = msg[am.AuthorizationToReadUpdateDirectlyRequestBody.UPDATE_NAME]
		peer_id = msg[am.AuthorizationToReadUpdateDirectlyRequestBody.PEER_ID]

		while self._delete_old_files is not None and self._delete_old_files.is_set():
			time.sleep(0.5)

		with self._threads_accessing_files_lock:
			self._threads_accessing_files += 1

		try:
			with self._files_lock:
				# TODO it could be improved by checking if the file has been completely written by the node when the nodes can write the file directly
				if update_name not in self._files_stored:
					raise Exception(f"Update with name: '{update_name}' does not exist")

			if return_weights:
				with open(os.path.join(self._used_path, update_name), "r") as f:
					update = json.load(f)

				response = am.GenericArchiveResponse.builder(am.ArchiveResponseTypes.DOWNLOAD_PEER_UPDATE, body=am.DownloadPeerUpdateResponseBody.builder(update[PeerUpdate.NAME], update[PeerUpdate.WEIGHTS], update[PeerUpdate.NUM_SAMPLES], update[PeerUpdate.PEER_ID]), round=self.current_round_number())
		
		except Exception as e:
			self._logger.record(f"Exception while downloading peer update. Peer id: {peer_id}", exc = e, logLevel = diagnostic.ERROR, identifier = self._node_name)
			raise e
		else:
			self._logger.record(f"Peer update downloaded. Peer id: {peer_id}. Update name: {update_name}", logLevel = diagnostic.DEBUG, identifier = self._node_name)
		finally:
			with self._threads_accessing_files_lock:
				self._threads_accessing_files -= 1

		return response

	def _handle_upload_aggregated_model_request(self, msg: dict) -> None:
		
		if type(msg) != dict:
			raise TypeError("Archive _handle_upload_aggregated_model_request method")
		elif any(key not in msg.keys() for key in am.AuthorizationToStoreAggregatedModelDirectlyRequestBody.list()):
			raise Exception(f"Invalid syntax in AuthorizationToStoreAggregatedModelDirectlyRequestBody. Message keys: {msg.keys()}")
			
		model_name = msg[am.AuthorizationToStoreAggregatedModelDirectlyRequestBody.MODEL_NAME]
		peer_id = msg[am.AuthorizationToStoreAggregatedModelDirectlyRequestBody.PEER_ID]

		while self._delete_old_files is not None and self._delete_old_files.is_set():
			time.sleep(0.5)
		
		with self._threads_accessing_files_lock:
			self._threads_accessing_files += 1

		try:
			with self._files_lock:
				if model_name in self._files_stored:
					raise Exception(f"Model with name: '{model_name}' already exists")
				
				self._files_stored[model_name] = {FileStored.TYPE: FileStoredType.MODEL, FileStored.ROUND: self.current_round_number()}

			# Optionally the model weights can be stored by the archive
			if am.UploadAggregatedModelRequestBody.MODEL_WEIGHTS in msg:

				if any(key not in msg.keys() for key in am.UploadAggregatedModelRequestBody.list()):
					raise Exception(f"Invalid syntax in UploadAggregatedModelRequestBody. Message keys: {msg.keys()}")

				with open(os.path.join(self._used_path, model_name), "w") as f:
					json.dump({AggregatedModel.WEIGHTS: msg[am.UploadAggregatedModelRequestBody.MODEL_WEIGHTS], AggregatedModel.NAME: model_name, AggregatedModel.OPTIMIZER: msg[am.UploadAggregatedModelRequestBody.MODEL_OPTIMIZER]}, f)

			# If the model weights are not stored, we keep track of the node that is currently writing the file
			else:
				with self._threads_accessing_files_lock:
					if peer_id in self._nodes_writing_files:
						if any(file_elem[FileWrittenByNode.FILE_NAME] == model_name for file_elem in self._nodes_writing_files[peer_id]):
							raise Exception(f"Node with id: '{peer_id}' is already writing that model. File name: {model_name}")
					
					else:
						self._nodes_writing_files[peer_id] = []

					# We keep track of the nodes that are currently writing files to the archive
					self._nodes_writing_files[peer_id].append({FileWrittenByNode.FILE_NAME: model_name, FileWrittenByNode.TIMESTAMP_REQUEST: time.time()})

		except Exception as e:
			self._logger.record(f"Exception while uploading aggregated model. Peer id: {peer_id}", exc = e, logLevel = diagnostic.ERROR, identifier = self._node_name)
			raise e
		else:
			self._logger.record(f"Aggregated model uploaded. Peer id: {peer_id}. Model name: {model_name}", logLevel = diagnostic.DEBUG, identifier = self._node_name)
		finally:
			with self._threads_accessing_files_lock:
				self._threads_accessing_files -= 1
	
	def _handle_download_aggregated_model_request(self, msg: dict, return_weights: bool):

		if type(msg) != dict or type(return_weights) != bool:
			raise TypeError("Archive _handle_download_aggregated_model_request method")
		elif any(key not in msg.keys() for key in am.DownloadAggregatedModelRequestBody.list()):
			raise Exception(f"Invalid syntax in DownloadAggregatedModelRequestBody. Message keys: {msg.keys()}")

		response = None
		model = None
		model_name = msg[am.DownloadAggregatedModelRequestBody.MODEL_NAME]
		peer_id = msg[am.DownloadAggregatedModelRequestBody.PEER_ID]

		while self._delete_old_files is not None and self._delete_old_files.is_set():
			time.sleep(0.5)
		
		with self._threads_accessing_files_lock:
			self._threads_accessing_files += 1

		try:
			with self._files_lock:
				if model_name not in self._files_stored:
					raise Exception(f"Model with name: '{model_name}' does not exist")

			if return_weights:
				with open(os.path.join(self._used_path, model_name), "r") as f:
					model = json.load(f)
				
				response = am.GenericArchiveResponse.builder(am.ArchiveResponseTypes.DOWNLOAD_AGGREGATED_MODEL, body=am.DownloadAggregatedModelResponseBody.builder(model[AggregatedModel.NAME], model[AggregatedModel.WEIGHTS], model[AggregatedModel.OPTIMIZER]), round=self.current_round_number())
		
		except Exception as e:
			self._logger.record(f"Exception while downloading aggregated model. Peer id: {peer_id}", exc = e, logLevel = diagnostic.ERROR, identifier = self._node_name)
			raise e
		else:
			self._logger.record(f"Aggregated model downloaded. Peer id: {peer_id}. Model name: {model_name}", logLevel = diagnostic.DEBUG, identifier = self._node_name)
		finally:
			with self._threads_accessing_files_lock:
				self._threads_accessing_files -= 1

		return response

	def _handle_download_last_aggregated_model_request(self, msg: dict):
		
		if type(msg) != dict:
			raise TypeError("Archive _handle_download_last_aggregated_model_request method")
		elif any(key not in msg.keys() for key in am.DownloadLastAggregatedModelRequestBody.list()):
			raise Exception(f"Invalid syntax in DownloadLastAggregatedModelRequestBody. Message keys: {msg.keys()}")

		response = None
		model_name = None
		model_found = False
		model_block = None
		peer_id = msg[am.DownloadLastAggregatedModelRequestBody.PEER_ID]

		try:
			# Cycle back through the blockchain to find the last model block
			with self._blockchain_lock:
				for block in reversed(self._blockchain):
					if block.get_block_type() == BlockType.MODEL:
						model_found = True
						model_block = block.to_json()
						model_name = block.get_global_model_name()
						break

			response =  am.GenericArchiveResponse.builder(am.ArchiveResponseTypes.DOWNLOAD_LAST_AGGREGATED_MODEL_BLOCK, body=am.DownloadLastAggregatedModelResponseBody.builder(model_block, model_found), round=self.current_round_number())
		except Exception as e:
			self._logger.record(f"Exception while downloading last aggregated model. Peer id: {peer_id}", exc = e, logLevel = diagnostic.ERROR, identifier = self._node_name)
			raise e
		else:
			self._logger.record(f"Last aggregated model downloaded. Peer id: {peer_id}. Model found: {model_found}. Model name: {model_name}", logLevel = diagnostic.DEBUG, identifier = self._node_name)

		return response

	def _handle_download_last_blockchain_block_request(self, msg: dict):

		if type(msg) != dict:
			raise TypeError("Archive _handle_download_last_blockchain_block_request method")
		elif any(key not in msg.keys() for key in am.DownloadLastBlockchainBlockRequestBody.list()):
			raise Exception(f"Invalid syntax in DownloadLastBlockchainBlockRequestBody. Message keys: {msg.keys()}")

		peer_id = msg[am.DownloadLastBlockchainBlockRequestBody.PEER_ID]
		response = None
		block_hash = None
		block_type = None

		try:
			with self._blockchain_lock:
				if len(self._blockchain) == 0:
					raise Exception("Blockchain is empty")
			
				block = self._blockchain[-1]
			
			block_type = block.get_block_type()
			block_hash = block.get_block_hash()
			response = am.GenericArchiveResponse.builder(am.ArchiveResponseTypes.DOWNLOAD_BLOCKCHAIN_BLOCK, body=am.DownloadBlockchainBlockResponseBody.builder(block.to_json(), block_type), round=self.current_round_number())
		except Exception as e:
			self._logger.record(f"Exception while downloading last blockchain block. Peer id: {peer_id}", exc = e, logLevel = diagnostic.ERROR, identifier = self._node_name)
			raise e
		else:
			self._logger.record(f"Last blockchain block downloaded. Peer id: {peer_id}. Block type: {block_type}. Block hash: {block_hash}", logLevel = diagnostic.DEBUG, identifier = self._node_name)

		return response

	def _handle_download_blockchain_block_request(self, msg: dict):
		
		if type(msg) != dict:
			raise TypeError("Archive _handle_download_blockchain_block_request method")
		elif any(key not in msg.keys() for key in am.DownloadBlockchainBlockRequestBody.list()):
			raise Exception(f"Invalid syntax in DownloadBlockchainBlockRequestBody. Message keys: {msg.keys()}")

		block_hash = msg[am.DownloadBlockchainBlockRequestBody.BLOCK_HASH]
		peer_id = msg[am.DownloadBlockchainBlockRequestBody.PEER_ID]
		response = None
		block_type = None

		try:
			with self._blockchain_lock:
				if block_hash not in self._block_index_list:
					raise Exception(f"Block with hash: {block_hash} not found")
		
				block = self._block_index_list[block_hash]
			
			block_type = block.get_block_type()
			response = am.GenericArchiveResponse.builder(am.ArchiveResponseTypes.DOWNLOAD_BLOCKCHAIN_BLOCK, body=am.DownloadBlockchainBlockResponseBody.builder(block.to_json(), block_type), round=self.current_round_number())
		except Exception as e:
			self._logger.record(f"Exception while downloading blockchain block. Peer id: {peer_id}. Block hash: {block_hash}", exc = e, logLevel = diagnostic.ERROR, identifier = self._node_name)
			raise e
		else:
			self._logger.record(f"Blockchain block downloaded. Peer id: {peer_id}. Block type: {block_type}. Block hash: {block_hash}", logLevel = diagnostic.DEBUG, identifier = self._node_name)

		return response

	def _handle_download_genesis_request(self, msg: dict) -> dict:
		
		if type(msg) != dict:
			raise TypeError("Archive _handle_download_genesis_request method")
		elif any(key not in msg.keys() for key in am.DownloadGenesisRequestBody.list()):
			raise Exception(f"Invalid syntax in DownloadGenesisRequestBody. Message keys: {msg.keys()}")
		
		peer_id = msg[am.DownloadGenesisRequestBody.PEER_ID]
		response = None

		try:
			with self._blockchain_lock:
				if len(self._blockchain) == 0:
					raise Exception("Blockchain is empty")
			
				genesis_block = self._blockchain[0]
			
			if genesis_block.get_block_type() != BlockType.GENESIS:
				raise Exception("Genesis block not found")

			response = am.GenericArchiveResponse.builder(am.ArchiveResponseTypes.DOWNLOAD_BLOCKCHAIN_BLOCK, body=am.DownloadBlockchainBlockResponseBody.builder(genesis_block.to_json(), genesis_block.get_block_type()), round=self.current_round_number())
		except Exception as e:
			self._logger.record(f"Exception while downloading genesis block. Peer id: {peer_id}", exc = e, logLevel = diagnostic.ERROR, identifier = self._node_name)
			raise e
		else:
			self._logger.record(f"Genesis block downloaded. Peer id: {peer_id}", logLevel = diagnostic.DEBUG, identifier = self._node_name)

		return response

	# IMPORTANT: It must be overriden in the subclass
	def _handle_upload_blockchain_aggregated_model_request(self, msg: dict):
		raise NotImplementedError
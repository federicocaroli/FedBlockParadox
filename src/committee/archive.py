import os, copy

from ..shared import diagnostic
from ..shared.archive import GenericArchive
from ..shared.enums import archive_messages as shared_am, archive_generic as ag

from .enums import archive_messages as am
from .block import ModelBlock

class CommitteeBasedArchive(GenericArchive):
	
	def _handle_upload_blockchain_aggregated_model_request(self, msg: dict):
		if type(msg) != dict:
			raise TypeError("Archive _handle_upload_blockchain_aggregated_model_request method")
		elif any(key not in msg.keys() for key in am.UploadBlockchainAggregatedModelRequestBody.list()):
			raise Exception(f"Invalid syntax in UploadBlockchainAggregatedModelRequestBody. Message keys: {msg.keys()}")

		#Retrieve the model object from the json
		block_hash = None
		peer_id = msg[am.UploadBlockchainAggregatedModelRequestBody.PEER_ID]
		model_name = msg[am.UploadBlockchainAggregatedModelRequestBody.MODEL_NAME]
		timestamp = msg[am.UploadBlockchainAggregatedModelRequestBody.TIMESTAMP]
		round = msg[am.UploadBlockchainAggregatedModelRequestBody.ROUND]
		involved_trainers = msg[am.UploadBlockchainAggregatedModelRequestBody.INVOLVED_TRAINERS]
		available_nodes = msg[am.UploadBlockchainAggregatedModelRequestBody.AVAILABLE_NODES]
		list_of_current_validators = msg[am.UploadBlockchainAggregatedModelRequestBody.LIST_OF_CURRENT_VALIDATORS]

		response = None

		try:		
			with self._current_round_number_lock:
				with self._blockchain_lock:
					previous_hash = self._blockchain[-1].get_block_hash()
					model = ModelBlock(previous_hash, timestamp, model_name, round, involved_trainers, list_of_current_validators, available_nodes)
					block_hash = model.get_block_hash()
				
					#Add the model to the blockchain
					self._blockchain.append(model)
					self._block_index_list[block_hash] = model

					self._current_round_number += 1
			
			with self._available_nodes_at_the_end_of_the_round_lock:
				self._available_nodes_at_the_end_of_the_round = available_nodes
				self._ready_message_received_from_available_nodes = []
		
			response = shared_am.GenericArchiveResponse.builder(shared_am.ArchiveResponseTypes.UPLOAD_BLOCKCHAIN_BLOCK, body=shared_am.UploadBlockchainBlockResponseBody.builder(block_hash), round=self.current_round_number() - 1)

		except Exception as e:
			self._logger.record(f"Exception while uploading new model block. Peer id: {peer_id}", exc = e, logLevel = diagnostic.ERROR, identifier = self._node_name)
			raise e
		else:
			self._logger.record(f"New aggregated model block uploaded. Peer id: {peer_id}. Model name: {model_name}. Model hash: {block_hash}", logLevel = diagnostic.DEBUG, identifier = self._node_name)

		return response

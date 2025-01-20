import datetime

from .common import AbstractEnum

###
# Generic enums
###

class ArchiveMessageTypes(AbstractEnum):
	REQUEST = "request"
	RESPONSE = "response"

class GenericArchiveMessage(AbstractEnum):
	TIMESTAMP = "timestamp"
	TYPE = "type"
	SUBTYPE = "subtype"
	BODY = "body"
	ROUND = "round"

	@staticmethod
	def builder(type, subtype, body, round, timestamp = None):
		
		if timestamp is None:
			timestamp = datetime.datetime.now().timestamp()

		return {GenericArchiveMessage.TYPE: type, GenericArchiveMessage.SUBTYPE: subtype, GenericArchiveMessage.BODY: body, GenericArchiveMessage.ROUND: round, GenericArchiveMessage.TIMESTAMP: timestamp}

###
# Requests enums
###

class GenericArchiveRequestTypes(AbstractEnum):
	DOWNLOAD_LAST_BLOCKCHAIN_BLOCK = "download_last_blockchain_block"
	DOWNLOAD_BLOCKCHAIN_BLOCK = "download_blockchain_block"
	DOWNLOAD_GENESIS = "download_genesis"

class GenericArchiveRequest(GenericArchiveMessage):
	@staticmethod
	def builder(subtype, body, round, timestamp = None):
		return GenericArchiveMessage.builder(ArchiveMessageTypes.REQUEST, subtype, body, round, timestamp)

class GenericArchiveRequestBody(AbstractEnum):
	PEER_ID = "peer_id"

	@staticmethod
	def builder(peer_id):
		return {GenericArchiveRequestBody.PEER_ID: peer_id}

class ArchiveRequestTypes(GenericArchiveRequestTypes):
	UPLOAD_PEER_UPDATE = "upload_peer_update"
	DOWNLOAD_PEER_UPDATE = "download_peer_update"
	UPLOAD_AGGREGATED_MODEL = "upload_aggregated_model"
	AUTHORIZATION_TO_STORE_AGGREGATED_MODEL_DIRECTLY = "authorization_to_store_aggregated_model_directly"
	AUTHORIZATION_TO_READ_AGGREGATED_MODEL_DIRECTLY = "authorization_to_read_aggregated_model_directly"
	AUTHORIZATION_TO_STORE_UPDATE_DIRECTLY = "authorization_to_store_update_directly"
	AUTHORIZATION_TO_READ_UPDATE_DIRECTLY = "authorization_to_read_update_directly"
	DOWNLOAD_AGGREGATED_MODEL = "download_aggregated_model"
	DOWNLOAD_LAST_AGGREGATED_MODEL_BLOCK = "download_last_aggregated_model_block"
	UPLOAD_BLOCKCHAIN_AGGREGATED_MODEL = "upload_blockchain_aggregated_model"
	READY_FOR_NEXT_ROUND = "ready_for_next_round"
	START_NEXT_ROUND = "start_next_round"

class AuthorizationToStoreUpdateDirectlyRequestBody(GenericArchiveRequestBody):
	UPDATE_NAME = "update_name"
	NUM_SAMPLES = "num_samples"

	@staticmethod
	def builder(peer_id, update_name, num_samples):
		body = GenericArchiveRequestBody.builder(peer_id)
		body = {**body, AuthorizationToStoreUpdateDirectlyRequestBody.UPDATE_NAME: update_name, AuthorizationToStoreUpdateDirectlyRequestBody.NUM_SAMPLES: num_samples}
		return body

class UploadPeerUpdateRequestBody(AuthorizationToStoreUpdateDirectlyRequestBody):
	UPDATE_WEIGHTS = "update_weights"

	@staticmethod
	def builder(peer_id, update_name, update_weights, num_samples):
		body = GenericArchiveRequestBody.builder(peer_id)
		body = {**body, UploadPeerUpdateRequestBody.UPDATE_NAME: update_name, UploadPeerUpdateRequestBody.UPDATE_WEIGHTS: update_weights, UploadPeerUpdateRequestBody.NUM_SAMPLES: num_samples}
		return body

class AuthorizationToReadUpdateDirectlyRequestBody(GenericArchiveRequestBody):
	UPDATE_NAME = "update_name"

	@staticmethod
	def builder(peer_id, update_name):
		body = GenericArchiveRequestBody.builder(peer_id)
		body = {**body, DownloadPeerUpdateRequestBody.UPDATE_NAME: update_name}
		return body

class DownloadPeerUpdateRequestBody(AuthorizationToReadUpdateDirectlyRequestBody):
	pass

class AuthorizationToStoreAggregatedModelDirectlyRequestBody(GenericArchiveRequestBody):
	MODEL_NAME = "model_name"

	@staticmethod
	def builder(peer_id, model_name, ):
		body = GenericArchiveRequestBody.builder(peer_id)
		body = {**body, AuthorizationToStoreAggregatedModelDirectlyRequestBody.MODEL_NAME: model_name}
		return body

class UploadAggregatedModelRequestBody(AuthorizationToStoreAggregatedModelDirectlyRequestBody):
	MODEL_WEIGHTS = "model_weights"
	MODEL_OPTIMIZER = "model_optimizer"

	@staticmethod
	def builder(peer_id, model_name, model_weights, model_optimizer):
		body = GenericArchiveRequestBody.builder(peer_id)
		body = {**body, UploadAggregatedModelRequestBody.MODEL_NAME: model_name, UploadAggregatedModelRequestBody.MODEL_WEIGHTS: model_weights, UploadAggregatedModelRequestBody.MODEL_OPTIMIZER: model_optimizer}
		return body

class AuthorizationToReadAggregatedModelDirectlyRequestBody(GenericArchiveRequestBody):
	MODEL_NAME = "model_name"

	@staticmethod
	def builder(peer_id, model_name):
		body = GenericArchiveRequestBody.builder(peer_id)
		body = {**body, DownloadAggregatedModelRequestBody.MODEL_NAME: model_name}
		return body

class DownloadAggregatedModelRequestBody(AuthorizationToReadAggregatedModelDirectlyRequestBody):
	pass

class DownloadLastAggregatedModelRequestBody(GenericArchiveRequestBody):
	pass

class UploadBlockchainUpdateRequestBody(GenericArchiveRequestBody):
	UPDATE_NAME = "update_name"
	UPDATER_ID = "updater_id"
	SCORE = "score"
	ROUND = "round"
	TIMESTAMP = "timestamp"

	@staticmethod
	def builder(peer_id, updater_id, update_name, score, round, timestamp):
		body = GenericArchiveRequestBody.builder(peer_id)
		body = {**body, UploadBlockchainUpdateRequestBody.UPDATE_NAME: update_name, UploadBlockchainUpdateRequestBody.SCORE: score, UploadBlockchainUpdateRequestBody.ROUND: round, UploadBlockchainUpdateRequestBody.TIMESTAMP: timestamp, UploadBlockchainUpdateRequestBody.UPDATER_ID: updater_id}
		return body

# IMPORTANT: It must be overridden by the subclasses
class UploadBlockchainAggregatedModelRequestBody(GenericArchiveRequestBody):

	@staticmethod
	def builder():
		raise NotImplementedError

class DownloadLastBlockchainBlockRequestBody(GenericArchiveRequestBody):
	pass

class DownloadBlockchainBlockRequestBody(GenericArchiveRequestBody):
	BLOCK_HASH = "block_hash"

	@staticmethod
	def builder(peer_id, block_hash):
		body = GenericArchiveRequestBody.builder(peer_id)
		body = {**body, DownloadBlockchainBlockRequestBody.BLOCK_HASH: block_hash}
		return body

class DownloadGenesisRequestBody(GenericArchiveRequestBody):
	pass

class ReadyForNextRoundRequestBody(GenericArchiveRequestBody):
	pass

class StartNextRoundRequestBody(GenericArchiveRequestBody):
	pass

###
# Responses enums
###

class GenericArchiveResponseTypes(AbstractEnum):
	GENERIC_ERROR = "error"
	GENERIC_SUCCESS = "generic_success"
	DOWNLOAD_BLOCKCHAIN_BLOCK = "download_blockchain_block"
	INVALID_ROUND_NUMBER = "invalid_round_number"

class GenericArchiveResponse(GenericArchiveMessage):
	@staticmethod
	def builder(subtype: str, body: dict, round, timestamp: float = None):
		return GenericArchiveMessage.builder(ArchiveMessageTypes.RESPONSE, subtype, body, round, timestamp)

class GenericArchiveResponseBody(AbstractEnum):
	@staticmethod
	def builder():
		return {}	

class ArchiveResponseTypes(GenericArchiveResponseTypes):
	DOWNLOAD_PEER_UPDATE = "download_peer_update"
	DOWNLOAD_AGGREGATED_MODEL = "download_aggregated_model"
	DOWNLOAD_LAST_AGGREGATED_MODEL_BLOCK = "download_last_aggregated_model_block"
	UPLOAD_BLOCKCHAIN_BLOCK = "upload_blockchain_block"
	WAIT_FOR_NEXT_ROUND = "wait_for_next_round"

class MaxNumOfUpdatesReachedResponseBody(GenericArchiveResponseBody):
	pass

class DownloadPeerUpdateResponseBody(GenericArchiveResponseBody):
	UPDATE_NAME = "update_name"
	UPDATE_JSON_WEIGHTS = "update_weights"
	NUM_SAMPLES = "num_samples"
	PEER_ID = "peer_id"

	@staticmethod
	def builder(update_name: str, update_weights: dict, num_samples: int, peer_id: int):
		body = GenericArchiveResponseBody.builder()
		body = {**body, DownloadPeerUpdateResponseBody.UPDATE_NAME: update_name, DownloadPeerUpdateResponseBody.NUM_SAMPLES: num_samples, DownloadPeerUpdateResponseBody.UPDATE_JSON_WEIGHTS: update_weights, DownloadPeerUpdateResponseBody.PEER_ID: peer_id}
		return body

class DownloadAggregatedModelResponseBody(GenericArchiveResponseBody):
	MODEL_NAME = "model_name"
	MODEL_WEIGHTS = "model_weights"
	MODEL_OPTIMIZER = "model_optimizer"

	@staticmethod
	def builder(model_name: str, model_weights: dict, model_optimizer: dict):
		body = GenericArchiveResponseBody.builder()
		body = {**body, DownloadAggregatedModelResponseBody.MODEL_NAME: model_name, DownloadAggregatedModelResponseBody.MODEL_WEIGHTS: model_weights, DownloadAggregatedModelResponseBody.MODEL_OPTIMIZER: model_optimizer}
		return body

class DownloadLastAggregatedModelResponseBody(GenericArchiveResponse):
	BLOCK = "block"
	MODEL_FOUND = "model_found"

	@staticmethod
	def builder(block, model_found):
		body = GenericArchiveResponseBody.builder()
		body = {**body, DownloadLastAggregatedModelResponseBody.BLOCK: block, DownloadLastAggregatedModelResponseBody.MODEL_FOUND: model_found}
		return body
	
class UploadBlockchainBlockResponseBody(GenericArchiveResponse):
	BLOCK_HASH = "block_hash"

	@staticmethod
	def builder(block_hash):
		body = GenericArchiveResponseBody.builder()
		body = {**body, UploadBlockchainBlockResponseBody.BLOCK_HASH: block_hash}
		return body

class ErrorResponseBody(GenericArchiveResponseBody):
	ERROR = "error"

	@staticmethod
	def builder(error: str):
		body = GenericArchiveResponseBody.builder()
		body = {**body, ErrorResponseBody.ERROR: error}
		return body
	
class WaitResponseBody(GenericArchiveResponseBody):
	pass

class SuccessResponseBody(GenericArchiveResponseBody):
	pass

class InvalidRoundNumberResponseBody(GenericArchiveResponseBody):
	pass
	
class DownloadBlockchainBlockResponseBody(GenericArchiveResponse):
	BLOCK = "block"
	BLOCK_TYPE = "type"

	@staticmethod
	def builder(block, block_type):
		body = GenericArchiveResponseBody.builder()
		body = {**body, DownloadBlockchainBlockResponseBody.BLOCK: block, DownloadBlockchainBlockResponseBody.BLOCK_TYPE:block_type}
		return body
import datetime

from ...shared.enums.common import AbstractEnum
from ...shared.enums.peer_messages import GenericPeerMessage

class NodeMessageTypes(AbstractEnum):
	BROADCAST_UPDATE_UPLOAD = "broadcast_update_upload"
	BROADCAST_VALIDATION_RESULT = "broadcast_validation_result"
	BROADCAST_NEW_MODEL_BLOCK_ALERT = "broadcast_new_model_block_alert"
	BROADCAST_MULTIPLE_VALIDATION_RESULTS = "broadcast_multiple_validation_results"

class GenericNodeMessageBody(AbstractEnum):
	@staticmethod
	def builder():
		return {}

class GenericNodeMessage(GenericPeerMessage):
	ROUND = "round"

	@staticmethod
	def builder(type, peer_id, body, round, timestamp = None):
		if timestamp is None:
			timestamp = datetime.datetime.now().timestamp()

		return {GenericNodeMessage.TYPE: type, GenericNodeMessage.PEER_ID: peer_id, GenericNodeMessage.TIMETAMP: timestamp, GenericNodeMessage.BODY: body, GenericNodeMessage.ROUND: round}

class BroadcastUpdateUpload(GenericNodeMessage):
	UPDATE_NAME = "update_name"

	@staticmethod
	def builder(update_name, peer_id, round):
		body = GenericNodeMessageBody.builder()
		body = {**body, BroadcastUpdateUpload.UPDATE_NAME: update_name}
		return GenericNodeMessage.builder(NodeMessageTypes.BROADCAST_UPDATE_UPLOAD, peer_id, body, round)
	
class BroadcastValidationResult(GenericNodeMessage):
	RESULT = "result"
	SCORE = "score"
	UPDATE_NAME = "update_name"
	UPDATER_ID = "updater_id"

	@staticmethod
	def builder(result, score, update_name, updater_id, validator_id, round):
		body = GenericNodeMessageBody.builder()
		body = {**body, BroadcastValidationResult.RESULT: result, BroadcastValidationResult.SCORE: score, BroadcastValidationResult.UPDATE_NAME: update_name, BroadcastValidationResult.UPDATER_ID: updater_id}
		return GenericNodeMessage.builder(NodeMessageTypes.BROADCAST_VALIDATION_RESULT, validator_id, body, round)
	
class BroadcastMultipleValidationResults(GenericNodeMessage):
	
	RESULTS = "results"

	@staticmethod
	def builder(results, validator_id, round):
		body = GenericNodeMessageBody.builder()
		body = {**body, BroadcastMultipleValidationResults.RESULTS: results}
		return GenericNodeMessage.builder(NodeMessageTypes.BROADCAST_MULTIPLE_VALIDATION_RESULTS, validator_id, body, round)

class BroadcastNewModelBlockAlert(GenericNodeMessage):
	BLOCK_HASH = "block_hash"

	@staticmethod
	def builder(block_hash, peer_id, round):
		body = GenericNodeMessageBody.builder()
		body = {**body, BroadcastNewModelBlockAlert.BLOCK_HASH: block_hash}
		return GenericNodeMessage.builder(NodeMessageTypes.BROADCAST_NEW_MODEL_BLOCK_ALERT, peer_id, body, round)
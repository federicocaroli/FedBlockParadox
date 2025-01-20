from ...shared.enums import archive_messages as shared_am

class UploadBlockchainAggregatedModelRequestBody(shared_am.GenericArchiveRequestBody):
	MODEL_NAME = "model_name"
	ROUND = "round"
	TIMESTAMP = "timestamp"
	INVOLVED_TRAINERS = "involved_trainers"
	LIST_OF_CURRENT_VALIDATORS = "list_of_current_validators"
	AVAILABLE_NODES = "available_nodes"
	NODE_STAKES = "node_stakes"

	@staticmethod
	def builder(peer_id, model_name, round, timestamp, involved_trainers, list_of_current_validators, available_nodes, node_stakes):
		body = shared_am.GenericArchiveRequestBody.builder(peer_id)
		body = {**body, UploadBlockchainAggregatedModelRequestBody.MODEL_NAME: model_name, UploadBlockchainAggregatedModelRequestBody.ROUND: round, UploadBlockchainAggregatedModelRequestBody.TIMESTAMP: timestamp, UploadBlockchainAggregatedModelRequestBody.INVOLVED_TRAINERS: involved_trainers, UploadBlockchainAggregatedModelRequestBody.LIST_OF_CURRENT_VALIDATORS: list_of_current_validators, UploadBlockchainAggregatedModelRequestBody.AVAILABLE_NODES: available_nodes, UploadBlockchainAggregatedModelRequestBody.NODE_STAKES: node_stakes}
		return body
from ...shared.enums.common import AbstractEnum

class InvalidRoundException(Exception):
	'''Exception for Invalid Round'''

	pass

class ValidationResultsListElemFields(AbstractEnum):
	UPDATE_NAME = "update_name"
	VALIDATION_RESULT = "validation_result"
	VALIDATION_SCORE = "validation_score"

class ModelToFitDictFields(AbstractEnum):
	WEIGHTS = "weights"
	OPTIMIZER = "optimizer"
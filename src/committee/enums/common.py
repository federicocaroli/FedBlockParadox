from ...shared.enums.common import AbstractEnum

# DIVERSO DALLE ALTRE ARCHITETTURE
class AdditionalCommitteeSpecificConfigParams(AbstractEnum):
	NUM_OF_VALIDATORS = "num_of_validators"
	PERC_THRESHOLD_TO_PASS_VALIDATION = "perc_threshold_to_pass_validation"
	PERC_OF_TRAINERS_ACTIVE_IN_EACH_ROUND = "perc_of_trainers_active_in_each_round"
	LIST_OF_ACTIVE_TRAINERS_IN_THE_FIRST_ROUND = "list_of_active_trainers_in_the_first_round"
	LIST_OF_NODE_IDS_OF_FIRST_VALIDATORS = "list_of_node_ids_of_first_validators"



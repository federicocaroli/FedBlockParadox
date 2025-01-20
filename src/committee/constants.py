from .enums.common import AdditionalCommitteeSpecificConfigParams

# DIVERSO DALLE ALTRE ARCHITETTURE
DEFAULT_COMMITTEE_SPECIFIC_CONFIG = {
	AdditionalCommitteeSpecificConfigParams.NUM_OF_VALIDATORS: 5,
	AdditionalCommitteeSpecificConfigParams.PERC_THRESHOLD_TO_PASS_VALIDATION: 0.8,
	AdditionalCommitteeSpecificConfigParams.PERC_OF_TRAINERS_ACTIVE_IN_EACH_ROUND: 0.8,
    AdditionalCommitteeSpecificConfigParams.LIST_OF_ACTIVE_TRAINERS_IN_THE_FIRST_ROUND: None,
    AdditionalCommitteeSpecificConfigParams.LIST_OF_NODE_IDS_OF_FIRST_VALIDATORS: None,
}
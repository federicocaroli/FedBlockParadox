from .enums.common import AdditionalPosSpecificConfigParams

# DIVERSO DALLE ALTRE ARCHITETTURE
DEFAULT_POS_SPECIFIC_CONFIG = {
	AdditionalPosSpecificConfigParams.NUM_OF_VALIDATORS: 3,
	AdditionalPosSpecificConfigParams.PERC_THRESHOLD_TO_PASS_VALIDATION: 0.7,
	AdditionalPosSpecificConfigParams.PERC_OF_TRAINERS_ACTIVE_IN_EACH_ROUND: 1,
    AdditionalPosSpecificConfigParams.LIST_OF_ACTIVE_TRAINERS_IN_THE_FIRST_ROUND: None,
    AdditionalPosSpecificConfigParams.LIST_OF_NODE_IDS_OF_FIRST_VALIDATORS: None,
    AdditionalPosSpecificConfigParams.DEFAULT_INITIAL_NODE_STAKE: 10,
    AdditionalPosSpecificConfigParams.KNOWN_INITIAL_NODES_STAKES: {},
    AdditionalPosSpecificConfigParams.AMOUNT_OF_STAKE_TO_ADD: 5
}
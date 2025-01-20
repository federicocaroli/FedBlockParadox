from .enums.common import AdditionalPowSpecificConfigParams

# DIVERSO DALLE ALTRE ARCHITETTURE
DEFAULT_POW_SPECIFIC_CONFIG = {
	AdditionalPowSpecificConfigParams.PERC_OF_TRAINERS_ACTIVE_IN_EACH_ROUND: 1,
    AdditionalPowSpecificConfigParams.LIST_OF_ACTIVE_TRAINERS_IN_THE_FIRST_ROUND: None,
    AdditionalPowSpecificConfigParams.NODES_COMPUTING_POWER: []
}
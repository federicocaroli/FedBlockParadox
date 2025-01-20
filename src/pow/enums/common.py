from ...shared.enums.common import AbstractEnum, NodeSpecificConfigParams, AttackerSpecificConfigParams

class AdditionalPowSpecificConfigParams(AbstractEnum):
	PERC_OF_TRAINERS_ACTIVE_IN_EACH_ROUND = "perc_of_trainers_active_in_each_round"
	LIST_OF_ACTIVE_TRAINERS_IN_THE_FIRST_ROUND = "list_of_active_trainers_in_the_first_round"
	NODES_COMPUTING_POWER = "nodes_computing_power"

class NodesComputingPowerConfigElemFields(AbstractEnum):
	ALIAS = "alias"
	NODE_IDS = "node_ids"
	COMPUTING_POWER_FACTOR = "computing_power_factor"

class PowBasedNodeSpecificConfigParams(NodeSpecificConfigParams):
	COMPUTING_POWER_FACTOR = "computing_power_factor"

class PowBasedAttackerSpecificConfigParams(AttackerSpecificConfigParams):
	COMPUTING_POWER_FACTOR = "computing_power_factor"

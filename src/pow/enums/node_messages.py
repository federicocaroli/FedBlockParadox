from ...shared.enums import node_messages as shared_nm

class PowBasedNodeMessageTypes(shared_nm.NodeMessageTypes):
    BROADCAST_NODE_COMPUTING_POWER_FACTOR = "broadcast_node_computing_power_factor"

class BroadcastNodeComputingPowerFactor(shared_nm.GenericNodeMessage):
    COMPUTING_POWER_FACTOR = "computing_power_factor"

    @staticmethod
    def builder(computing_power_factor, peer_id, round):
        body = shared_nm.GenericNodeMessageBody.builder()
        body = {**body, BroadcastNodeComputingPowerFactor.COMPUTING_POWER_FACTOR: computing_power_factor}
        return shared_nm.GenericNodeMessage.builder(PowBasedNodeMessageTypes.BROADCAST_NODE_COMPUTING_POWER_FACTOR, peer_id, body, round)
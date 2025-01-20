import copy

from .common import AbstractEnum

class PeersListElement(AbstractEnum):
	HOST = "host"
	PORT = "port"
	STATUS = "status"
	LAST_UPDATE_TIMESTAMP = "last_update_timestamp"

	@staticmethod
	def builder(host, port, status, last_update_timestamp):
		return {
			PeersListElement.HOST: host,
			PeersListElement.PORT: port,
			PeersListElement.STATUS: status,
			PeersListElement.LAST_UPDATE_TIMESTAMP: last_update_timestamp
		}

class PeersListUtilities:
	@staticmethod
	def _merge_two_peers_list(new_list, current_list):
		
		need_to_propagate_changes_again = False
		merged_list = copy.deepcopy(new_list)

		for peer_id in current_list:
			if peer_id not in new_list:
				merged_list[peer_id] = current_list[peer_id]
				need_to_propagate_changes_again = True
			elif current_list[peer_id][PeersListElement.LAST_UPDATE_TIMESTAMP] > new_list[peer_id][PeersListElement.LAST_UPDATE_TIMESTAMP]:
				merged_list[peer_id] = current_list[peer_id]
				need_to_propagate_changes_again = True

		return merged_list, need_to_propagate_changes_again
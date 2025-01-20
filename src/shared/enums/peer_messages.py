import datetime

from .common import AbstractEnum

class PeerMessageTypes(AbstractEnum):
	HELLO = "hello"
	GOODBYE = "goodbye"
	BROADCAST_NEW_PEER_IN_PEERS_LIST = "broadcast_new_peer_in_peers_list"
	BROADCAST_CHANGED_PEERS_LIST = "broadcast_changed_peers_list"
	
class GenericPeerMessage(AbstractEnum):
	TYPE = "type"
	PEER_ID = "peer_id"
	TIMETAMP = "timestamp"
	BODY = "body"

	@staticmethod
	def builder(type, peer_id, body, timestamp = None):
		if timestamp is None:
			timestamp = datetime.datetime.now().timestamp()

		return {GenericPeerMessage.TYPE: type, GenericPeerMessage.PEER_ID: peer_id, GenericPeerMessage.TIMETAMP: timestamp, GenericPeerMessage.BODY: body}

class GenericPeerMessageBody(AbstractEnum):
	@staticmethod
	def builder():
		return {}

class HelloMessageBody(GenericPeerMessageBody):
	HOST = "host"
	PORT = "port"

	@staticmethod
	def builder(host, port, peer_id):
		body = GenericPeerMessageBody.builder()
		body = {**body, HelloMessageBody.HOST: host, HelloMessageBody.PORT: port}
		return GenericPeerMessage.builder(PeerMessageTypes.HELLO, peer_id, body)

class BroadcastChangedPeersListMessageBody(GenericPeerMessageBody):
	PEERS_LIST = "peers_list"

	@staticmethod
	def builder(peers_list, peer_id):
		body = GenericPeerMessageBody.builder()
		body = {**body, BroadcastChangedPeersListMessageBody.PEERS_LIST: peers_list}
		return GenericPeerMessage.builder(PeerMessageTypes.BROADCAST_CHANGED_PEERS_LIST, peer_id, body)
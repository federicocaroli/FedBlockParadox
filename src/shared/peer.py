import socket, copy, datetime, time, gc

from threading import Lock
from .enums.peer_messages import GenericPeerMessage, HelloMessageBody, BroadcastChangedPeersListMessageBody, PeerMessageTypes
from .enums.peer_generic import PeersListElement, PeersListUtilities
from .socket_node import SocketNode
from . import diagnostic

class Peer(SocketNode):
	"""
	Represents a peer in the simulator.

	Attributes:
		_base_log_msg (str): The base log message for the peer.
		_peer_id (int): The ID of the peer.
		_peers (dict): A dictionary of available peers.
		_peers_lock (Lock): A lock for thread safety.

	Methods:
		__init__(peer_id, host, port, already_available_peers_list={}): Initializes a new instance of the Peer class.
		stop(): Stops the peer.
		_handle_msg(msg): Handles an incoming message.
		_handle_custom_msg(msg): Handles a custom message.
		_handle_hello_msg(peer_id, timestamp, msg): Handles a hello message.
		_handle_changed_peers_list_msg(msg): Handles a changed peers list message.
		_send_message(message, available_peers_for_broadcast=None, receiver_peer_host_for_unicast=None, receiver_peer_port_for_unicast=None): Sends a message to other peers.
		connect_to_peer(host, port): Connects to a peer.
	"""
	def __init__(self, logger, peer_id: int, host: str, port: int, already_available_peers_list: dict = {}, allowed_to_write_redudant_log_messages: bool = False):
		"""
		Initializes a new instance of the Peer class.

		Args:
			peer_id (int): The ID of the peer.
			host (str): The host of the peer.
			port (int): The port of the peer.
			already_available_peers_list (dict, optional): A dictionary of already available peers. Defaults to an empty dictionary.
		"""

		if type(peer_id) != int or type(host) != str or type(port) != int or type(already_available_peers_list) != dict or type(allowed_to_write_redudant_log_messages) != bool:
			raise TypeError("Peer __init__ method")

		super().__init__(f"Peer-{peer_id}", host, port)

		self._str_identifier = f"Peer-{peer_id}"        
		self._peer_id = peer_id
		self._peers = copy.deepcopy(already_available_peers_list)
		self._peers_lock = Lock()
		self._logger = logger
		self._allowed_to_write_redudant_log_messages = allowed_to_write_redudant_log_messages

		if self._peer_id not in self._peers:
			with self._peers_lock:
				self._peers[self._peer_id] = PeersListElement.builder(host, port, True, datetime.datetime.now().timestamp())
		elif self._peers[self._peer_id][PeersListElement.HOST] != host or self._peers[self._peer_id][PeersListElement.PORT] != port:
			raise Exception("Peer id is already in use")
		elif self._peers[self._peer_id][PeersListElement.STATUS] is False:
			with self._peers_lock:
				self._peers[self._peer_id][PeersListElement.STATUS] = True
				self._peers[self._peer_id][PeersListElement.LAST_UPDATE_TIMESTAMP] = datetime.datetime.now().timestamp()

	def is_peer_alive(self) -> bool:
		"""
		Checks if the peer is alive.

		Returns:
			bool: True if the peer is alive, False otherwise.
		"""
		with self._peers_lock:
			return self._peers[self._peer_id][PeersListElement.STATUS]

	def get_number_of_peers_alive(self) -> int:
		"""
		Gets the number of peers that are alive.

		Returns:
			int: The number of peers that are alive.
		"""
		with self._peers_lock:
			return sum(1 for peer_id in self._peers if self._peers[peer_id][PeersListElement.STATUS] is True)

	def stop(self):
		"""
		Stops the peer.
		"""
		
		if self.is_peer_alive() is True:
			
			current_timestamp = datetime.datetime.now().timestamp()

			with self._peers_lock:
				self._peers[self._peer_id][PeersListElement.STATUS] = False
				self._peers[self._peer_id][PeersListElement.LAST_UPDATE_TIMESTAMP] = current_timestamp

			self._send_message(BroadcastChangedPeersListMessageBody.builder(self._peers, self._peer_id))

		super().stop()

	def _handle_msg(self, msg: dict) -> (dict | None):
		"""
		Handles an incoming message.

		Args:
			msg (dict): The incoming message.

		Returns:
			Optional[dict]: The reply message, if any.
		"""
		reply = None

		try:
			if any(key not in msg.keys() for key in GenericPeerMessage.list()):
				raise Exception(f"Invalid syntax in GenericPeerMessage. Message keys: {msg.keys()}")
			
			peer_id = msg[GenericPeerMessage.PEER_ID]
			timestamp = msg[GenericPeerMessage.TIMETAMP]

			msg_body = msg[GenericPeerMessage.BODY]

			if msg[GenericPeerMessage.TYPE] == PeerMessageTypes.HELLO:
				self._handle_hello_msg(peer_id, timestamp, msg_body)
			
			elif msg[GenericPeerMessage.TYPE] == PeerMessageTypes.BROADCAST_CHANGED_PEERS_LIST:
				self._handle_changed_peers_list_msg(msg_body)
			
			else:
				reply = self._handle_custom_msg(msg)

		except Exception as e:
			self._logger.record(msg = f"Exception while handling a message. Msg: {msg}", exc = e, logLevel = diagnostic.ERROR, identifier = self._str_identifier)

		return reply
	
	def _handle_custom_msg(self, msg: dict) -> (dict | None):
		"""
		Handles a custom message.

		Args:
			msg (dict): The custom message.

		Returns:
			Optional[dict]: The reply message, if any.
		"""
		raise NotImplementedError

	def _handle_hello_msg(self, peer_id, timestamp, msg):
		"""
		Handles a hello message.

		Args:
			peer_id (int): The ID of the peer.
			timestamp (float): The timestamp of the message.
			msg (dict): The hello message.
		"""
		try:
			new_peer_host = msg[HelloMessageBody.HOST]
			new_peer_port = msg[HelloMessageBody.PORT]

			with self._peers_lock:
				if peer_id in self._peers:
					if self._peers[peer_id][PeersListElement.LAST_UPDATE_TIMESTAMP] >= timestamp:
						raise Exception("Peer status is more recent than the message's timestamp")
					
					if self._peers[peer_id][PeersListElement.STATUS] is True:
						raise Exception("Peer id is already in use")
					
					elif self._peers[peer_id][PeersListElement.HOST] != new_peer_host or self._peers[peer_id][PeersListElement.PORT] != new_peer_port:
						raise Exception("Peer id is associated to a different host/port pair")
	
				elif any(config[PeersListElement.HOST] == new_peer_host and config[PeersListElement.PORT] == new_peer_port for config in self._peers.values()):
					raise Exception("Peer host/port pair is already in use")
				
				self._peers[peer_id] = PeersListElement.builder(new_peer_host, new_peer_port, True, timestamp)

			printed_peers = {peer_id: {PeersListElement.STATUS: self._peers[peer_id][PeersListElement.STATUS]} for peer_id in self._peers}

			if self._allowed_to_write_redudant_log_messages:
				self._logger.record(msg = f"New peer connected. New peer id: {peer_id}. New peer host: {new_peer_host}. New peer port: {new_peer_port}. New list: {printed_peers}", logLevel = diagnostic.DEBUG, identifier = self._str_identifier)

			self._send_message(BroadcastChangedPeersListMessageBody.builder(self._peers, self._peer_id))

		except Exception as e:
			self._logger.record(msg = f"Exception while handling hello message. Message: {msg}", exc = e, logLevel = diagnostic.ERROR, identifier = self._str_identifier)

	def _handle_changed_peers_list_msg(self, msg):
		"""
		Handles a changed peers list message.

		Args:
			msg (dict): The changed peers list message.
		"""
		if any(key not in msg for key in BroadcastChangedPeersListMessageBody.list()):
			raise Exception(f"Invalid syntax in BroadcastChangedPeersListMessageBody. Message keys: {msg.keys()}")
		
		with self._peers_lock:
			changed_peers_list = msg[BroadcastChangedPeersListMessageBody.PEERS_LIST]
			
			parsed_changed_peers_list = dict()
			for peer_id in changed_peers_list:
				parsed_changed_peers_list[int(peer_id)] = changed_peers_list[peer_id]

			merged_list, need_to_propagate_changes_again = PeersListUtilities._merge_two_peers_list(parsed_changed_peers_list, self._peers)

			if need_to_propagate_changes_again:
				printed_peers = {peer_id: {PeersListElement.STATUS: self._peers[peer_id][PeersListElement.STATUS]} for peer_id in self._peers}
				printed_merged_list = {peer_id: {PeersListElement.STATUS: merged_list[peer_id][PeersListElement.STATUS]} for peer_id in merged_list}
				printed_changed_peers_list = {peer_id: {PeersListElement.STATUS: parsed_changed_peers_list[peer_id][PeersListElement.STATUS]} for peer_id in parsed_changed_peers_list}
				
				if self._allowed_to_write_redudant_log_messages:
					self._logger.record(msg = f"Merging two peers list's changes. Current list: {printed_peers}. New: {printed_changed_peers_list}. Merged list: {printed_merged_list}", logLevel = diagnostic.DEBUG, identifier = self._str_identifier)

			self._peers = merged_list

		if need_to_propagate_changes_again:
			if self._allowed_to_write_redudant_log_messages:
				printed_peers = {peer_id: {PeersListElement.STATUS: self._peers[peer_id][PeersListElement.STATUS]} for peer_id in self._peers}
				self._logger.record(msg = f"New list: {printed_peers}", logLevel = diagnostic.DEBUG, identifier = self._str_identifier)

			self._send_message(BroadcastChangedPeersListMessageBody.builder(self._peers, self._peer_id))

	def _send_message(self, message: dict, available_peers_for_broadcast: (dict | None) = None, receiver_peer_host_for_unicast: (str | None) = None, receiver_peer_port_for_unicast: (int | None) = None, max_number_of_connection_attempts = 3):
		"""
		Sends a message to other peers.

		Args:
			message (dict): The message to send.
			available_peers_for_broadcast (dict, optional): A dictionary of available peers for broadcasting the message. Defaults to None.
			receiver_peer_host_for_unicast (str, optional): The host of the receiver peer for unicast. Defaults to None.
			receiver_peer_port_for_unicast (int, optional): The port of the receiver peer for unicast. Defaults to None.
		"""
		
		if any(key not in message for key in GenericPeerMessage.list()):
			raise Exception(f"Invalid syntax in GenericPeerMessage. Message keys: {message.keys()}")
		elif (type(available_peers_for_broadcast) != dict and available_peers_for_broadcast is not None) or (type(receiver_peer_host_for_unicast) != str and receiver_peer_host_for_unicast is not None) or (type(receiver_peer_port_for_unicast) != int and receiver_peer_port_for_unicast is not None):
			raise TypeError("Peer _send_message method")

		try:
			receivers_host_port = list()

			if available_peers_for_broadcast is None:
				with self._peers_lock:
					available_peers_for_broadcast = copy.deepcopy(self._peers)

			if receiver_peer_host_for_unicast is not None and receiver_peer_port_for_unicast is not None:
				receivers_host_port = [(None, receiver_peer_host_for_unicast, receiver_peer_port_for_unicast)]

			else:
				for peer_id in available_peers_for_broadcast:					
					if available_peers_for_broadcast[peer_id][PeersListElement.STATUS] is True and peer_id != self._peer_id:
						receivers_host_port.append((peer_id, available_peers_for_broadcast[peer_id][PeersListElement.HOST], available_peers_for_broadcast[peer_id][PeersListElement.PORT]))

			message_to_send = self._prepare_socket_msg_to_send(message)

			for config in receivers_host_port:
			
				s = None
				socket_is_connected = False

				try:
					receiver_peer_id = config[0]

					with self._peers_lock:
						if receiver_peer_id is not None and self._peers[receiver_peer_id][PeersListElement.STATUS] is False:
							continue        

					socket_is_connected = False
					s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

					last_conn_exception = None
					for _ in range(max_number_of_connection_attempts):
						try:
							s.connect((config[1], config[2]))
						
						except Exception as e:
							last_conn_exception = e
							
						else:
							socket_is_connected = True
							break

						time.sleep(1)
					else:						
						# In case our peer is shutting down it's probable that also the other peers are shutting down, so problems of this kind can be ignored
						raise Exception(f"Receiver is unreachable. Last conn exception: {last_conn_exception}")
					
					s.sendall(message_to_send)

				except Exception as e:
					self._logger.record(msg = f"Exception while sending a message. Receiver config: {config}. Message: {message}", exc = e, logLevel = diagnostic.ERROR, identifier = self._str_identifier)
				
					with self._peers_lock:
						if receiver_peer_id is not None:
							self._peers[receiver_peer_id][PeersListElement.STATUS] = False
							self._peers[receiver_peer_id][PeersListElement.LAST_UPDATE_TIMESTAMP] = datetime.datetime.now().timestamp()

				finally:
					if s is not None:
						if socket_is_connected:
							s.shutdown(socket.SHUT_RDWR)

						s.close()
						del s

		except Exception as e:
			self._logger.record(msg = f"Exception while sending a message. Message: {message}", exc = e, logLevel = diagnostic.ERROR, identifier = self._str_identifier)
		finally:
			del message_to_send
			del receivers_host_port
			del available_peers_for_broadcast
			gc.collect()

	def connect_to_peer(self, host, port):            
		"""
		Connects to a peer.

		Args:
			host (str): The host of the peer to connect to.
			port (int): The port of the peer to connect to.
		"""
		try:
			self._send_message(HelloMessageBody.builder(self._host, self._port, self._peer_id), receiver_peer_host_for_unicast=host, receiver_peer_port_for_unicast=port)    
		except Exception as e:
			self._logger.record(msg = f"Error connecting to peer. Receiver peer id: ({host}, {port})", exc = e, logLevel = diagnostic.ERROR, identifier = self._str_identifier)
			
import socket, json, gc, datetime, sys

from concurrent.futures import ThreadPoolExecutor
from threading import Event, Thread

class SocketNode(Thread):
	"""
	Represents a socket node that listens for incoming socket messages and handles connections.

	Args:
		node_name (str): The name of the node.
		host (str): The host address to bind the socket to.
		port (int): The port number to bind the socket to.
	"""

	def __init__(self, node_name, host, port, num_of_threads_in_pool = 10):
		self._node_name = node_name
		self._host = host
		self._port = port
		self._num_of_threads_in_pool = num_of_threads_in_pool
		self._kill_thread = None

		super().__init__(name=f"{node_name} Main thread", daemon=True)

	def start(self):
		"""
		Starts the socket node.

		Raises:
			Exception: If the kill thread is already set.
		"""
		if isinstance(self._kill_thread, Event) and self._kill_thread.is_set() is False:
			raise Exception

		if self._kill_thread is None:
			self._kill_thread = Event()

		super().start()

	def stop(self):
		"""
		Stops the socket node.

		Raises:
			Exception: If the kill thread is not set or already set.
		"""
		if self._kill_thread is None:
			raise Exception("Kill thread is None")
		elif isinstance(self._kill_thread, Event) and self._kill_thread.is_set() is True:
			return

		self._kill_thread.set()

	def _prepare_socket_msg_to_send(self, msg: dict) -> bytes:
		"""
		Prepares the message to be sent over the socket.
		"""

		str_message = json.dumps(msg)
		return f"LENGTH:{len(str_message)}||MSG:{str_message}".encode()

	def _handle_msg(self, msg: dict) -> (dict | None):
		"""
		Handles the received socket message.

		Args:
			msg (str): The received socket message.

		Returns:
			Optional[str]: The response message, or None if no response is needed.

		Raises:
			NotImplementedError: If the method is not implemented in a subclass.
		"""
		raise NotImplementedError

	def _handle_connection(self, client_sock):
		"""
		Handles a connection with a client socket.

		Args:
			client_sock: The client socket object.
		"""
		try:
			while self._kill_thread.is_set() is False:
				msg = self._handle_socket_recv(client_sock)

				if msg is None:
					break

				response = self._handle_msg(json.loads(msg))
				del msg

				if response is not None:
					if type(response) != dict:
						raise TypeError

					response_encoded = self._prepare_socket_msg_to_send(response)
					client_sock.sendall(response_encoded)
					del response_encoded
				
				del response

		except Exception as e:
			print(f"{self._node_name} Exception while handling connection. Exception: {e}")
		finally:
			client_sock.shutdown(socket.SHUT_RDWR)
			client_sock.close()
			gc.collect()

	def monitor_executor(self, executor, futures):
		"""
		Monitors the executor and prints the number of pending and running tasks.
		"""
		# Count pending tasks
		pending_tasks = 0
		running_tasks = 0
		
		futures_to_remove = []

		# Remove finished futures from list
		for f in futures:
			if f.done():
				futures_to_remove.append(f)
			elif f.running():
				running_tasks += 1
			else:
				pending_tasks += 1
		
		for f in futures_to_remove:
			futures.remove(f)

		# Calculate free and busy workers
		total_workers = executor._max_workers
		free_workers = total_workers - running_tasks
		
		print(f"[{self._port}] Completed Tasks: {len(futures_to_remove)}", file=sys.stderr)
		print(f"[{self._port}] Pending Tasks: {pending_tasks}", file=sys.stderr)
		print(f"[{self._port}] Free Workers: {free_workers}", file=sys.stderr)
		print(f"[{self._port}] Busy Workers: {running_tasks}", file=sys.stderr)
		print(f"[{self._port}] Total Workers: {total_workers}", file=sys.stderr)

	def run(self):
		"""
		The main thread function that runs the socket node.
		"""
		# Debug lines
		#futures = list()

		thread_pool = ThreadPoolExecutor(max_workers= self._num_of_threads_in_pool, thread_name_prefix= self._node_name)

		server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
		server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
		server.bind((self._host, self._port))
		server.settimeout(1)
		server.listen()

		while self._kill_thread.is_set() is False:
			try:
				client, addr = server.accept()
			except socket.timeout:
				continue
			# Debug lines
			#finally:
			#	if self._port in []:
			#		if int(datetime.datetime.now().timestamp()) % 30 == 0:
			#			self.monitor_executor(thread_pool, futures)
			#
			#result = thread_pool.submit(self._handle_connection, client)
			#futures.append(result)

			thread_pool.submit(self._handle_connection, client)

		server.shutdown(socket.SHUT_RDWR)
		server.close()
		thread_pool.shutdown(wait=False)

	def _handle_socket_recv(self, client_sock, buff_size=4096):
		"""
		Receives and decodes the socket message from the client socket.

		Args:
			client_sock: The client socket object.
			buff_size (int): The buffer size for receiving data.

		Returns:
			str: The decoded socket message.

		Raises:
			Exception: If an invalid prefix is found in the message.
		"""
		encoded_msg = client_sock.recv(buff_size)

		if not encoded_msg:
			return None

		elif b"LENGTH:" not in encoded_msg:
			raise Exception(f"Invalid prefix in msg. Prefix: {encoded_msg}")

		while b"||MSG:" not in encoded_msg:
			encoded_msg += client_sock.recv(buff_size)

		tmp = encoded_msg.split(b'||MSG:')
		header = tmp[0]
		encoded_msg = tmp[1]

		msg_size = int(header.replace(b'LENGTH:', b''))

		while msg_size > len(encoded_msg):
			encoded_msg += client_sock.recv(buff_size)

		return encoded_msg.decode("utf-8")

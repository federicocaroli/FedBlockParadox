import setproctitle, string, json, os.path, os, numpy as np
from tensorflow import convert_to_tensor, GradientTape, math, float32
from tensorflow.random import normal as normal_tensorflow

from multiprocessing import Process, current_process

from . import dataset_utils as du, diagnostic, utils
from .enums import node_generic as ng
from .node import GenericNode, SampleCounterCallback
from .constants import VERBOSE, DIRECTORY_WHERE_TO_STORE_TMP_FILES

class LabelFlippingNode():
	def __init__(self, node_instance: GenericNode, selected_classes: list=[], target_classes: list=[], num_of_samples: int = 0):
		if type(selected_classes) != list or type(target_classes) != list or type(num_of_samples) != int or isinstance(node_instance, GenericNode) is False:
			raise TypeError("LabelFlippingNode costructor")
		elif num_of_samples < 0 or len(selected_classes) != len(target_classes) or len(selected_classes) == 0:
			raise ValueError("LabelFlippingNode constructor")
		
		self._node_instance = node_instance
		self._selected_classes = selected_classes
		self._target_classes = target_classes
		self._num_of_samples = num_of_samples

	def _define_nodes_active_in_the_next_round(self):
		"""
		Define the nodes that will be active in the next round. Due to the fact that the node is malicious, it will always be active.
		"""
		if self._node_instance.is_validator():
			if self._node_instance._allowed_to_write_redudant_log_messages:
				self._node_instance._logger.record(msg = f"Node {self._node_instance._peer_id} is malicious and will be a validator in the next round", logLevel = diagnostic.DEBUG, identifier= self._node_instance._str_identifier)

			return []
		
		if self._node_instance._allowed_to_write_redudant_log_messages:
			self._node_instance._logger.record(msg = f"Node {self._node_instance._peer_id} is malicious and will be a trainer in the next round", logLevel = diagnostic.DEBUG, identifier= self._node_instance._str_identifier)

		return [self._node_instance._peer_id]

	@staticmethod
	def _weight_fit_operations(is_subprocess: bool, lazy_loading: bool, model_architecture: dict, model_info: dict, dataset_quanta_paths: list, selected_classes: list, target_classes: list, fit_epochs: int, batch_size: int, filename: str | None = None):
		result = None
		
		try:
			if type(is_subprocess) != bool or type(lazy_loading) != bool or type(dataset_quanta_paths) != list or type(selected_classes) != list or type(target_classes) != list or type(fit_epochs) != int or type(batch_size) != int or (is_subprocess and (filename is not None and type(filename) != str)):
				raise TypeError("LabelFlippingNode _weight_fit_operations method")
			elif len(selected_classes) != len(target_classes) or len(selected_classes) == 0 or fit_epochs <= 0 or batch_size <= 0:
				raise ValueError("LabelFlippingNode _weight_fit_operations method")

			if is_subprocess:
				setproctitle.setproctitle(f"{current_process().name}")

			training_set = utils.get_training_set(dataset_quanta_paths, lazy_loading, batch_size)

			model = utils.build_model_from_architecture_and_weights(model_architecture, model_info[ng.ModelToFitDictFields.WEIGHTS], optimizer_variables= model_info[ng.ModelToFitDictFields.OPTIMIZER])

			if lazy_loading:
				malicious_set = du.create_lazy_flipped_dataset(training_set, selected_classes, target_classes)
			else:
				malicious_set = du.create_eager_flipped_dataset(training_set, selected_classes, target_classes, batch_size)

			if lazy_loading:
				sample_counter_callback = SampleCounterCallback(batch_size)
				history = model.fit(malicious_set, epochs=fit_epochs, callbacks=[sample_counter_callback], verbose=VERBOSE)
				num_of_samples = sample_counter_callback.result
			else:
				history = model.fit(malicious_set['img'], malicious_set['label'], epochs=fit_epochs, batch_size=batch_size, verbose=VERBOSE)
				num_of_samples = len(malicious_set['img'])
			
			weights = model.get_weights()
			weights = [arr.tolist() for arr in weights]

			result = {"weights": weights, "history": history.history, "num_of_samples": num_of_samples}

		except Exception as e:
			if is_subprocess:
				result = {"error": f"{type(e)}:{str(e)}"}
			else:
				raise e
		
		if is_subprocess:
			with open(filename, "w") as f:
				json.dump(result, f)
		else:
			return result

	def _weight_fit(self):
		"""
		Perform the malicious label-flipping training to obtain new weights.
		"""
		try:	
			with self._node_instance._malicious_training_semaphore:
				filename = os.path.join(DIRECTORY_WHERE_TO_STORE_TMP_FILES, ''.join(np.random.choice(list(string.ascii_lowercase + string.digits), size=24)) + ".json")

				training_process = Process(target = LabelFlippingNode._weight_fit_operations, args = (True, self._node_instance._lazy_loading, self._node_instance._model_architecture, self._node_instance._model_to_fit_when_not_validator, self._node_instance._dataset_quanta_paths, self._selected_classes, self._target_classes, self._node_instance._fit_epochs, self._node_instance._batch_size, filename), name= f"FedBlockSimulator - trainer_{self._node_instance._peer_id}_label_flip_weight_train", daemon= True)
				training_process.start()
				training_process.join()

			with open(filename, "r") as f:
				result = json.load(f)

			os.remove(filename)

			if type(result) != dict:
				raise Exception("Result from subprocess is not dict")
			elif "error" in result:
				raise Exception(f"Error while performing label flipping weights-based training. Error: {result['error']}")
			
			history = result["history"]
			num_of_samples = result["num_of_samples"]
			weights = result["weights"]

			if self._node_instance._allowed_to_write_redudant_log_messages:
				self._node_instance._logger.record(msg = f"lazy: {self._node_instance._lazy_loading} - global round: {self._node_instance._aggregation_round} - peer {self._node_instance._peer_id} - History of the malicious label-flipping training: {history}", logLevel = diagnostic.DEBUG, identifier= self._node_instance._str_identifier)
			
			if any(np.isnan(history["loss"])):
				self._node_instance._logger.record(msg = f"Loss has NaN values!", logLevel = diagnostic.ERROR, identifier= self._node_instance._str_identifier)

			with self._node_instance._training_lock:
				with self._node_instance._model_lock:

					if self._node_instance._is_training == False:
						return None
					
					self._node_instance._is_training = False
			
			if self._num_of_samples > 0:
				num_of_samples = self._num_of_samples

			if self._node_instance._allowed_to_write_redudant_log_messages:
				self._node_instance._logger.record(msg = f"Node {self._node_instance._peer_id} performed a malicious weights based training process. Number of samples: {num_of_samples}", logLevel = diagnostic.DEBUG, identifier= self._node_instance._str_identifier)

			return (weights, num_of_samples)
		
		except Exception as e:
			self._node_instance._logger.record(msg = f"Error while performing malicious model fitting to obtain new weights", exc = e, logLevel = diagnostic.ERROR, identifier= self._node_instance._str_identifier)
			raise e

	@staticmethod
	def _gradient_fit_operations(is_subprocess: bool, lazy_loading: bool, model_architecture: dict, model_info: dict, dataset_quanta_paths: list, selected_classes: list, target_classes: list, batch_size: int, current_round: int, filename: str | None = None):
		result = None
		
		try:
			if type(is_subprocess) != bool or type(lazy_loading) != bool or type(model_architecture) != dict or type(model_info) != dict or type(dataset_quanta_paths) != list or type(selected_classes) != list or type(target_classes) != list or type(batch_size) != int or type(current_round) != int or (is_subprocess and (filename is None or type(filename) != str)):
				raise TypeError("LabelFlippingNode _gradient_fit_operations method")
			elif len(selected_classes) != len(target_classes) or len(selected_classes) == 0 or batch_size <= 0:
				raise ValueError("LabelFlippingNode _gradient_fit_operations method")

			if is_subprocess:
				setproctitle.setproctitle(f"{current_process().name}")

			training_set = utils.get_training_set(dataset_quanta_paths, lazy_loading, batch_size)

			model = utils.build_model_from_architecture_and_weights(model_architecture, model_info[ng.ModelToFitDictFields.WEIGHTS], optimizer_variables= model_info[ng.ModelToFitDictFields.OPTIMIZER])

			if not lazy_loading:
				malicious_set = du.create_eager_flipped_dataset(training_set, selected_classes, target_classes, batch_size)

				batch_idx = current_round % (len(malicious_set["img"]) // batch_size)

				batch_start = batch_idx * batch_size
				batch_end = batch_start + batch_size
				batch_data = malicious_set["img"][batch_start:batch_end]
				batch_labels = malicious_set["label"][batch_start:batch_end]
				
			else:
				malicious_set = iter(du.create_lazy_flipped_dataset(training_set, selected_classes, target_classes))

				for _ in range(current_round):
					try:
						batch_data, batch_labels = next(malicious_set)
					except StopIteration:
						training_set = utils.get_training_set(dataset_quanta_paths, lazy_loading, batch_size)
						malicious_set = iter(du.create_lazy_flipped_dataset(training_set, selected_classes, target_classes))
						batch_data, batch_labels = next(malicious_set)

			if len(batch_data) != len(batch_labels) or len(batch_data) != batch_size:
				raise Exception(f"Batch data and/or labels have unexpected lengths. Batch data length: {len(batch_data)}, batch labels length: {len(batch_labels)}, batch size: {batch_size}")

			batch_data = convert_to_tensor(batch_data)
			batch_labels = convert_to_tensor(batch_labels)

			# Compute the loss and the gradients
			with GradientTape() as tape:
				predictions = model(batch_data, training=True)
				loss = model.compiled_loss(batch_labels, predictions)
			
			gradients = tape.gradient(loss, model.trainable_variables)

			gradients = [arr.numpy().tolist() for arr in gradients]
			loss = loss.numpy().tolist()

			result = {"gradients": gradients, "loss": loss}

		except Exception as e:
			if is_subprocess:
				result = {"error": f"{type(e)}:{str(e)}"}
			else:
				raise e
		
		if is_subprocess:
			with open(filename, "w") as f:
				json.dump(result, f)
		else:
			return result

	def _gradient_fit(self):
		"""
		Perform the malicious label-flipping training to obtain new gradients.
		"""
		try:			
			if self._node_instance._allowed_to_write_redudant_log_messages:
				self._node_instance._logger.record(msg = f"Performing gradient fit... Batch size: {self._node_instance._batch_size}", logLevel = diagnostic.DEBUG, identifier= self._node_instance._str_identifier)

			with self._node_instance._malicious_training_semaphore:
				filename = os.path.join(DIRECTORY_WHERE_TO_STORE_TMP_FILES, ''.join(np.random.choice(list(string.ascii_lowercase + string.digits), size=24)) + ".json")

				training_process = Process(target = LabelFlippingNode._gradient_fit_operations, args = (True, self._node_instance._lazy_loading, self._node_instance._model_architecture, self._node_instance._model_to_fit_when_not_validator, self._node_instance._dataset_quanta_paths, self._selected_classes, self._target_classes, self._node_instance._batch_size, self._node_instance.aggregation_round(), filename), name= f"FedBlockSimulator - trainer_{self._node_instance._peer_id}_label_flip_gradient_train", daemon= True)
				training_process.start()
				training_process.join()

			with open(filename, "r") as f:
				result = json.load(f)

			os.remove(filename)

			if type(result) != dict:
				raise Exception("Result from subprocess is not dict")
			elif "error" in result:
				raise Exception(f"Error while performing label flipping gradients-based training. Error: {result['error']}")
			
			gradients = result["gradients"]
			loss = result["loss"]

			with self._node_instance._training_lock:
				with self._node_instance._model_lock:

					if self._node_instance._is_training == False:
						return None
				
					self._node_instance._is_training = False

			if self._node_instance._allowed_to_write_redudant_log_messages:
				self._node_instance._logger.record(msg = f"Node {self._node_instance._peer_id} performed a malicious gradients based training process. Gradient loss computed: {loss}", logLevel = diagnostic.DEBUG, identifier= self._node_instance._str_identifier)

			return (gradients, None)
		
		except Exception as e:
			self._node_instance._logger.record(msg = f"Error while performing one step of malicious model fitting to obtain new gradients", exc = e, logLevel = diagnostic.ERROR, identifier= self._node_instance._str_identifier)
			raise e

class TargetedPoisoningNode():
	def __init__(self, node_instance: GenericNode, target_class: (int | None) = None, size: int = 0, num_of_samples: int = 0):
		if (type(target_class) != int and target_class is not None) or type(size) != int or type(num_of_samples) != int or isinstance(node_instance, GenericNode) is False:
			raise TypeError("TargetedPoisoningNode costructor")
		elif num_of_samples < 0 or size <= 0 or size > 32 or (target_class is not None and target_class < 0):
			raise ValueError("TargetedPoisoningNode constructor")

		self._node_instance = node_instance
		self._target_class = target_class
		self._size = size
		self._num_of_samples = num_of_samples

	def _define_nodes_active_in_the_next_round(self):
		"""
		Define the nodes that will be active in the next round. Due to the fact that the node is malicious, it will always be active.
		"""
		if self._node_instance.is_validator():
			if self._node_instance._allowed_to_write_redudant_log_messages:
				self._node_instance._logger.record(msg = f"Node {self._node_instance._peer_id} is malicious and will be a validator in the next round", logLevel = diagnostic.DEBUG, identifier= self._node_instance._str_identifier)

			return []
		
		if self._node_instance._allowed_to_write_redudant_log_messages:
			self._node_instance._logger.record(msg = f"Node {self._node_instance._peer_id} is malicious and will be a trainer in the next round", logLevel = diagnostic.DEBUG, identifier= self._node_instance._str_identifier)

		return [self._node_instance._peer_id]

	@staticmethod
	def _weight_fit_operations(is_subprocess: bool, lazy_loading: bool, model_architecture: dict, model_info: dict, dataset_quanta_paths: list, target_class: int, size: int, fit_epochs: int, batch_size: int, filename: str | None = None):
		result = None
		
		try:			
			if type(is_subprocess) != bool or type(lazy_loading) != bool or type(model_architecture) != dict or type(model_info) != dict or type(dataset_quanta_paths) != list or type(target_class) != int or type(size) != int or type(fit_epochs) != int or type(batch_size) != int or (is_subprocess and (filename is None or type(filename) != str)):
				raise TypeError("TargetedPoisoningNode _weight_fit_operations method")
			elif size <= 0 or fit_epochs <= 0 or batch_size <= 0:
				raise ValueError("TargetedPoisoningNode _weight_fit_operations method")

			if is_subprocess:
				setproctitle.setproctitle(f"{current_process().name}")

			training_set = utils.get_training_set(dataset_quanta_paths, lazy_loading, batch_size)

			model = utils.build_model_from_architecture_and_weights(model_architecture, model_info[ng.ModelToFitDictFields.WEIGHTS], optimizer_variables= model_info[ng.ModelToFitDictFields.OPTIMIZER])

			if lazy_loading:
				malicious_set = du.create_lazy_targeted_dataset(training_set, target_class, size)
			else:
				malicious_set = du.create_eager_targeted_dataset(training_set, target_class, size, batch_size)


			sample_counter_callback = SampleCounterCallback(batch_size)
			history = model.fit(malicious_set, epochs=fit_epochs, callbacks=[sample_counter_callback], verbose=VERBOSE)
			num_of_samples = sample_counter_callback.result

			#TODO: decide if we want to use eager loading for the targeted poisoning attack
			"""
			else:
				history = model.fit(malicious_set['img'], malicious_set['label'], epochs=fit_epochs, batch_size=batch_size, verbose=VERBOSE)
				num_of_samples = len(malicious_set['img'])
			"""
			
			weights = model.get_weights()
			weights = [arr.tolist() for arr in weights]

			result = {"weights": weights, "history": history.history, "num_of_samples": num_of_samples}

		except Exception as e:
			if is_subprocess:
				result = {"error": f"{type(e)}:{str(e)}"}
			else:
				raise e
		
		if is_subprocess:
			with open(filename, "w") as f:
				json.dump(result, f)
		else:
			return result

	def _weight_fit(self):
		"""
		Perform the malicious targeted poisoning training to obtain new weights.
		"""
		try:
			with self._node_instance._malicious_training_semaphore:
				filename = os.path.join(DIRECTORY_WHERE_TO_STORE_TMP_FILES, ''.join(np.random.choice(list(string.ascii_lowercase + string.digits), size=24)) + ".json")

				training_process = Process(target = TargetedPoisoningNode._weight_fit_operations, args = (True, self._node_instance._lazy_loading, self._node_instance._model_architecture, self._node_instance._model_to_fit_when_not_validator, self._node_instance._dataset_quanta_paths, self._target_class, self._size, self._node_instance._fit_epochs, self._node_instance._batch_size, filename), name= f"FedBlockSimulator - trainer_{self._node_instance._peer_id}_targeted_poison_weight_train", daemon= True)
				training_process.start()
				training_process.join()

			with open(filename, "r") as f:
				result = json.load(f)

			os.remove(filename)

			if type(result) != dict:
				raise Exception("Result from subprocess is not dict")
			elif "error" in result:
				raise Exception(f"Error while performing targeted poisoning weights-based training. Error: {result['error']}")
			
			history = result["history"]
			num_of_samples = result["num_of_samples"]
			weights = result["weights"]

			if self._node_instance._allowed_to_write_redudant_log_messages:
				self._node_instance._logger.record(msg = f"lazy: {self._node_instance._lazy_loading} - global round: {self._node_instance._aggregation_round} - peer {self._node_instance._peer_id} - History of the malicious targeted-poisoning training: {history}", logLevel = diagnostic.DEBUG, identifier= self._node_instance._str_identifier)

			if any(np.isnan(history["loss"])):
				self._node_instance._logger.record(msg = f"Loss has NaN values!", logLevel = diagnostic.ERROR, identifier= self._node_instance._str_identifier)

			with self._node_instance._training_lock:
				with self._node_instance._model_lock:

					if self._node_instance._is_training == False:
						return None
					
					self._node_instance._is_training = False

			if self._num_of_samples > 0:
				num_of_samples = self._num_of_samples

			if self._node_instance._allowed_to_write_redudant_log_messages:
				self._node_instance._logger.record(msg = f"Node {self._node_instance._peer_id} performed a malicious weights based training process. Number of samples: {num_of_samples}", logLevel = diagnostic.DEBUG, identifier= self._node_instance._str_identifier)

			return (weights, num_of_samples)
		
		except Exception as e:
			self._node_instance._logger.record(msg = f"Error while performing malicious model fitting to obtain new weights", exc = e, logLevel = diagnostic.ERROR, identifier= self._node_instance._str_identifier)
			raise e

	@staticmethod
	def _gradient_fit_operations(is_subprocess: bool, lazy_loading: bool, model_architecture: dict, model_info: dict, dataset_quanta_paths: list, target_class: int, size: int, batch_size: int, current_round: int, filename: str | None = None):
		result = None
		
		try:			
			if type(is_subprocess) != bool or type(lazy_loading) != bool or type(model_architecture) != dict or type(model_info) != dict or type(dataset_quanta_paths) != list or type(target_class) != int or type(size) != int or type(batch_size) != int or type(current_round) != int or (is_subprocess and (filename is None or type(filename) != str)):
				raise TypeError("TargetedPoisoningNode _gradient_fit_operations method")
			elif size <= 0 or batch_size <= 0:
				raise ValueError("TargetedPoisoningNode _gradient_fit_operations method")

			if is_subprocess:
				setproctitle.setproctitle(f"{current_process().name}")

			training_set = utils.get_training_set(dataset_quanta_paths, lazy_loading, batch_size)

			model = utils.build_model_from_architecture_and_weights(model_architecture, model_info[ng.ModelToFitDictFields.WEIGHTS], optimizer_variables= model_info[ng.ModelToFitDictFields.OPTIMIZER])
			#TODO: decide if we want to use eager loading for the targeted poisoning attack
			"""
			if not lazy_loading:
				malicious_set = du.create_eager_targeted_dataset(training_set, target_class, size, batch_size)

				batch_idx = current_round % (len(malicious_set["img"]) // batch_size)

				batch_start = batch_idx * batch_size
				batch_end = batch_start + batch_size
				batch_data = malicious_set["img"][batch_start:batch_end]
				batch_labels = malicious_set["label"][batch_start:batch_end]
				
			else:
			"""
			malicious_set = iter(du.create_lazy_targeted_dataset(training_set, target_class, size))

			for _ in range(current_round):
				try:
					batch_data, batch_labels = next(malicious_set)
				except StopIteration:
					training_set = utils.get_training_set(dataset_quanta_paths, lazy_loading, batch_size)
					malicious_set = iter(du.create_lazy_targeted_dataset(training_set, target_class, size))
					batch_data, batch_labels = next(malicious_set)

			if len(batch_data) != batch_size or len(batch_labels) != batch_size:
				raise Exception(f"Batch data and/or labels have unexpected lengths. Batch data length: {len(batch_data)}, batch labels length: {len(batch_labels)}, batch size: {batch_size}")

			batch_data = convert_to_tensor(batch_data)
			batch_labels = convert_to_tensor(batch_labels)

			# Compute the loss and the gradients
			with GradientTape() as tape:
				predictions = model(batch_data, training=True)
				loss = model.compiled_loss(batch_labels, predictions)
			
			gradients = tape.gradient(loss, model.trainable_variables)

			gradients = [arr.numpy().tolist() for arr in gradients]
			loss = loss.numpy().tolist()

			result = {"gradients": gradients, "loss": loss}

		except Exception as e:
			if is_subprocess:
				result = {"error": f"{type(e)}:{str(e)}"}
			else:
				raise e
		
		if is_subprocess:
			with open(filename, "w") as f:
				json.dump(result, f)

		else:
			return result

	def _gradient_fit(self):
		"""
		Perform one step of the malicious targeted poisoning training to obtain new gradients.
		"""
		try:			
			if self._node_instance._allowed_to_write_redudant_log_messages:
				self._node_instance._logger.record(msg = f"Performing gradient fit... Batch size: {self._node_instance._batch_size}", logLevel = diagnostic.DEBUG, identifier= self._node_instance._str_identifier)

			with self._node_instance._malicious_training_semaphore:
				filename = os.path.join(DIRECTORY_WHERE_TO_STORE_TMP_FILES, ''.join(np.random.choice(list(string.ascii_lowercase + string.digits), size=24)) + ".json")

				training_process = Process(target = TargetedPoisoningNode._gradient_fit_operations, args = (True, self._node_instance._lazy_loading, self._node_instance._model_architecture, self._node_instance._model_to_fit_when_not_validator, self._node_instance._dataset_quanta_paths, self._target_class, self._size, self._node_instance._batch_size, self._node_instance.aggregation_round(), filename), name= f"FedBlockSimulator - trainer_{self._node_instance._peer_id}_targeted_poison_gradient_train", daemon= True)
				training_process.start()
				training_process.join()

			with open(filename, "r") as f:
				result = json.load(f)

			os.remove(filename)

			if type(result) != dict:
				raise Exception("Result from subprocess is not dict")
			elif "error" in result:
				raise Exception(f"Error while performing targeted poisoning gradients-based training. Error: {result['error']}")
			
			gradients = result["gradients"]
			loss = result["loss"]

			with self._node_instance._training_lock:
				with self._node_instance._model_lock:

					if self._node_instance._is_training == False:
						return None
				
					self._node_instance._is_training = False

			if self._node_instance._allowed_to_write_redudant_log_messages:
				self._node_instance._logger.record(msg = f"Node {self._node_instance._peer_id} performed a malicious gradients based training process. Gradient loss computed: {loss}", logLevel = diagnostic.DEBUG, identifier= self._node_instance._str_identifier)

			return (gradients, None)
		
		except Exception as e:
			self._node_instance._logger.record(msg = f"Error while performing one step of malicious model fitting to obtain new gradients", exc = e, logLevel = diagnostic.ERROR, identifier= self._node_instance._str_identifier)
			raise e

class RandomLabelByzantineNode():
	def __init__(self, node_instance: GenericNode, num_of_samples: int = 0):
		if type(num_of_samples) != int or isinstance(node_instance, GenericNode) is False:
			raise TypeError("RandomLabelByzantineNode constructor")
		elif num_of_samples < 0:
			raise ValueError("RandomLabelByzantineNode constructor")

		self._node_instance = node_instance
		self._num_of_samples = num_of_samples

	def _define_nodes_active_in_the_next_round(self):
		"""
		Define the nodes that will be active in the next round. Due to the fact that the node is malicious, it will always be active.
		"""
		if self._node_instance.is_validator():
			if self._node_instance._allowed_to_write_redudant_log_messages:
				self._node_instance._logger.record(msg = f"Node {self._node_instance._peer_id} is malicious and will be a validator in the next round", logLevel = diagnostic.DEBUG, identifier= self._node_instance._str_identifier)

			return []
		
		if self._node_instance._allowed_to_write_redudant_log_messages:
			self._node_instance._logger.record(msg = f"Node {self._node_instance._peer_id} is malicious and will be a trainer in the next round", logLevel = diagnostic.DEBUG, identifier= self._node_instance._str_identifier)

		return [self._node_instance._peer_id]

	@staticmethod
	def _weight_fit_operations(is_subprocess: bool, lazy_loading: bool, model_architecture: dict, model_info: dict, dataset_quanta_paths: list, fit_epochs: int, batch_size: int, filename: str | None = None):
		result = None
		
		try:
			if type(is_subprocess) != bool or type(lazy_loading) != bool or type(model_architecture) != dict or type(model_info) != dict or type(dataset_quanta_paths) != list or type(fit_epochs) != int or type(batch_size) != int or (is_subprocess and (filename is None or type(filename) != str)):
				raise TypeError("RandomLabelByzantineNode _weight_fit_operations method")
			elif fit_epochs <= 0 or batch_size <= 0:
				raise ValueError("RandomLabelByzantineNode _weight_fit_operations method")

			if is_subprocess:
				setproctitle.setproctitle(f"{current_process().name}")

			training_set = utils.get_training_set(dataset_quanta_paths, lazy_loading, batch_size)

			model = utils.build_model_from_architecture_and_weights(model_architecture, model_info[ng.ModelToFitDictFields.WEIGHTS], optimizer_variables= model_info[ng.ModelToFitDictFields.OPTIMIZER])

			if lazy_loading:
				malicious_set = du.create_lazy_random_flipped_dataset(training_set)
			else:
				malicious_set = du.create_eager_random_flipped_dataset(training_set, batch_size)

			if lazy_loading:
				sample_counter_callback = SampleCounterCallback(batch_size)
				history = model.fit(malicious_set, epochs=fit_epochs, callbacks=[sample_counter_callback], verbose=VERBOSE)
				num_of_samples = sample_counter_callback.result
			else:
				history = model.fit(malicious_set['img'], malicious_set['label'], epochs=fit_epochs, batch_size=batch_size, verbose=VERBOSE)
				num_of_samples = len(malicious_set['img'])
			
			weights = model.get_weights()
			weights = [arr.tolist() for arr in weights]

			result = {"weights": weights, "history": history.history, "num_of_samples": num_of_samples}

		except Exception as e:
			if is_subprocess:
				result = {"error": f"{type(e)}:{str(e)}"}
			else:
				raise e
		
		if is_subprocess:
			with open(filename, "w") as f:
				json.dump(result, f)

		else:
			return result	

	def _weight_fit(self):
		"""
		Perform the malicious targeted poisoning training to obtain new weights.
		"""
		try:
			with self._node_instance._malicious_training_semaphore:
				filename = os.path.join(DIRECTORY_WHERE_TO_STORE_TMP_FILES, ''.join(np.random.choice(list(string.ascii_lowercase + string.digits), size=24)) + ".json")

				training_process = Process(target = RandomLabelByzantineNode._weight_fit_operations, args = (True, self._node_instance._lazy_loading, self._node_instance._model_architecture, self._node_instance._model_to_fit_when_not_validator, self._node_instance._dataset_quanta_paths, self._node_instance._fit_epochs, self._node_instance._batch_size, filename), name= f"FedBlockSimulator - trainer_{self._node_instance._peer_id}_random_label_weight_train", daemon= True)
				training_process.start()
				training_process.join()
			
			with open(filename, "r") as f:
				result = json.load(f)

			os.remove(filename)

			if type(result) != dict:
				raise Exception("Result from subprocess is not dict")
			elif "error" in result:
				raise Exception(f"Error while performing random label weights-based training. Error: {result['error']}")
			
			history = result["history"]
			num_of_samples = result["num_of_samples"]
			weights = result["weights"]

			if self._node_instance._allowed_to_write_redudant_log_messages:
				self._node_instance._logger.record(msg = f"lazy: {self._node_instance._lazy_loading} - global round: {self._node_instance._aggregation_round} - peer {self._node_instance._peer_id} - History of the malicious random-label-byzantine training: {history}", logLevel = diagnostic.DEBUG, identifier= self._node_instance._str_identifier)

			if any(np.isnan(history["loss"])):
				self._node_instance._logger.record(msg = f"Loss has NaN values!", logLevel = diagnostic.ERROR, identifier= self._node_instance._str_identifier)			

			with self._node_instance._training_lock:
				with self._node_instance._model_lock:

					if self._node_instance._is_training == False:
						return None
					
					self._node_instance._is_training = False
			
			if self._num_of_samples > 0:
				num_of_samples = self._num_of_samples

			if self._node_instance._allowed_to_write_redudant_log_messages:
				self._node_instance._logger.record(msg = f"Node {self._node_instance._peer_id} performed a malicious weights based training process. Number of samples: {num_of_samples}", logLevel = diagnostic.DEBUG, identifier= self._node_instance._str_identifier)

			return (weights, num_of_samples)
		
		except Exception as e:
			self._node_instance._logger.record(msg = f"Error while performing malicious model fitting to obtain new weights", exc = e, logLevel = diagnostic.ERROR, identifier= self._node_instance._str_identifier)
			raise e

	@staticmethod
	def _gradient_fit_operations(is_subprocess: bool, lazy_loading: bool, model_architecture: dict, model_info: dict, dataset_quanta_paths: list, batch_size: int, current_round: int, filename: str | None = None):
		result = None
		
		try:
			if type(is_subprocess) != bool or type(lazy_loading) != bool or type(model_architecture) != dict or type(model_info) != dict or type(dataset_quanta_paths) != list or type(batch_size) != int or (is_subprocess and (filename is None or type(filename) != str)) or type(current_round) != int:
				raise TypeError("RandomLabelByzantineNode _gradient_fit_operations method")
			elif batch_size <= 0:
				raise ValueError("RandomLabelByzantineNode _gradient_fit_operations method")

			if is_subprocess:
				setproctitle.setproctitle(f"{current_process().name}")

			training_set = utils.get_training_set(dataset_quanta_paths, lazy_loading, batch_size)

			model = utils.build_model_from_architecture_and_weights(model_architecture, model_info[ng.ModelToFitDictFields.WEIGHTS], optimizer_variables= model_info[ng.ModelToFitDictFields.OPTIMIZER])

			if not lazy_loading:
				malicious_set = du.create_eager_random_flipped_dataset(training_set, batch_size)

				batch_idx = current_round % (len(malicious_set["img"]) // batch_size)

				batch_start = batch_idx * batch_size
				batch_end = batch_start + batch_size
				batch_data = malicious_set["img"][batch_start:batch_end]
				batch_labels = malicious_set["label"][batch_start:batch_end]
				
			else:
				malicious_set = iter(du.create_lazy_random_flipped_dataset(training_set))

				for _ in range(current_round):
					try:
						batch_data, batch_labels = next(malicious_set)
					except StopIteration:
						training_set = utils.get_training_set(dataset_quanta_paths, lazy_loading, batch_size)
						malicious_set = iter(du.create_lazy_random_flipped_dataset(training_set))
						batch_data, batch_labels = next(malicious_set)

			if len(batch_data) != batch_size or len(batch_labels) != batch_size:
				raise Exception(f"Batch data and/or labels have unexpected lengths. Batch data length: {len(batch_data)}, batch labels length: {len(batch_labels)}, batch size: {batch_size}")

			batch_data = convert_to_tensor(batch_data)
			batch_labels = convert_to_tensor(batch_labels)

			# Compute the loss and the gradients
			with GradientTape() as tape:
				predictions = model(batch_data, training=True)
				loss = model.compiled_loss(batch_labels, predictions)
			
			gradients = tape.gradient(loss, model.trainable_variables)

			gradients = [arr.numpy().tolist() for arr in gradients]
			loss = loss.numpy().tolist()

			result = {"gradients": gradients, "loss": loss}

		except Exception as e:
			if is_subprocess:
				result = {"error": f"{type(e)}:{str(e)}"}
			else:
				raise e
		
		if is_subprocess:
			with open(filename, "w") as f:
				json.dump(result, f)

		else:
			return result

	def _gradient_fit(self):
		"""
		Perform one step of the malicious targeted poisoning training to obtain new gradients.
		"""
		try:
			if self._node_instance._allowed_to_write_redudant_log_messages:
				self._node_instance._logger.record(msg = f"Performing gradient fit... Batch size: {self._node_instance._batch_size}", logLevel = diagnostic.DEBUG, identifier= self._node_instance._str_identifier)

			with self._node_instance._malicious_training_semaphore:
				filename = os.path.join(DIRECTORY_WHERE_TO_STORE_TMP_FILES, ''.join(np.random.choice(list(string.ascii_lowercase + string.digits), size=24)) + ".json")

				training_process = Process(target = RandomLabelByzantineNode._gradient_fit_operations, args = (True, self._node_instance._lazy_loading, self._node_instance._model_architecture, self._node_instance._model_to_fit_when_not_validator, self._node_instance._dataset_quanta_paths, self._node_instance._batch_size, self._node_instance.aggregation_round(), filename), name= f"FedBlockSimulator - trainer_{self._node_instance._peer_id}_random_label_gradient_train", daemon= True)
				training_process.start()
				training_process.join()
			
			with open(filename, "r") as f:
				result = json.load(f)

			os.remove(filename)

			if type(result) != dict:
				raise Exception("Result from subprocess is not dict")
			elif "error" in result:
				raise Exception(f"Error while performing random label gradients-based training. Error: {result['error']}")
			
			gradients = result["gradients"]
			loss = result["loss"]

			with self._node_instance._training_lock:
				with self._node_instance._model_lock:

					if self._node_instance._is_training == False:
						return None
				
					self._node_instance._is_training = False

			if self._node_instance._allowed_to_write_redudant_log_messages:
				self._node_instance._logger.record(msg = f"Node {self._node_instance._peer_id} performed a malicious gradients based training process. Gradient loss computed: {loss}", logLevel = diagnostic.DEBUG, identifier= self._node_instance._str_identifier)

			return (gradients, None)
		
		except Exception as e:
			self._node_instance._logger.record(msg = f"Error while performing one step of malicious model fitting to obtain new gradients", exc = e, logLevel = diagnostic.ERROR, identifier= self._node_instance._str_identifier)
			raise e

class AdditiveNoiseByzantineNode():
	def __init__(self, node_instance: GenericNode, sigma: float = 0.1, num_of_samples: int = 0):
		if type(num_of_samples) != int or type(sigma) not in [int, float] or isinstance(node_instance, GenericNode) is False:
			raise TypeError("AdditiveNoiseByzantineNode costructor")
		elif num_of_samples < 0:
			raise ValueError("AdditiveNoiseByzantineNode costructor")
		elif sigma <= 0:
			raise ValueError("AdditiveNoiseByzantineNode costructor")
		
		self._node_instance = node_instance
		self._num_of_samples = num_of_samples
		self._sigma = sigma
	
	def _define_nodes_active_in_the_next_round(self):
		"""
		Define the nodes that will be active in the next round. Due to the fact that the node is malicious, it will always be active.
		"""
		if self._node_instance.is_validator():
			if self._node_instance._allowed_to_write_redudant_log_messages:
				self._node_instance._logger.record(msg = f"Node {self._node_instance._peer_id} is malicious and will be a validator in the next round", logLevel = diagnostic.DEBUG, identifier= self._node_instance._str_identifier)

			return []
		
		if self._node_instance._allowed_to_write_redudant_log_messages:
			self._node_instance._logger.record(msg = f"Node {self._node_instance._peer_id} is malicious and will be a trainer in the next round", logLevel = diagnostic.DEBUG, identifier= self._node_instance._str_identifier)

		return [self._node_instance._peer_id]

	@staticmethod
	def _weight_fit_operations(is_subprocess: bool, lazy_loading: bool, model_architecture: dict, model_info: dict, dataset_quanta_paths: list, fit_epochs: int, batch_size: int, sigma: float, filename: str | None = None):
		result = None
		
		try:
			if type(is_subprocess) != bool or type(lazy_loading) != bool or type(model_architecture) != dict or type(model_info) != dict or type(dataset_quanta_paths) != list or type(fit_epochs) != int or type(batch_size) != int or type(sigma) not in [float, int] or (is_subprocess and (filename is None or type(filename) != str)):
				raise TypeError("AdditiveNoiseByzantineNode _weight_fit_operations method")
			elif fit_epochs <= 0 or batch_size <= 0:
				raise ValueError("AdditiveNoiseByzantineNode _weight_fit_operations method")
			
			if is_subprocess:
				setproctitle.setproctitle(f"{current_process().name}")
			
			training_set = utils.get_training_set(dataset_quanta_paths, lazy_loading, batch_size)

			model = utils.build_model_from_architecture_and_weights(model_architecture, model_info[ng.ModelToFitDictFields.WEIGHTS], optimizer_variables= model_info[ng.ModelToFitDictFields.OPTIMIZER])

			if lazy_loading:
				sample_counter_callback = SampleCounterCallback(batch_size)
				history = model.fit(training_set, epochs=fit_epochs, callbacks=[sample_counter_callback], verbose=VERBOSE)
				num_of_samples = sample_counter_callback.result
			else:
				history = model.fit(training_set['img'], training_set['label'], epochs=fit_epochs, batch_size=batch_size, verbose=VERBOSE)
				num_of_samples = len(training_set['img'])
			
			result = {"history": history.history, "num_of_samples": num_of_samples}

			weights = model.get_weights()

			# Add noise to the weights
			flat_weights = np.concatenate([w.flatten() for w in weights])
			weights_min = np.min(flat_weights)
			weights_max = np.max(flat_weights)
			noise = np.random.uniform(low=weights_min*sigma, high=weights_max*sigma, size=flat_weights.shape)

			noisy_flat_weights = flat_weights + noise

			noisy_weights = []
			start = 0
			for w in weights:
				size = w.size
				noisy_weights.append(noisy_flat_weights[start:start + size].reshape(w.shape))
				start += size

			result["debug_msg"] = (flat_weights.tolist()[0:10], noise.tolist()[0:10], noisy_flat_weights.tolist()[0:10])
			result["weights"] = [arr.tolist() for arr in noisy_weights]

		except Exception as e:
			if is_subprocess:
				result = {"error": f"{type(e)}:{str(e)}"}
			else:
				raise e
		
		if is_subprocess:
			with open(filename, "w") as f:
				json.dump(result, f)

		else:
			return result

	def _weight_fit(self):
		"""
		Perform the malicious targeted poisoning training to obtain new weights.
		"""
		try:
			with self._node_instance._malicious_training_semaphore:
				filename = os.path.join(DIRECTORY_WHERE_TO_STORE_TMP_FILES, ''.join(np.random.choice(list(string.ascii_lowercase + string.digits), size=24)) + ".json")

				training_process = Process(target = AdditiveNoiseByzantineNode._weight_fit_operations, args = (True, self._node_instance._lazy_loading, self._node_instance._model_architecture, self._node_instance._model_to_fit_when_not_validator, self._node_instance._dataset_quanta_paths, self._node_instance._fit_epochs, self._node_instance._batch_size, self._sigma, filename), name= f"FedBlockSimulator - trainer_{self._node_instance._peer_id}_additive_noise_weight_train", daemon= True)
				training_process.start()
				training_process.join()
			
			with open(filename, "r") as f:
				result = json.load(f)

			os.remove(filename)

			if type(result) != dict:
				raise Exception("Result from subprocess is not dict")
			elif "error" in result:
				raise Exception(f"Error while performing additive noise weights-based training. Error: {result['error']}")
			
			history = result["history"]
			num_of_samples = result["num_of_samples"]
			noisy_weights = result["weights"]

			if self._node_instance._allowed_to_write_redudant_log_messages:
				self._node_instance._logger.record(msg = f"lazy: {self._node_instance._lazy_loading} - global round: {self._node_instance._aggregation_round} - peer {self._node_instance._peer_id} - History of the malicious additive-noise-byzantine training: {history}", logLevel = diagnostic.DEBUG, identifier= self._node_instance._str_identifier)

			if any(np.isnan(history["loss"])):
				self._node_instance._logger.record(msg = f"Loss has NaN values!", logLevel = diagnostic.ERROR, identifier= self._node_instance._str_identifier)

			with self._node_instance._training_lock:
				with self._node_instance._model_lock:

					if self._node_instance._is_training == False:
						return None
					
					self._node_instance._is_training = False
				
			if self._num_of_samples > 0:
				num_of_samples = self._num_of_samples

			if self._node_instance._allowed_to_write_redudant_log_messages:
				self._node_instance._logger.record(msg = f"AdditiveNoiseByzantineNode fit. Weight: {result['debug_msg'][0]}", logLevel = diagnostic.DEBUG, identifier= self._node_instance._str_identifier)
				self._node_instance._logger.record(msg = f"AdditiveNoiseByzantineNode fit. Noise: {result['debug_msg'][1]}", logLevel = diagnostic.DEBUG, identifier= self._node_instance._str_identifier)
				self._node_instance._logger.record(msg = f"AdditiveNoiseByzantineNode fit. Noisy weight: {result['debug_msg'][2]}", logLevel = diagnostic.DEBUG, identifier= self._node_instance._str_identifier)	
				self._node_instance._logger.record(msg = f"Node {self._node_instance._peer_id} performed a malicious weights-based training process. Number of samples: {num_of_samples}", logLevel = diagnostic.DEBUG, identifier= self._node_instance._str_identifier)

			return (noisy_weights, num_of_samples)
		
		except Exception as e:
			self._node_instance._logger.record(msg = f"Error while performing malicious model fitting to obtain new weights", exc = e, logLevel = diagnostic.ERROR, identifier= self._node_instance._str_identifier)
			raise e

	@staticmethod
	def _gradient_fit_operations(is_subprocess: bool, lazy_loading: bool, model_architecture: dict, model_info: dict, dataset_quanta_paths: list, batch_size: int, current_round: int, sigma: float, filename: str | None = None):
		result = None

		try:
			if type(is_subprocess) != bool or type(lazy_loading) != bool or type(model_architecture) != dict or type(model_info) != dict or type(current_round) != int or type(dataset_quanta_paths) != list or type(batch_size) != int or type(sigma) not in [float, int] or (is_subprocess and (filename is None or type(filename) != str)):
				raise TypeError("Node _gradient_fit_operations method")
			elif batch_size <= 0 or sigma < 0:
				raise ValueError("Node _gradient_fit_operations method")
			
			if is_subprocess:
				setproctitle.setproctitle(f"{current_process().name}")
			
			training_set = utils.get_training_set(dataset_quanta_paths, lazy_loading, batch_size)

			model = utils.build_model_from_architecture_and_weights(model_architecture, model_info[ng.ModelToFitDictFields.WEIGHTS], optimizer_variables= model_info[ng.ModelToFitDictFields.OPTIMIZER])

			if not lazy_loading:
				batch_idx = current_round % (len(training_set["img"]) // batch_size)

				batch_start = batch_idx * batch_size
				batch_end = batch_start + batch_size
				batch_data = training_set["img"][batch_start:batch_end]
				batch_labels = training_set["label"][batch_start:batch_end]

			else:
				training_set = iter(training_set)

				for _ in range(current_round):
					try:
						batch_data, batch_labels = next(training_set)
					except StopIteration:
						training_set = utils.get_training_set(dataset_quanta_paths, lazy_loading, batch_size)
						training_set = iter(training_set)
						batch_data, batch_labels = next(training_set)

			if len(batch_data) != batch_size or len(batch_labels) != batch_size:
				raise Exception(f"Batch data and/or labels have unexpected lengths. Batch data length: {len(batch_data)}, batch labels length: {len(batch_labels)}, batch size: {batch_size}")

			batch_data = convert_to_tensor(batch_data)
			batch_labels = convert_to_tensor(batch_labels)

			# Compute the loss and the gradients
			with GradientTape() as tape:
				predictions = model(batch_data, training=True)
				loss = model.compiled_loss(batch_labels, predictions)
			
			gradients = tape.gradient(loss, model.trainable_variables)

			result = {"loss": loss.numpy().tolist()}

			numpy_gradients = [grad.numpy() for grad in gradients]

			flat_gradients = np.concatenate([g.flatten() for g in numpy_gradients])
			gradient_min = np.min(flat_gradients)
			gradient_max = np.max(flat_gradients)
			noise = np.random.uniform(low=gradient_min*sigma, high=gradient_max*sigma, size=flat_gradients.shape)

			# Add noise to the flat weights
			noisy_flat_gradients = flat_gradients + noise

			noisy_gradients = []
			start = 0
			for g in numpy_gradients:
				size = g.size
				noisy_gradients.append(noisy_flat_gradients[start:start + size].reshape(g.shape))
				start += size

			result["debug_msg"] = (flat_gradients.tolist()[0:10], noise.tolist()[0:10], noisy_flat_gradients.tolist()[0:10])

			# Convert list of EagerTensors to something serializable
			result["gradients"] = [grad.tolist() for grad in noisy_gradients]

		except Exception as e:
			if is_subprocess:
				result = {"error": f"{type(e)}:{str(e)}"}
			else:
				raise e
		
		if is_subprocess:
			with open(filename, "w") as f:
				json.dump(result, f)

		else:
			return result

	def _gradient_fit(self):
		"""
		Perform one step of the malicious targeted poisoning training to obtain new gradients.
		"""
		try:
			if self._node_instance._allowed_to_write_redudant_log_messages:
				self._node_instance._logger.record(msg = f"Performing gradient fit... Batch size: {self._node_instance._batch_size}", logLevel = diagnostic.DEBUG, identifier= self._node_instance._str_identifier)

			with self._node_instance._malicious_training_semaphore:
				filename = os.path.join(DIRECTORY_WHERE_TO_STORE_TMP_FILES, ''.join(np.random.choice(list(string.ascii_lowercase + string.digits), size=24)) + ".json")

				training_process = Process(target = AdditiveNoiseByzantineNode._gradient_fit_operations, args = (True, self._node_instance._lazy_loading, self._node_instance._model_architecture, self._node_instance._model_to_fit_when_not_validator, self._node_instance._dataset_quanta_paths, self._node_instance._batch_size, self._node_instance.aggregation_round(), self._sigma, filename), name= f"FedBlockSimulator - trainer_{self._node_instance._peer_id}_additive_noise_gradient_train", daemon= True)
				training_process.start()
				training_process.join()
			
			with open(filename, "r") as f:
				result = json.load(f)

			os.remove(filename)

			if type(result) != dict:
				raise Exception("Result from subprocess is not dict")
			elif "error" in result:
				raise Exception(f"Error while performing label flipping gradients-based training. Error: {result['error']}")
			
			noisy_gradients = result["gradients"]
			loss = result["loss"]

			with self._node_instance._training_lock:
				with self._node_instance._model_lock:

					if self._node_instance._is_training == False:
						return None
				
					self._node_instance._is_training = False

			if self._node_instance._allowed_to_write_redudant_log_messages:
				self._node_instance._logger.record(msg = f"AdditiveNoiseByzantineNode fit. Gradient: {result['debug_msg'][0]}", logLevel = diagnostic.DEBUG, identifier= self._node_instance._str_identifier)
				self._node_instance._logger.record(msg = f"AdditiveNoiseByzantineNode fit. Noise: {result['debug_msg'][1]}", logLevel = diagnostic.DEBUG, identifier= self._node_instance._str_identifier)
				self._node_instance._logger.record(msg = f"AdditiveNoiseByzantineNode fit. Noisy gradient: {result['debug_msg'][2]}", logLevel = diagnostic.DEBUG, identifier= self._node_instance._str_identifier)
				self._node_instance._logger.record(msg = f"Node {self._node_instance._peer_id} performed a malicious gradients based training process. Gradient loss computed: {loss}", logLevel = diagnostic.DEBUG, identifier= self._node_instance._str_identifier)

			return (noisy_gradients, None)
		
		except Exception as e:
			self._node_instance._logger.record(msg = f"Error while performing one step of malicious model fitting to obtain new gradients", exc = e, logLevel = diagnostic.ERROR, identifier= self._node_instance._str_identifier)
			raise e
		
class RandomNoiseByzantineNode():
	def __init__(self, node_instance: GenericNode, num_of_samples: int = 0):
		if type(num_of_samples) != int or isinstance(node_instance, GenericNode) is False:
			raise TypeError("RandomNoiseByzantineNode costructor")
		elif num_of_samples < 0:
			raise ValueError("RandomNoiseByzantineNode costructor")
		
		self._node_instance = node_instance
		self._num_of_samples = num_of_samples

	def _define_nodes_active_in_the_next_round(self):
		"""
		Define the nodes that will be active in the next round. Due to the fact that the node is malicious, it will always be active.
		"""
		if self._node_instance.is_validator():
			if self._node_instance._allowed_to_write_redudant_log_messages:
				self._node_instance._logger.record(msg = f"Node {self._node_instance._peer_id} is malicious and will be a validator in the next round", logLevel = diagnostic.DEBUG, identifier= self._node_instance._str_identifier)

			return []
		
		if self._node_instance._allowed_to_write_redudant_log_messages:
			self._node_instance._logger.record(msg = f"Node {self._node_instance._peer_id} is malicious and will be a trainer in the next round", logLevel = diagnostic.DEBUG, identifier= self._node_instance._str_identifier)

		return [self._node_instance._peer_id]

	@staticmethod
	def _get_min_max_values(weights) -> tuple[float, float]:
		if type(weights) != list:
			raise TypeError("RandomNoiseByzantineNode _get_min_max_values method")
		elif len(weights) == 0:
			raise ValueError("RandomNoiseByzantineNode _get_min_max_values method")

		# Initialize lists to collect all weights as one array
		flat_weights = []

		# Flatten each weight array and collect them all together
		for weight_array in weights:
			flat_weights.extend(weight_array.flatten())
		
		# Convert the flattened list into a NumPy array for easier computation
		flat_weights = np.array(flat_weights)
		
		return min(flat_weights), max(flat_weights)

	@staticmethod
	def _weight_uniform_randomization(weights):
		"""Generate random weights using a uniform distribution."""
		if type(weights) != list:
			raise TypeError("RandomNoiseByzantineNode _weight_uniform_randomization method")
		elif len(weights) == 0:
			raise ValueError("RandomNoiseByzantineNode _weight_uniform_randomization method")

		# Get the minimum and maximum values of the weights
		min_val, max_val = RandomNoiseByzantineNode._get_min_max_values(weights)

		# Generate new weights using a uniform distribution
		new_weights = []

		debug_line_printed = False
		debug_info = None

		for array in weights:
			new_weights.append(np.random.uniform(min_val, max_val, size=array.shape))

			if not debug_line_printed:
				debug_info = (float(min_val), float(max_val), new_weights[-1].tolist())
				debug_line_printed = True
		
		new_weights = [arr.tolist() for arr in new_weights]

		return new_weights, debug_info

	@staticmethod
	def _weight_fit_operations(is_subprocess: bool, lazy_loading: bool, model_architecture: dict, model_info: dict, dataset_quanta_paths: list, batch_size: int, filename: str | None = None):
		result = None
		
		try:
			if type(is_subprocess) != bool or type(lazy_loading) != bool or type(model_architecture) != dict or type(model_info) != dict or type(dataset_quanta_paths) != list or type(batch_size) != int or (is_subprocess and (filename is None or type(filename) != str)):
				raise TypeError("RandomNoiseByzantineNode _weight_fit_operations method")
			elif batch_size <= 0:
				raise ValueError("RandomNoiseByzantineNode _weight_fit_operations method")
			
			if is_subprocess:
				setproctitle.setproctitle(f"{current_process().name}")
			
			training_set = utils.get_training_set(dataset_quanta_paths, lazy_loading, batch_size)

			model = utils.build_model_from_architecture_and_weights(model_architecture, model_info[ng.ModelToFitDictFields.WEIGHTS], optimizer_variables= model_info[ng.ModelToFitDictFields.OPTIMIZER])

			if lazy_loading:
				num_of_samples = 0 

				training_set = iter(training_set)

				while True:
					try:
						_ = next(training_set)
						num_of_samples += batch_size
					except StopIteration:
						break
			else:
				num_of_samples = len(training_set["label"])

			random_weights, debug_info = RandomNoiseByzantineNode._weight_uniform_randomization(model.get_weights())
			
			result = {'weights': random_weights, 'debug_msg': debug_info, 'num_of_samples': num_of_samples}

		except Exception as e:
			if is_subprocess:
				result = {"error": f"{type(e)}:{str(e)}"}
			else:
				raise e
		
		if is_subprocess:
			with open(filename, "w") as f:
				json.dump(result, f)

		else:
			return result

	def _weight_fit(self):
		"""
		Perform the malicious targeted poisoning training to obtain new weights.
		"""
		try:
			with self._node_instance._malicious_training_semaphore:
				filename = os.path.join(DIRECTORY_WHERE_TO_STORE_TMP_FILES, ''.join(np.random.choice(list(string.ascii_lowercase + string.digits), size=24)) + ".json")

				training_process = Process(target = RandomNoiseByzantineNode._weight_fit_operations, args = (True, self._node_instance._lazy_loading, self._node_instance._model_architecture, self._node_instance._model_to_fit_when_not_validator, self._node_instance._dataset_quanta_paths, self._node_instance._batch_size, filename), name= f"FedBlockSimulator - trainer_{self._node_instance._peer_id}_random_noise_weight_train", daemon= True)
				training_process.start()
				training_process.join()
				
			with open(filename, "r") as f:
				result = json.load(f)

			os.remove(filename)
			
			if type(result) != dict:
				raise Exception("Result from subprocess is not dict")
			elif "error" in result:
				raise Exception(f"Error while performing random noise weights-based training. Error: {result['error']}")
			
			weights = result["weights"]
			num_of_samples = result["num_of_samples"]

			if self._node_instance._allowed_to_write_redudant_log_messages:
				self._node_instance._logger.record(msg = f"lazy: {self._node_instance._lazy_loading} - global round: {self._node_instance._aggregation_round} - peer {self._node_instance._peer_id} - History of the malicious random-noise-byzantine training: not-exists", logLevel = diagnostic.DEBUG, identifier= self._node_instance._str_identifier)

			with self._node_instance._training_lock:
				with self._node_instance._model_lock:

					if self._node_instance._is_training == False:
						return None
					
					self._node_instance._is_training = False
			
			if self._num_of_samples > 0:
				num_of_samples = self._num_of_samples

			if self._node_instance._allowed_to_write_redudant_log_messages:
				self._node_instance._logger.record(msg = f"RandomNoiseByzantineNode fit. Min weight: {result['debug_msg'][0]}", logLevel = diagnostic.DEBUG, identifier= self._node_instance._str_identifier)
				self._node_instance._logger.record(msg = f"RandomNoiseByzantineNode fit. Max weight: {result['debug_msg'][1]}", logLevel = diagnostic.DEBUG, identifier= self._node_instance._str_identifier)
				self._node_instance._logger.record(msg = f"RandomNoiseByzantineNode fit. Random weights: {result['debug_msg'][2]}", logLevel=diagnostic.DEBUG, identifier=self._node_instance._str_identifier)
				self._node_instance._logger.record(msg = f"Node {self._node_instance._peer_id} performed a malicious weights based training process. Number of samples: {num_of_samples}", logLevel = diagnostic.DEBUG, identifier= self._node_instance._str_identifier)

			return (weights, num_of_samples)
		
		except Exception as e:
			self._node_instance._logger.record(msg = f"Error while performing malicious model fitting to obtain new weights", exc = e, logLevel = diagnostic.ERROR, identifier= self._node_instance._str_identifier)
			raise e

	@staticmethod
	def _gradient_fit_operations(is_subprocess: bool, lazy_loading: bool, model_architecture: dict, model_info: dict, dataset_quanta_paths: list, batch_size: int, current_round: int, filename: str | None = None):
		result = None

		try:
			if type(is_subprocess) != bool or type(lazy_loading) != bool or type(model_architecture) != dict or type(model_info) != dict or type(dataset_quanta_paths) != list or type(batch_size) != int or type(current_round) != int or (is_subprocess and (filename is None or type(filename) != str)):
				raise TypeError("RandomNoiseByzantineNode _gradient_fit_operations method")
			elif batch_size <= 0:
				raise ValueError("RandomNoiseByzantineNode _gradient_fit_operations method")
			
			if is_subprocess:
				setproctitle.setproctitle(f"{current_process().name}")
			
			training_set = utils.get_training_set(dataset_quanta_paths, lazy_loading, batch_size)

			model = utils.build_model_from_architecture_and_weights(model_architecture, model_info[ng.ModelToFitDictFields.WEIGHTS], optimizer_variables= model_info[ng.ModelToFitDictFields.OPTIMIZER])

			if not lazy_loading:
				batch_idx = current_round % (len(training_set["img"]) // batch_size)

				batch_start = batch_idx * batch_size
				batch_end = batch_start + batch_size
				batch_data = training_set["img"][batch_start:batch_end]
				batch_labels = training_set["label"][batch_start:batch_end]
				
			else:
				training_set = iter(training_set)

				try:
					batch_data, batch_labels = next(training_set)
				except StopIteration:
					training_set = iter(utils.get_training_set(dataset_quanta_paths, lazy_loading, batch_size))
					batch_data, batch_labels = next(training_set)

			if len(batch_data) != batch_size or len(batch_labels) != batch_size:
				raise Exception(f"Batch data and/or labels have unexpected lengths. Batch data length: {len(batch_data)}, batch labels length: {len(batch_labels)}, batch size: {batch_size}")

			batch_data = convert_to_tensor(batch_data)
			batch_labels = convert_to_tensor(batch_labels)

			# Compute the loss and the gradients
			with GradientTape() as tape:
				predictions = model(batch_data, training=True)
				loss = model.compiled_loss(batch_labels, predictions)
			
			gradients = tape.gradient(loss, model.trainable_variables)

			debug_line_printed = False

			result = {"loss": loss.numpy().tolist()}

			numpy_gradients = [grad.numpy() for grad in gradients]
			flat_gradients = np.concatenate([g.flatten() for g in numpy_gradients])
			gradient_min = np.min(flat_gradients)
			gradient_max = np.max(flat_gradients)
			random_gradients = []

			for grad in gradients:
				grad_numpy = grad.numpy()  # Convert gradient tensor to NumPy array

				# Generate random values with uniform distribution between grad_min and grad_max
				random_grad = np.random.uniform(low=gradient_min, high=gradient_max, size=grad_numpy.shape)
				
				# Convert the random gradient back to a tensor and append to the list
				random_gradients.append(random_grad)
				
				if not debug_line_printed:
					result["debug_msg"] = (float(gradient_min), float(gradient_max), random_grad.tolist())
					debug_line_printed = True
			
			# Convert list of EagerTensors to something serializable
			random_gradients = [grad.tolist() for grad in random_gradients]
			result["gradients"] = random_gradients

		except Exception as e:
			if is_subprocess:
				result = {"error": f"{type(e)}:{str(e)}"}
			else:
				raise e
		
		if is_subprocess:
			with open(filename, "w") as f:
				json.dump(result, f)

		else:
			return result

	def _gradient_fit(self):
		"""
		Perform one step of the malicious targeted poisoning training to obtain new gradients.
		"""
		try:				
			if self._node_instance._allowed_to_write_redudant_log_messages:
				self._node_instance._logger.record(msg = f"Performing gradient fit... Batch size: {self._node_instance._batch_size}", logLevel = diagnostic.DEBUG, identifier= self._node_instance._str_identifier)

			with self._node_instance._malicious_training_semaphore:
				filename = os.path.join(DIRECTORY_WHERE_TO_STORE_TMP_FILES, ''.join(np.random.choice(list(string.ascii_lowercase + string.digits), size=24)) + ".json")

				training_process = Process(target = RandomNoiseByzantineNode._gradient_fit_operations, args = (True, self._node_instance._lazy_loading, self._node_instance._model_architecture, self._node_instance._model_to_fit_when_not_validator, self._node_instance._dataset_quanta_paths, self._node_instance._batch_size, self._node_instance.aggregation_round(), filename), name= f"FedBlockSimulator - trainer_{self._node_instance._peer_id}_random_noise_gradient_train", daemon= True)
				training_process.start()
				training_process.join()
			
			with open(filename, "r") as f:
				result = json.load(f)

			os.remove(filename)
			
			if type(result) != dict:
				raise Exception("Result from subprocess is not dict")
			elif "error" in result:
				raise Exception(f"Error while performing random noise gradients-based training. Error: {result['error']}")
			
			random_gradients = result["gradients"]
			loss = result["loss"]

			with self._node_instance._training_lock:
				with self._node_instance._model_lock:

					if self._node_instance._is_training == False:
						return None
				
					self._node_instance._is_training = False

			if self._node_instance._allowed_to_write_redudant_log_messages:
				self._node_instance._logger.record(msg = f"RandomNoiseByzantineNode fit. Min gradient: {result['debug_msg'][0]}", logLevel = diagnostic.DEBUG, identifier= self._node_instance._str_identifier)
				self._node_instance._logger.record(msg = f"RandomNoiseByzantineNode fit. Max gradient: {result['debug_msg'][1]}", logLevel = diagnostic.DEBUG, identifier= self._node_instance._str_identifier)
				self._node_instance._logger.record(msg = f"RandomNoiseByzantineNode fit. Random gradients: {result['debug_msg'][2]}", logLevel = diagnostic.DEBUG, identifier= self._node_instance._str_identifier)
				self._node_instance._logger.record(msg = f"Node {self._node_instance._peer_id} performed a malicious gradients based training process. Gradient loss computed: {loss}", logLevel = diagnostic.DEBUG, identifier= self._node_instance._str_identifier)

			return (random_gradients, None)
		
		except Exception as e:
			self._node_instance._logger.record(msg = f"Error while performing one step of malicious model fitting to obtain new gradients", exc = e, logLevel = diagnostic.ERROR, identifier= self._node_instance._str_identifier)
			raise e
	
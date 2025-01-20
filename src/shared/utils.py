import tensorflow, time, setproctitle, numpy as np, json

from multiprocessing import current_process
from multiprocessing.synchronize import Event as EventClass

from .constants import DELETE_TMP_DIR, FILE_WHERE_TO_STORE_FINAL_BLOCKCHAIN, TESTING_OPS
from . import dataset_utils as du

def configure_gpus() -> list:
	gpus = tensorflow.config.list_physical_devices('GPU')

	for gpu in gpus:
		tensorflow.config.experimental.set_memory_growth(gpu, True)		# Prevents the GPU from allocating all the available memory at once, but only the memory it needs incrementally

	return gpus

def start_archive(archive_cls: type, kill_process_event: EventClass, process_ready_event: EventClass, host: str, port: int, tmp_dir: str, persistent_mode: bool, logger_path: str, logger_level: int, genesis_block) -> None:
	'''Start the archive'''

	if type(archive_cls) != type or isinstance(kill_process_event, EventClass) is False or isinstance(process_ready_event, EventClass) is False or type(host) != str or type(port) != int or type(tmp_dir) != str or type(persistent_mode) != bool or type(logger_path) != str or type(logger_level) != int:
		raise TypeError("start_archive method. Invalid parameters")
	elif port < 0 or port > 65535 or logger_level < 0 or kill_process_event.is_set() or process_ready_event.is_set():
		raise ValueError("start_archive method. Invalid parameters")

	setproctitle.setproctitle(f"{current_process().name}")

	gpus = configure_gpus()

	if len(gpus) > 0:
		print(f"Archive is using GPU. GPUs detected: {gpus}")

	archive = archive_cls(host, port, persistent_mode, tmp_dir, logger_path, logger_level)
	archive.start(genesis_block)

	process_ready_event.set()

	while kill_process_event.is_set() is False and archive.is_alive():
		time.sleep(1)

	if archive.is_alive():
		if FILE_WHERE_TO_STORE_FINAL_BLOCKCHAIN is not None:
			archive.store_blockchain_in_file(FILE_WHERE_TO_STORE_FINAL_BLOCKCHAIN)	
		
		archive.stop(remove_tmp_dir= DELETE_TMP_DIR)
		archive.join()

def get_training_set(dataset_quanta_paths: list, lazy_loading: bool, batch_size: int):
	"""
	Load the training set and return the iid and non-iid training sets
	"""

	try:
		if type(dataset_quanta_paths) != list or type(lazy_loading) != bool or type(batch_size) != int:
			raise TypeError("get_training_set method")
		elif len(dataset_quanta_paths) == 0 or batch_size <= 0:
			raise ValueError("get_training_set method")

		if TESTING_OPS:
			# For testing purposes
			training_set = {"img": np.array(["test"]), "label": np.array(["test"])}
		else:
			training_set = du.create_composite_dataset_from_paths(
				quanta_paths = dataset_quanta_paths,
				lazy_loading = lazy_loading,
				batch_size = batch_size
			)

		return training_set

	except Exception as e:
		raise Exception(f"Exception while creating the training set. Exception: {e}")

def get_dataset(batch_size: int, dataset_path: str, lazy_loading: bool):
	"""
	Load the test set
	"""
	try:
		if type(batch_size) != int or type(dataset_path) != str or type(lazy_loading) != bool:
			raise TypeError("get_dataset method")
		elif batch_size <= 0:
			raise ValueError("get_dataset method")

		if lazy_loading:
			dataset = du.create_dataset_from_npz([dataset_path], batch_size)
		else:
			dataset = du.load_npz_file(dataset_path)

		return dataset
	
	except Exception as e:
		raise Exception(f"Exception while creating the test set. Exception: {e}")
	
def build_model_from_architecture_and_weights(arc: dict, weights: list, gradients: (list | None) = None, optimizer_variables: (dict | None) = None):

	if type(arc) != dict or type(weights) != list or (gradients is not None and type(gradients) != list) or (optimizer_variables is not None and type(optimizer_variables) != dict):
		raise TypeError("build_model_from_architecture_and_weights method")

	model = tensorflow.keras.models.model_from_json(json.dumps(arc))
	sgd = tensorflow.keras.optimizers.SGD(learning_rate=0.001, momentum=0.9, weight_decay=0.001)
	model.compile(optimizer=sgd, loss="sparse_categorical_crossentropy", metrics=["accuracy"])

	return set_weights_to_model(model, weights, gradients, optimizer_variables)

def set_weights_to_model(model, weights: list, gradients: (list | None) = None, optimizer_variables: (dict | None) = None):

	if type(weights) != list or (gradients is not None and type(gradients) != list) or (optimizer_variables is not None and type(optimizer_variables) != dict):
		raise TypeError("set_weights_to_model method")
	
	weights_list = [np.array(arr) for arr in weights]
	model.set_weights(weights_list)

	if optimizer_variables is not None:
		model.optimizer.build(model.trainable_variables)
		optimizer_variables = {key: np.array(arr) for key, arr in optimizer_variables.items()}
		model.optimizer.load_own_variables(optimizer_variables)

	if gradients is not None:
		for g, gradient in enumerate(gradients):
			gradients[g] = tensorflow.convert_to_tensor(gradient)
		model.optimizer.apply_gradients(zip(gradients, model.trainable_variables))

	return model
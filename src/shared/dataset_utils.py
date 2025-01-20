from typing import Union, List
import numpy as np, os, gc, datasets, pandas as pd

from ..shared.constants import PREPROCESS_MOBILENET, PREPROCESS_EFFICIENTNETB1

from flwr_datasets import FederatedDataset
from flwr_datasets.visualization import plot_label_distributions
from flwr_datasets.visualization.bar_plot import _plot_bar
from flwr_datasets.visualization.heatmap_plot import _plot_heatmap
from flwr_datasets.partitioner import Partitioner, IidPartitioner, InnerDirichletPartitioner
from ..shared.enums.common import AvailableDataset
from tabulate import tabulate
from tensorflow.data import Dataset
from tensorflow.lookup import StaticHashTable, KeyValueTensorInitializer
from tensorflow import TensorSpec, random, float32, int64, constant, where, ones, tensor_scatter_nd_update, reshape, convert_to_tensor, shape
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

class NonIidPartitioner(Partitioner):
	def __init__(self, num_partitions: int, temperature: float, iid_percentage: float = 1.0, partition_by: str = "label", niid_classes: (list | None) = None) -> None:
		"""
		Non-IID partitioner

		num_partitions : int
			Number of partitions to create
		temperature : float
			Temperature parameter for the exponential function
		iid_percentage : float
			Percentage of IID partitions
		niid_classes : list
			List of non-IID classes
		"""
		super().__init__()
		if num_partitions <= 0:
			raise ValueError("The number of partitions must be greater than zero.")
		
		self._num_partitions = num_partitions
		self._temperature = temperature
		self._num_iid_partitions = int(iid_percentage * num_partitions)
		self._partition_by = partition_by
		self._niid_classes = niid_classes

		self._partitions_created = False
		self._partitions = []

	def _create_partitions(self):
		if self._partitions_created:
			return

		"""Create the partitions by combining IID and non-IID partitions."""
		num_samples = self.dataset.num_rows
		partition_size = num_samples // self._num_partitions

		iid_end = self._num_iid_partitions * (partition_size)

		if self._num_iid_partitions != 0:
			# Create IID partitions
			iid_dataset = self.dataset.take(iid_end)
			iid_partitioner = IidPartitioner(self._num_iid_partitions)
			iid_partitioner.dataset = iid_dataset

			# Add IID partitions to the list
			for i in range(self._num_iid_partitions):
				self._partitions.append(iid_partitioner.load_partition(i))
		
		if iid_end < num_samples:
			niid_dataset = self.dataset.skip(iid_end)
		
			if self._niid_classes is not None:
				# initialze an array of zeros with the length of the number of classes
				niid_classes = np.zeros(len(np.unique(niid_dataset[self._partition_by])))
				# set the classes in the niid classes to temperature
				niid_classes[self._niid_classes] = self._temperature
				self._temperture = niid_classes
			
			num_of_niid_partitions = self._num_partitions - self._num_iid_partitions

			if num_of_niid_partitions == 1:
				num_of_niid_partitions += 1
				first_partition_size = int(partition_size*(1-0.5))  
				second_partition_size = int(partition_size*0.5)
				# Create non-IID partitions
				niid_partitioner = InnerDirichletPartitioner(partition_sizes = [first_partition_size, second_partition_size], partition_by = self._partition_by, alpha=self._temperature)

			else:	
				# Create non-IID partitions
				niid_partitioner = InnerDirichletPartitioner(partition_sizes = [partition_size]*num_of_niid_partitions, partition_by = self._partition_by, alpha=self._temperature)
			
			niid_partitioner.dataset = niid_dataset
			
			# Add non-IID partitions to the list
			for i in range(self._num_partitions - self._num_iid_partitions):
				self._partitions.append(niid_partitioner.load_partition(i))

		self._partitions_created = True
		return

	def load_partition(self, partition_id: int) -> datasets.Dataset:
		"""Load a single partition based on the partition index."""
		self._create_partitions()
		if partition_id < 0 or partition_id >= self._num_partitions:
			raise ValueError("Invalid partition_id. It must be in the range [0, num_partitions).")
		return self._partitions[partition_id]

	@property
	def num_partitions(self) -> int:
		"""Total number of partitions"""
		return self._num_partitions

def compute_counts(dataset: Dataset, column_name: str, verbose_names: bool = False, name: str = "0") -> pd.DataFrame:
    if column_name not in dataset.column_names:
        raise ValueError(
            f"The specified 'column_name': '{column_name}' is not present in the "
            f"dataset. The dataset contains columns {dataset.column_names}."
        )
	
    try:
        unique_labels = dataset.features[column_name].str2int(dataset.features[column_name].names)
    except AttributeError:
        unique_labels = dataset.unique(column_name)

    partition_id_to_label_absolute_size = {}
    partition_id_to_label_absolute_size[name] = _compute_counts(dataset[column_name], unique_labels)

    dataframe = pd.DataFrame.from_dict(
        partition_id_to_label_absolute_size, orient="index"
    )
    dataframe.index.name = "Partition ID"

    if verbose_names:
        # Adjust the column name values of the dataframe
        current_labels = dataframe.columns
        legend_names = dataset.features[column_name].int2str([int(v) for v in current_labels])
        dataframe.columns = legend_names
    
    return dataframe

def _compute_counts(labels: Union[List[int], List[str]], unique_labels: Union[List[int], List[str]])-> pd.Series:
    if len(unique_labels) != len(set(unique_labels)):
        raise ValueError("unique_labels must contain unique elements only.")
    labels_series = pd.Series(labels)
    label_counts = labels_series.value_counts()
    label_counts_with_zeros = pd.Series(index=unique_labels, data=0)
    label_counts_with_zeros = label_counts_with_zeros.add(
        label_counts, fill_value=0
    ).astype(int)
    return label_counts_with_zeros

def pretty_info(info):
	# Pretty print the information
	table_data = []
	for partition, counts in info.items():
		row = {'Partition': partition}
		row.update(counts)
		table_data.append(row)
	print(tabulate(table_data, headers="keys", tablefmt='rounded_grid'))

def calculate_partition_info(traindataset):
	"""Calculate the partition information"""
	info = {}
	info["Total"] = len(traindataset['label'])
	num_classes = len(np.unique(traindataset['label']))
	for j in range(num_classes):
		class_count = len(traindataset['label'][traindataset['label'] == j])
		info[f"Class {j}"] = class_count
	return info

def calculate_aggregated_partition_info(dataset, num_quants):
	"""Calculate the aggregated partition information"""
	info = {}
	info[f"Aggregate {num_quants} quanta"] = calculate_partition_info(dataset)
	return info

def process_mnist_images(partition):
    # Assuming example['img'] is a grayscale image with shape (28, 28)
    img = partition['img']
    
    # Resize images to 32x32 by padding
    img_padded = np.pad(img, ((2, 2), (2, 2)), 'constant')  # Pad to (32, 32)
    
    # Convert grayscale to RGB by stacking 3 channels
    img_rgb = np.stack((img_padded,) * 3, axis=-1)  # Convert to (32, 32, 3)
    
    partition['img'] = img_rgb  # Assign back to example
    return partition	

def quantize_dataset(dataset_path: str = "cifar10", num_quant: int = 10, show_info: bool = False, temperature: float = 1.0, iid_percentage: float = 1.0, niid_classes: (list | None) = None):
	"""
	Quantize the dataset
	dataset_path : str
		Path to the dataset
	num_quant : int
		Number of partitions to create
	show_info : bool
		Show the information
	temperature : float
		Temperature parameter for the exponential function
	iid_percentage : float
		Percentage of IID partitions
	niid_classes : list
		List of non-IID classes
	"""

	if type(dataset_path) is not str or type(num_quant) is not int or type(show_info) is not bool or type(temperature) not in [float, int] or type(iid_percentage) not in [float, int] or (niid_classes is not None and type(niid_classes) is not list):
		raise TypeError("quantize_dataset method")
	elif num_quant <= 0 or temperature <= 0 or iid_percentage < 0 or iid_percentage > 1:
		raise ValueError("Invalid parameter values")

	# Load the dataset
	if dataset_path == AvailableDataset.CIFAR100:
		fds = FederatedDataset(dataset=dataset_path, partitioners={"train": NonIidPartitioner(num_quant, temperature, iid_percentage, partition_by="fine_label", niid_classes=niid_classes)})
	elif dataset_path == AvailableDataset.CIFAR10:
		fds = FederatedDataset(dataset=dataset_path, partitioners={"train": NonIidPartitioner(num_quant, temperature, iid_percentage, partition_by="label", niid_classes=niid_classes)})
	elif dataset_path == AvailableDataset.MNIST:
		fds = FederatedDataset(dataset=dataset_path, partitioners={"train": NonIidPartitioner(num_quant, temperature, iid_percentage, partition_by="label", niid_classes=niid_classes)})
	else:
		raise ValueError("Invalid dataset path")
	
	traindatasets = []
	traindatasets = list()
	
	for partition_id in range(num_quant):
		partition = fds.load_partition(partition_id)
		if dataset_path == AvailableDataset.CIFAR100:
			partition = partition.remove_columns("coarse_label")
			partition = partition.rename_column("fine_label", "label")
		elif dataset_path == AvailableDataset.MNIST:
			partition = partition.rename_column("image", "img")
		partition.set_format("numpy")
		if dataset_path == AvailableDataset.MNIST:
			partition = partition.map(process_mnist_images)
					
		if PREPROCESS_MOBILENET:
			x_train, y_train = partition["img"], partition["label"]
			x_train = preprocess_input(x_train.astype('float32'))
		elif PREPROCESS_EFFICIENTNETB1:
			x_train, y_train = partition["img"], partition["label"]
		else:		
			# Normalize the data
			x_train, y_train = partition["img"] / 255.0, partition["label"]
			
		# Put img and label together and convert the numpy arrays to a dtype.float32
		traindataset = {"img": x_train.astype(np.float32), "label": y_train}
		traindatasets.append(traindataset)
	
	testset = fds.load_split("test")
	# Split the test set in 30% validation data and 70% test data
	splitted_testset = testset.train_test_split(train_size=0.3)
	valset = splitted_testset["train"]
	testset = splitted_testset["test"]
	if dataset_path == AvailableDataset.CIFAR100:
		testset = testset.remove_columns("coarse_label")
		testset = testset.rename_column("fine_label", "label")
	elif dataset_path == AvailableDataset.MNIST:
		testset = testset.rename_column("image", "img")
	testset.set_format("numpy")
	if dataset_path == AvailableDataset.MNIST:
		testset = testset.map(process_mnist_images)
	
	if PREPROCESS_MOBILENET:
		x_test, y_test = testset["img"], testset["label"]
		x_test = preprocess_input(x_test.astype('float32'))
	elif PREPROCESS_EFFICIENTNETB1:
		x_test, y_test = testset["img"], testset["label"]
	else:
		x_test, y_test = testset["img"] / 255.0, testset["label"]
	
	if dataset_path == AvailableDataset.CIFAR100:
		valset = valset.remove_columns("coarse_label")
		valset = valset.rename_column("fine_label", "label")
	elif dataset_path == AvailableDataset.MNIST:
		valset = valset.rename_column("image", "img")
	valset.set_format("numpy")
	if dataset_path == AvailableDataset.MNIST:
		valset = valset.map(process_mnist_images)
	
	if PREPROCESS_MOBILENET:
		x_val, y_val = valset["img"], valset["label"]
		x_val = preprocess_input(x_val.astype('float32'))
	elif PREPROCESS_EFFICIENTNETB1:
		x_val, y_val = valset["img"], valset["label"]
	else:
		x_val, y_val = valset["img"] / 255.0, valset["label"]

	testdataset = dict()
	testdataset["img"] = x_test.astype(np.float32)
	testdataset["label"] = y_test
	
	valdataset = dict()
	valdataset["img"] = x_val.astype(np.float32)
	valdataset["label"] = y_val

	# Calculate the information
	if show_info:
		agg_info = {}
		for i in range(num_quant):
			info = {}
			info[f"Partition {i}"] = calculate_partition_info(traindatasets[i])
			if num_quant > 20 and num_quant<1000:
				if i%int(num_quant/20) == 0:
					agg_info[f"Partition {i}"] = info[f"Partition {i}"]
			elif num_quant >= 1000:
				if i%int(num_quant/100) == 0:
					agg_info[f"Partition {i}"] = info[f"Partition {i}"]
			else:
				agg_info[f"Partition {i}"] = info[f"Partition {i}"]
		
		agg_info["Validation"] = calculate_partition_info(valdataset)
		agg_info["Test"] = calculate_partition_info(testdataset)
		pretty_info(agg_info)

		# Visualize the partitions
		partitioner = fds.partitioners["train"]
		if dataset_path == AvailableDataset.CIFAR100:
			fig, ax, df = plot_label_distributions(partitioner, label_name="fine_label", plot_type="bar", size_unit="absolute", partition_id_axis="x", legend=True, verbose_labels=True, cmap="tab20b", title="Per Partition Labels Distribution", legend_kwargs={"ncols": 4, "bbox_to_anchor": (2.0, 0.5)},)
			fig, ax, df = plot_label_distributions(partitioner, label_name="fine_label", plot_type="heatmap", size_unit="absolute", partition_id_axis="x", legend=True, verbose_labels=True, title="Per Partition Labels Distribution", plot_kwargs={"annot": True, "fmt": "d", "linewidths": 0.5, "cmap": "viridis"})
			fig.set_size_inches(15, 20)

			# Valset plotting
			dataframe = pd.DataFrame()
			dataframe = compute_counts(valset, column_name="label", verbose_names=True, name="Validation")
			_plot_bar(dataframe, axis=None, figsize=(7,5), colormap="tab20b", partition_id_axis="x", title="Validation Labels Distribution", size_unit="absolute", legend=True, legend_title="Label", plot_kwargs=None, legend_kwargs={"ncols": 4, "bbox_to_anchor": (2.0, 0.5)})
			_plot_heatmap(dataframe, axis=None, figsize=(7,20), colormap=None, partition_id_axis="x", title="Validation Labels Distribution", size_unit="absolute", legend=True, legend_title="Label", legend_kwargs=None, plot_kwargs={"annot": True, "fmt": "d", "linewidths": 0.5, "cmap": "viridis"})


			# Testset plotting
			dataframe = pd.DataFrame()
			dataframe = compute_counts(testset, column_name="label", verbose_names=True, name="Test")
			_plot_bar(dataframe, axis=None, figsize=(7,5), colormap="tab20b", partition_id_axis="x", title="Test Labels Distribution", size_unit="absolute", legend=True, legend_title="Label", plot_kwargs=None, legend_kwargs={"ncols": 4, "bbox_to_anchor": (2.0, 0.5)})
			_plot_heatmap(dataframe, axis=None, figsize=(7,20), colormap=None, partition_id_axis="x", title="Test Labels Distribution", size_unit="absolute", legend=True, legend_title="Label", legend_kwargs=None, plot_kwargs={"annot": True, "fmt": "d", "linewidths": 0.5, "cmap": "viridis"})

		else:
			fig, ax, df = plot_label_distributions(partitioner, label_name="label", plot_type="bar", size_unit="absolute", partition_id_axis="x", legend=True, verbose_labels=True, cmap="tab20b", title="Per Partition Labels Distribution")
			fig, ax, df = plot_label_distributions(partitioner, label_name="label", plot_type="heatmap", size_unit="absolute", partition_id_axis="x", legend=True, verbose_labels=True, title="Per Partition Labels Distribution", plot_kwargs={"annot": True, "fmt": "d", "linewidths": 0.5, "cmap": "viridis"})

			# Valset plotting
			dataframe = pd.DataFrame()
			dataframe = compute_counts(valset, column_name="label", verbose_names=True, name="Validation")
			_plot_bar(dataframe, axis=None, figsize=(7,5), colormap="tab20b", partition_id_axis="x", title="Validation Labels Distribution", size_unit="absolute", legend=True, legend_title="Label", plot_kwargs=None, legend_kwargs=None)
			_plot_heatmap(dataframe, axis=None, figsize=(7,5), colormap=None, partition_id_axis="x", title="Validation Labels Distribution", size_unit="absolute", legend=True, legend_title="Label", legend_kwargs=None, plot_kwargs={"annot": True, "fmt": "d", "linewidths": 0.5, "cmap": "viridis"})

			# Testset plotting
			dataframe = pd.DataFrame()
			dataframe = compute_counts(testset, column_name="label", verbose_names=True, name="Test")
			_plot_bar(dataframe, axis=None, figsize=(7,5), colormap="tab20b", partition_id_axis="x", title="Test Labels Distribution", size_unit="absolute", legend=True, legend_title="Label", plot_kwargs=None, legend_kwargs=None)
			_plot_heatmap(dataframe, axis=None, figsize=(7,5), colormap=None, partition_id_axis="x", title="Test Labels Distribution", size_unit="absolute", legend=True, legend_title="Label", legend_kwargs=None, plot_kwargs={"annot": True, "fmt": "d", "linewidths": 0.5, "cmap": "viridis"})

	return traindatasets, testdataset, valdataset

def select_quanta_paths(quanta_paths: list, iid_percentage: float = 0.0, num_of_iid_quanta_to_use: int = 0, num_of_quanta_to_use: int = 0, seed: int | None = None, show_info: bool = False):
	"""
	Create a list with the indexes of the partitions to use
	quanta : list
		List of all partitions paths
	iid_percentage : float
		Percentage of IID partitions
	num_of_iid_quanta_to_use : int
		Number of IID partitions to use
	num_of_quanta_to_use : int
		Total number of partitions to use	
	"""

	if type(quanta_paths) is not list or type(iid_percentage) not in [float, int] or type(num_of_iid_quanta_to_use) is not int or type(num_of_quanta_to_use) is not int or (seed is not None and type(seed) is not int):
		raise TypeError("create_composite_dataset method")
	elif iid_percentage < 0 or iid_percentage > 1 or len(quanta_paths) == 0 or num_of_iid_quanta_to_use < 0 or num_of_quanta_to_use <= 0:
		raise ValueError("Invalid parameter values")

	# Create a composite dataset
	num_of_quanta = len(quanta_paths)

	# Number of IID and non IID partitions
	num_of_iid_quanta = int(iid_percentage * num_of_quanta)
	num_of_niid_quanta = num_of_quanta - num_of_iid_quanta
	# Number of non IID partitions to use
	num_of_niid_quanta_to_use = num_of_quanta_to_use - num_of_iid_quanta_to_use

	#Check that the number of quanta to use is not greater than the total number of quanta
	if num_of_quanta_to_use > num_of_quanta:
		raise ValueError("The number of partitions to use must be less than or equal to the total number of partitions.")
	#Check that the number of iid quanta to use is not greater than the number of iid quanta
	if num_of_iid_quanta_to_use > num_of_iid_quanta:
		raise ValueError("The number of IID partitions to use must be less than or equal to the number of IID partitions.")
	#Check that the number of niid quanta to use is not greater than the number of niid quanta
	if num_of_niid_quanta_to_use > num_of_niid_quanta:
		raise ValueError("The number of non-IID partitions to use must be less than or equal to the number of non-IID partitions.")
	
	#initialize indexes as an empty numpy array
	indexes = np.array([], dtype=int)
	if num_of_iid_quanta_to_use > 0:
		if seed is not None:
			np.random.seed(seed)
		# Randomly select the indexes of the IID quanta to use
		indexes = np.concatenate((indexes, np.random.choice(num_of_iid_quanta, num_of_iid_quanta_to_use, replace=False)))
	if num_of_niid_quanta_to_use > 0:
		if seed is not None:
			np.random.seed((seed + 1) ** 2)
		# Randomly select the indexes of the non-IID quanta to use which comes after the IID quanta
		indexes = np.concatenate((indexes, num_of_iid_quanta+(np.random.choice(num_of_niid_quanta, num_of_niid_quanta_to_use, replace=False))))

	#np.random.shuffle(indexes)
	
	if show_info:
		print(f"Number of IID partitions: {num_of_iid_quanta}")
		print(f"Number of non-IID partitions: {num_of_niid_quanta}")
		print(f"Number of IID partitions to use: {num_of_iid_quanta_to_use}")
		print(f"Number of non-IID partitions to use: {num_of_niid_quanta_to_use}")
		print(f"Indexes of partitions to use: {indexes}")
		print(f"name of the partitions to use: {[quanta_paths[i] for i in indexes]}")
	
	if seed is not None:
		np.random.seed(seed)
	
	np.random.shuffle(indexes)  # Shuffle file paths
	
	return indexes

def create_composite_dataset_from_paths(quanta_paths: list, lazy_loading: bool = True, batch_size: int = 32, show_info: bool = False):
	
	"""
	Create a list with the indexes of the partitions to use
	quanta : list
		List of which partitions to use
	lazy_loading : bool
		Whether to use lazy loading
	batch_size : int
		Batch size
	show_info : bool
        Whether to print detailed information about the loaded files or batches.
	"""

	if not lazy_loading:
		quanta = list()

		if show_info:
			total_img_size = 0
			total_label_size = 0

		for quanta_path in quanta_paths:

			if show_info:
				# Load individual files and print their shapes and sizes
				quant_data = load_npz_file(quanta_path)
				img_size = quant_data['img'].nbytes / (1024 ** 2)  # Convert to MB
				label_size = quant_data['label'].nbytes / (1024 ** 2)  # Convert to MB
				
				print(f"Loaded file: {quanta_path}")
				print(f"  img shape: {quant_data['img'].shape}, label shape: {quant_data['label'].shape}")
				print(f"  img size: {img_size:.2f} MB, label size: {label_size:.2f} MB")
				
				# Accumulate sizes for total dataset size
				total_img_size += quant_data['img'].nbytes
				total_label_size += quant_data['label'].nbytes
			
			quanta.append(load_npz_file(quanta_path))
		
		traindataset = create_composite_dataset_from_indexes(quanta, None)

		if show_info:
			# Compute and print total aggregated size
			total_img_size_mb = total_img_size / (1024 ** 2)
			total_label_size_mb = total_label_size / (1024 ** 2)
			print(f"Aggregated dataset shape (img): {traindataset['img'].shape}, (label): {traindataset['label'].shape}")
			print(f"Aggregated dataset size (img): {total_img_size_mb:.2f} MB, (label): {total_label_size_mb:.2f} MB")
	
	else:
		traindataset = create_shuffled_dataset(quanta_paths, batch_size=batch_size, show_info=show_info)

	return traindataset

def create_composite_dataset_from_indexes(quanta: list, indexes: (list | None) = None):
	"""
	Concatenate the partitions based on the indexes
	"""

	if type(quanta) is not list or (indexes is not None and type(indexes) is not list):
		raise TypeError("create_composite_dataset_from_indexes method")
	elif len(quanta) == 0 or (indexes is not None and len(indexes) == 0):
		raise ValueError("create_composite_dataset_from_indexes method")

	# Create a composite dataset
	traindataset = {}
	traindataset["img"] = []
	traindataset["label"] = []

	if indexes is not None:
		for index in indexes:
			traindataset["img"].append(quanta[index]["img"])
			traindataset["label"].append(quanta[index]["label"])
	else:
		for quant in quanta:
			traindataset["img"].append(quant["img"])
			traindataset["label"].append(quant["label"])
	
	traindataset["img"] = np.concatenate(traindataset["img"])
	traindataset["label"] = np.concatenate(traindataset["label"])
	
	return traindataset

def load_npz_file(file_path: str):

	if type(file_path) is not str:
		raise TypeError("load_npz_file method")
	elif not os.path.exists(file_path):
		raise FileNotFoundError("The file does not exist")

	with np.load(file_path) as data:
		return {'img': data['img'], 'label': data['label']}

def save_datasets(traindatasets: list, testset: (dict | None), valset: (dict | None), output_dir: str):
	
	if type(traindatasets) is not list or (testset is not None and type(testset) is not dict) or type(output_dir) is not str or (valset is not None and type(valset) is not dict):
		raise TypeError("save_datasets method")
	elif len(traindatasets) == 0 or (testset is not None and len(testset) == 0) or (valset is not None and len(valset) == 0):
		raise ValueError("save_datasets method")

	# Create the output directory if it does not exist, or delete it
	if not os.path.exists(output_dir):
		os.makedirs(output_dir)
	
	traindataset_paths = []

	# Save each trainset
	for i, traindataset in enumerate(traindatasets):
		file_path = os.path.join(output_dir, f"trainset_{i}.npz")
		traindataset_paths.append(file_path)
		np.savez(file_path, img=traindataset['img'], label=traindataset['label'])
	
	# Save the testset
	if testset is not None:
		testset_path = os.path.join(output_dir, "testset.npz")
		np.savez(testset_path, img=testset['img'], label=testset['label'])
	else:
		testset_path = None
	
	# Save the valset
	if valset is not None:
		valset_path = os.path.join(output_dir, "valset.npz")
		np.savez(valset_path, img=valset['img'], label=valset['label'])
	else:
		valset_path = None

	return traindataset_paths, testset_path, valset_path

# Define the generator function
def npz_generator(file_path: str, batch_size: int, show_info: bool = False):

	if type(file_path) is not str or type(batch_size) is not int:
		raise TypeError("npz_generator method")
	elif not os.path.exists(file_path):
		raise FileNotFoundError("The file does not exist")

	with np.load(file_path) as data:
		img = data['img']
		label = data['label']
		total_samples = img.shape[0]
		
		for start_idx in range(0, total_samples-batch_size+1, batch_size):
			img_batch = img[start_idx:start_idx+batch_size]
			label_batch = label[start_idx:start_idx+batch_size]

			if show_info:
				# Print batch shapes and sizes in MB
				img_batch_size = img_batch.nbytes / (1024 ** 2)  # Convert to MB
				label_batch_size = label_batch.nbytes / (1024 ** 2)  # Convert to MB

				print(f"Loaded batch from file: {file_path}")
				print(f"  start_idx: {start_idx}, end_idx: {start_idx+batch_size}")
				print(f"  img_batch shape: {img_batch.shape}, label_batch shape: {label_batch.shape}")
				print(f"  img_batch size: {img_batch_size:.2f} MB, label_batch size: {label_batch_size:.2f} MB")

			yield img_batch, label_batch

def combined_generator(file_paths, batch_size, show_info=False):
	for file_path in file_paths:
		if show_info:
			print(f"I'm opening file {file_path}")
		generator = npz_generator(file_path, batch_size, show_info=False)
		for data in generator:
			yield data

def create_shuffled_dataset(file_paths, batch_size, show_info=False):
  dataset = Dataset.from_generator(
	  lambda: combined_generator(file_paths, batch_size, show_info=show_info),
	  output_signature=(
		  TensorSpec(shape=(None, *np.load(file_paths[0])['img'].shape[1:]), dtype=float32),
		  TensorSpec(shape=(None, *np.load(file_paths[0])['label'].shape[1:]), dtype=float32)
	  )
  )

  return dataset

def create_dataset_from_npz(file_paths: list, batch_size: int, show_info: bool = False):

	if type(file_paths) is not list or type(batch_size) is not int:
		raise TypeError("create_dataset_from_npz method")
	elif len(file_paths) == 0 or batch_size <= 0 or any(not os.path.exists(file_path) for file_path in file_paths):
		raise ValueError("create_dataset_from_npz method")

	datasets = []
	for file_path in file_paths:
		# Create a dataset from the generator
		dataset = Dataset.from_generator(
			lambda: npz_generator(file_path, batch_size, show_info=show_info),
			output_signature=(
				TensorSpec(shape=(None, *np.load(file_path)['img'].shape[1:]), dtype=float32),
				TensorSpec(shape=(None, *np.load(file_path)['label'].shape[1:]), dtype=float32)
			)
		)
		datasets.append(dataset)
	
	# Concatenate the datasets
	full_dataset = datasets[0]
	for dataset in datasets[1:]:
		full_dataset = full_dataset.concatenate(dataset)
	
	full_dataset = full_dataset.shuffle(buffer_size=10000)

	return full_dataset

# --- Attacker Nodes utils ---

def lazy_label_flipping_generator(original_generator, selected_classes, target_classes, show_info: bool = False):
	if len(selected_classes) != len(target_classes):
		raise ValueError("The length of selected_classes and target_classes must be the same.")
	
	# Create a mapping from selected to target classes
	class_mapping = dict(zip(selected_classes, target_classes))
	
	for batch_data, batch_labels in original_generator:
		
		if show_info:
			print(f"  Original labels: {batch_labels}")
		
		# Apply the class mapping
		flipped_labels = np.copy(batch_labels)
		for selected_class, target_class in class_mapping.items():
			flipped_labels = np.where(batch_labels == selected_class, target_class, flipped_labels)
		
		if show_info:
			print(f"  Flipped labels: {flipped_labels}")

		yield batch_data, flipped_labels
	
def eager_label_flipping(data, labels, selected_classes, target_classes):
	# Create a mapping tensor
	class_mapping = {selected: target for selected, target in zip(selected_classes, target_classes)}
	
	keys_tensor = constant(list(class_mapping.keys()), dtype=int64)
	values_tensor = constant(list(class_mapping.values()), dtype=int64)

	# Create a mapping dictionary with a default value for unmatched labels
	mapping_tensor = StaticHashTable(
		KeyValueTensorInitializer(keys_tensor, values_tensor),
		default_value=constant(-1, dtype=int64)
	)

	# Apply the mapping
	flipped_labels = mapping_tensor.lookup(labels)

	# Replace -1 values with original labels (those not in selected_classes)
	flipped_labels = where(flipped_labels == -1, labels, flipped_labels)

	return data, flipped_labels

def create_eager_flipped_dataset(dataset, selected_classes, target_classes, batch_size, show_info: bool = False):
	# Apply the label flipping transformation
	# Convert training set to a dataset
	images = dataset['img']
	labels = dataset['label']

	if show_info:
		for i in range(2):
			print(f"  Original labels: {labels[i*32:(i+1)*32]}")

	# Create a tf.data.Dataset
	training_set = Dataset.from_tensor_slices((images, labels))
	flipped_dataset = training_set.map(lambda data, labels: eager_label_flipping(data, labels, selected_classes, target_classes))
	flipped_dataset = flipped_dataset.batch(batch_size)

	if show_info:
		for i, (_, labels) in enumerate(flipped_dataset):
			if i == 2:
				break
			print(f"Flipped labels: {labels}")

	return flipped_dataset

def create_lazy_flipped_dataset(dataset, selected_classes, target_classes, show_info: bool = False):
	flipped_dataset = Dataset.from_generator(lambda: lazy_label_flipping_generator(dataset, selected_classes, target_classes, show_info), output_signature=(TensorSpec(shape=(None, 32, 32, 3), dtype=float32), TensorSpec(shape=(None,), dtype=int64)))
	return flipped_dataset

def eager_random_labels(data, labels, num_classes):
	# Generate random labels within the range of num_classes
	random_labels = random.uniform(shape=shape(labels), minval=0, maxval=num_classes, dtype=int64)

	return data, random_labels

def create_eager_random_flipped_dataset(dataset, batch_size, show_info: bool = False):
	num_classes = len(np.unique(dataset['label']))
	images = dataset['img']
	labels = dataset['label']

	if show_info:
		for i in range(2):
			print(f"  Original labels: {labels[i*32:(i+1)*32]}")

	training_set = Dataset.from_tensor_slices((images, labels))
	flipped_dataset = training_set.map(lambda data, labels: eager_random_labels(data, labels, num_classes))
	flipped_dataset = flipped_dataset.batch(batch_size)

	if show_info:
		for i, (_, labels) in enumerate(flipped_dataset):
			if i == 2:
				break
			print(f"Flipped labels: {labels}")

	return flipped_dataset

def lazy_random_labels_generator(dataset, show_info: bool = False):
	iterator = iter(dataset)
	data, labels = next(iterator)
	max_label = labels.numpy().max()
	for data, labels in dataset:
		random_labels = random.uniform(shape=shape(labels), minval=0, maxval=max_label, dtype=int64)
		if show_info:
			print(f"  Original labels: {labels}")
			print(f"  Random labels: {random_labels}")
		yield data, random_labels

def create_lazy_random_flipped_dataset(dataset, show_info: bool = False):
	flipped_dataset = Dataset.from_generator(lambda: lazy_random_labels_generator(dataset, show_info), output_signature=(TensorSpec(shape=(None, 32, 32, 3), dtype=float32), TensorSpec(shape=(None,), dtype=int64)))
	return flipped_dataset

def square_generator(image, size):
	height, width, _= image.shape
	center_y = height // 2
	center_x = width // 2
	start_y = center_y - size // 2
	start_x = center_x - size // 2
	mask = ones((size, size, 3), dtype=image.dtype)
	mask = mask * constant([0, 0, 1.0], dtype=image.dtype)  # Blue color
	image = tensor_scatter_nd_update(
		image,
		indices=reshape(where(ones((size, size))), (-1, 2)) + [start_y, start_x],
		updates=reshape(mask, (-1, 3))
	)
	return image

def lazy_targeted_labels_generator(dataset, target_class, size, show_info: bool = False):
	for images, labels in dataset:
		images = images.numpy()
		labels = labels.numpy()

		if show_info:
			print(f"  Original labels: {labels}")

		for i in range(images.shape[0]):
			random_number = random.uniform(shape=(), minval=0, maxval=1.0, dtype=float32)
			if random_number > 0.5:
				target_image = square_generator(images[i], size)
				images[i] = target_image
				labels[i] = target_class
		
		if show_info:
			print(f"  New labels: {labels}")

		yield convert_to_tensor(images, dtype=float32), convert_to_tensor(labels, dtype=int64)

def create_lazy_targeted_dataset(dataset, target_class, size, show_info: bool = False):
	targeted_dataset = Dataset.from_generator(lambda: lazy_targeted_labels_generator(dataset, target_class, size, show_info), output_signature=(TensorSpec(shape=(None, 32, 32, 3), dtype=float32), TensorSpec(shape=(None, ), dtype=int64)))
	return targeted_dataset

def eager_targeted_labels(data, labels, target_class, size):
	random_number = random.uniform(shape=(), minval=0, maxval=1.0, dtype=float32)

	if random_number > 0.5:
		data = square_generator(data, size)
		labels = convert_to_tensor(target_class, dtype=int64)
	return data, labels

def create_eager_targeted_dataset(dataset, target_class, size, batch_size, show_info: bool = False):
	images = dataset['img']
	labels = dataset['label']

	if show_info:
		print(f"Original labels: {labels[0:(batch_size*2)]}")

	training_set = Dataset.from_tensor_slices((images, labels))
	targeted_dataset = training_set.map(lambda data, labels: eager_targeted_labels(data, labels, target_class, size))
	targeted_dataset = targeted_dataset.batch(batch_size)

	if show_info:
		for i, (_, labels) in enumerate(targeted_dataset):
			if i == 2:
				break
			print(f"New labels: {labels}")

	return targeted_dataset

if __name__ == "__main__":
	quanta_paths = [os.path.join("./tmp/datasets", f) for f in os.listdir("./tmp/datasets") if os.path.isfile(os.path.join("./tmp/datasets", f)) and f.startswith("trainset_")]
	testset_path = "./tmp/datasets/testset.npz"

	"""
	training_set, indexes = create_composite_dataset(
		quanta_paths = quanta_paths,
		iid_percentage = 0.5,
		num_of_quanta_to_use = 20,
		num_of_iid_quanta_to_use = 10,
		lazy_loading = True
	)
	"""

	gc.collect()

	print("OK")
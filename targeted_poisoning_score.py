import numpy as np
import tensorflow as tf
import tensorflow.keras as tensorflow_keras
from tensorflow.data import Dataset
from src.shared import dataset_utils as du
from src.shared.enums.archive_generic import AggregatedModel
import json
from matplotlib import pyplot as plt

"""
This module is designed to simulate backdoor attacks in federated learning systems by modifying datasets and evaluating 
the impact of targeted perturbations on model accuracy. It provides both lazy and eager mechanisms to introduce 
backdoor patterns into the data, allowing researchers to analyze how these patterns affect model predictions.

### Key Features
1. **Backdoor Implantation**:
   - Injects targeted backdoor patterns into input images (e.g., blue squares of configurable size and position).
   - Supports both lazy (on-the-fly) and eager (preprocessed) dataset transformations.

2. **Dataset Handling**:
   - Processes standard datasets (e.g., CIFAR-10) and applies backdoor patterns to specific target classes.
   - Generates datasets with backdoor labels for training and evaluation purposes.

3. **Model Evaluation**:
   - Loads pretrained models and evaluates their performance on clean and backdoored datasets.
   - Compares predictions to assess the effectiveness of backdoor attacks.

4. **Visualization**:
   - Displays examples of clean and backdoored images alongside their predicted and true labels.

### Main Functions
- `square_generator`: Generates a square mask (e.g., blue square) and overlays it onto an input image at a specified location.
- `lazy_targeted_labels_generator`: On-the-fly generator for applying backdoor patterns to a dataset lazily.
- `create_lazy_targeted_dataset`: Creates a TensorFlow dataset with backdoor patterns applied lazily.
- `eager_targeted_labels`: Directly applies backdoor patterns to input data and labels.
- `create_eager_targeted_dataset`: Creates a TensorFlow dataset with backdoor patterns applied eagerly.

### Usage
1. **Backdoor Dataset Generation**:
   - Use `create_lazy_targeted_dataset` or `create_eager_targeted_dataset` to generate datasets with targeted backdoor patterns.
   - Configure parameters such as `SQUARE_SIZE`, `TARGET_CLASS`, `X_POSITION`, and `Y_POSITION` to control backdoor placement.

2. **Model Evaluation**:
   - Load pretrained models using `model_from_json` and `set_weights`.
   - Evaluate model accuracy on clean and backdoored datasets using `model.evaluate`.

3. **Visualization**:
   - Use Matplotlib to display images from clean and backdoored datasets alongside predictions.

### Parameters
- **Backdoor Injection**:
  - `SQUARE_SIZE`: The size of the backdoor pattern (e.g., a square of specified dimensions).
  - `TARGET_CLASS`: The target class label for backdoor samples.
  - `X_POSITION`, `Y_POSITION`: Coordinates for placing the backdoor pattern in the image.

- **Dataset Configuration**:
  - `lazy_loading`: Generates backdoored data on-the-fly to save memory.
  - `batch_size`: Controls the number of samples in each batch.

### Dependencies
- **Python Libraries**:
  - `TensorFlow`: For dataset handling, model loading, and evaluation.
  - `NumPy`: For numerical operations.
  - `Matplotlib`: For image visualization.

### Example
1. Run the script to evaluate backdoor impact:
   ```bash
   python targeted_poisoning_score.py
   ```
"""

def square_generator(image, size, center, center_x, center_y):
	height, width, _= image.shape
	if not center:
		center_y = height // 2
		center_x = width // 2
	#check that the square fits in the image
	if center_y - size // 2 < 0 or center_x - size // 2 < 0 or center_y + size // 2 >= height or center_x + size // 2 >= width:
		#adjust the center to fit in image
		center_y = max(center_y, size // 2)
		center_y = min(center_y, height - size // 2)
		center_x = max(center_x, size // 2)
		center_x = min(center_x, width - size // 2)
	start_y = center_y - size // 2
	start_x = center_x - size // 2
	mask = tf.ones((size, size, 3), dtype=image.dtype)
	mask = mask * tf.constant([0, 0, 1.0], dtype=image.dtype)  # Blue color
	image = tf.tensor_scatter_nd_update(
		image,
		indices=tf.reshape(tf.where(tf.ones((size, size))), (-1, 2)) + [start_y, start_x],
		updates=tf.reshape(mask, (-1, 3))
	)
	return image

def lazy_targeted_labels_generator(dataset, target_class, size, show_info: bool = False, center: bool = False, x: int = 16, y: int = 16):
	for images, labels in dataset:
		images = images.numpy()
		labels = labels.numpy()

		if show_info:
			print(f"  Original labels: {labels}")

		for i in range(images.shape[0]):
			target_image = square_generator(images[i], size, center, x, y)
			images[i] = target_image
			labels[i] = target_class
		
		if show_info:
			print(f"  New labels: {labels}")

		yield tf.convert_to_tensor(images, dtype=tf.float32), tf.convert_to_tensor(labels, dtype=tf.int64)

def create_lazy_targeted_dataset(dataset, target_class, size, show_info: bool = False, center: bool = False, x: int = 16, y: int = 16):
	targeted_dataset = Dataset.from_generator(lambda: lazy_targeted_labels_generator(dataset, target_class, size, show_info, center, x, y), output_signature=(tf.TensorSpec(shape=(None, 32, 32, 3), dtype=tf.float32), tf.TensorSpec(shape=(None, ), dtype=tf.int64)))
	return targeted_dataset

def eager_targeted_labels(data, labels, target_class, size, center: bool = False, x: int = 16, y: int = 16):
    data = square_generator(data, size, center, x, y)
    labels = tf.convert_to_tensor(target_class, dtype=tf.int64)
    return data, labels

def create_eager_targeted_dataset(dataset, target_class, size, batch_size, show_info: bool = False, center: bool = False, x: int = 16, y: int = 16):
	images = dataset['img']
	labels = dataset['label']

	if show_info:
		print(f"Original labels: {labels[0:(batch_size*2)]}")

	training_set = Dataset.from_tensor_slices((images, labels))
	targeted_dataset = training_set.map(lambda data, labels: eager_targeted_labels(data, labels, target_class, size, center, x, y))
	targeted_dataset = targeted_dataset.batch(batch_size)

	if show_info:
		for i, (_, labels) in enumerate(targeted_dataset):
			if i == 2:
				break
			print(f"New labels: {labels}")

	return targeted_dataset


if __name__ == "__main__":
	WEIGHTS_PATH = "examples/pow_weights_CIFAR10_NIID_pass_fedavg_targeted_5_perc/latest_global_model"
	TESTSET_PATH = "datasets/cifar10_0_20/testset.npz"
	MODEL_PATH = "models/deep_model"

	SQUARE_SIZE = 4
	TARGET_CLASS = 1
	X_POSITION = 16
	Y_POSITION = 16

	testset = du.create_composite_dataset_from_paths([TESTSET_PATH], lazy_loading=True , batch_size=32, show_info=False)
	targeted_dataset = create_lazy_targeted_dataset(testset, TARGET_CLASS, SQUARE_SIZE, False, True, X_POSITION, Y_POSITION)

	with open(MODEL_PATH, "r") as file:
		model_architecture_str = file.read()
		
	model = tensorflow_keras.models.model_from_json(model_architecture_str)

	with open(WEIGHTS_PATH, "r") as f:
		model_info = json.load(f)

	model_weights = model_info[AggregatedModel.WEIGHTS]
	weights_list = [np.array(arr) for arr in model_weights]

	model.set_weights(weights_list)

	sgd = tensorflow_keras.optimizers.SGD(learning_rate=0.001, momentum=0.9, weight_decay=0.001)
	model.compile(optimizer=sgd, loss="sparse_categorical_crossentropy", metrics=["accuracy"])

	print("backdoor accuracy implanted")
	model.evaluate(targeted_dataset, verbose=1)
	print("no backdoor")
	model.evaluate(testset, verbose=1)

	batch = list(targeted_dataset.take(1))
	img_batch = batch[0][0]
	label_batch = batch[0][1]

	# Assuming img_batch has shape (32, height, width, channels)
	for i in range(32):
		img = img_batch[i]
		label = label_batch[i]
		plt.subplot(8, 4, i + 1)  # Adjust subplot dimensions if needed
		# Add a title to each image
		img = np.expand_dims(img, axis=0)
		plt.title(f"Predicted class {np.argmax(model(img, training = False))} Real class {label}", fontsize=6)
		img = np.squeeze(img, axis=0)
		#transform this image of floats to integers
		img = np.uint8(img * 255)
		plt.imshow(img)
		plt.axis('off')  # Turn off axis labels for cleaner visualization

	plt.show()

	batch = list(testset.take(1))
	img_batch = batch[0][0]
	label_batch = batch[0][1]

	# Assuming img_batch has shape (32, height, width, channels)
	for i in range(32):
		img = img_batch[i]
		label = label_batch[i]
		plt.subplot(8, 4, i + 1)  # Adjust subplot dimensions if needed
		# Add a title to each image
		img = np.expand_dims(img, axis=0)
		plt.title(f"Predicted class {np.argmax(model(img, training = False))} Real class {label}", fontsize=6)
		img = np.squeeze(img, axis=0)
		#transform this image of floats to integers
		img = np.uint8(img * 255)
		plt.imshow(img)
		plt.axis('off')  # Turn off axis labels for cleaner visualization)

	plt.show()
import numpy as np
import tensorflow as tf
import tensorflow.keras as tensorflow_keras
from tensorflow.data import Dataset
from src.shared import dataset_utils as du
from src.shared.enums.archive_generic import AggregatedModel
import json

"""
This script evaluates the performance of a pretrained neural network model
on a given test dataset. Specifically, it calculates:

1. Overall test set evaluation accuracy.
2. Class-wise accuracy: The accuracy for each class in the dataset.
3. Swap accuracy: The proportion of samples from specific "selected classes"
   that are misclassified as specific "target classes".
4. Unwanted misclassification accuracy: The misclassification accuracy
   for all other unwanted classes.

Parameters:
- `WEIGHTS_PATH`: Path to the file containing the pretrained model weights.
- `TESTSET_PATH`: Path to the file containing the test dataset in `.npz` format.
- `MODEL_PATH`: Path to the file containing the model architecture in JSON format.
- `SELECTED_CLASSES`: List of classes for which swap accuracy is evaluated.
- `TARGET_CLASSES`: List of target classes for swap misclassification evaluation.
"""

WEIGHTS_PATH = "examples/committee_weights_CIFAR10_IID_global_fedavg_label_flip_33_perc/latest_global_model"
TESTSET_PATH = "datasets/cifar10_20_0/testset.npz"
MODEL_PATH = "models/deep_model"

SELECTED_CLASSES = [0]
TARGET_CLASSES = [8]

testset = du.create_composite_dataset_from_paths([TESTSET_PATH], lazy_loading=True , batch_size=32, show_info=False)

with open(MODEL_PATH, "r") as file:
    model_architecture_str = file.read()
	
model = tensorflow_keras.models.model_from_json(model_architecture_str)
model.summary()

with open(WEIGHTS_PATH, "r") as f:
    model_info = json.load(f)

model_weights = model_info[AggregatedModel.WEIGHTS]
weights_list = [np.array(arr) for arr in model_weights]

model.set_weights(weights_list)

sgd = tensorflow_keras.optimizers.SGD(learning_rate=0.001, momentum=0.9, weight_decay=0.001)
model.compile(optimizer=sgd, loss="sparse_categorical_crossentropy", metrics=["accuracy"])

print("----- Total testset evaluation -----")
model.evaluate(testset, verbose=1)

print("----- Class accuracy -----")
y_true = []
y_pred = []

for batch in testset:
    images, labels = batch
    predictions = model.predict(images, verbose=0)
    y_true.extend(labels.numpy())
    y_pred.extend(np.argmax(predictions, axis=1))

# Convert lists to numpy arrays for easier processing
y_true = np.array(y_true)
y_pred = np.array(y_pred)

# Calculate accuracy per class without sklearn
num_classes = len(np.unique(y_true))
for i in range(num_classes):
    # Mask for current class
    mask = (y_true == i)
    # Count correct predictions for the current class
    correct_predictions = np.sum(y_pred[mask] == y_true[mask])
    # Total number of samples in the current class
    total_samples = np.sum(mask)
    # Calculate accuracy for the current class
    class_accuracy = correct_predictions / total_samples if total_samples > 0 else 0
    print(f"Accuracy for class {i}: {class_accuracy:.4f}")

# Calculate misclassification accuracy for swapped classes
print("----- Swap accuracy -----")
for selected_class, target_class in zip(SELECTED_CLASSES, TARGET_CLASSES):
    # Mask for samples with the selected class label
    selected_mask = (y_true == selected_class)
    # Count how many of these are misclassified as the target class
    misclassified_as_target = np.sum((y_pred[selected_mask] == target_class))
    # Total number of samples in the selected class
    total_selected = np.sum(selected_mask)
    # Calculate misclassification accuracy
    misclassification_accuracy = misclassified_as_target / total_selected if total_selected > 0 else 0
    print(f"Accuracy of class {selected_class} misclassified as {target_class}: {misclassification_accuracy:.4f}")

    for i in range(num_classes):
        if i != target_class and i != selected_class:
            # Mask for samples with the target class label
            mask = (y_true == selected_class)
            # Count how many of these are misclassified as the selected class
            misclassified = np.sum((y_pred[mask] == i))
            misclassification_accuracy = misclassified / total_selected if total_selected > 0 else 0
            print(f"Accuracy of class {selected_class} misclassified as unwanted {i}: {misclassification_accuracy:.4f}")


from abc import ABC, abstractmethod
import gc
from numpy import median, asarray, array
from typing import List, Tuple
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy import FedAvg
from flwr.server.client_proxy import ClientProxy
from flwr.common import parameters_to_ndarrays, ndarrays_to_parameters, FitRes, Status
from tensorflow import convert_to_tensor

class MockClientProxy(ClientProxy):
	def __init__(self, cid: str):
		self.cid = cid

	def get_parameters(self):
		pass

	def fit(self):
		pass

	def evaluate(self):
		pass

	def get_properties(self):
		pass

	def reconnect(self):
		pass

class AggregateClass(ABC):
	
	@abstractmethod
	def aggregate(self):
		raise NotImplementedError

class MeanAgg(AggregateClass):
	
	def aggregate(self, updates: list):
		# This will handle the aggregation of gradients by averaging them
		if type(updates) != list:
			raise TypeError("MeanAgg aggregate method: updates must be a list")
		elif len(updates) == 0:
			raise ValueError("MeanAgg aggregate method: No updates found")

		for i in range(len(updates)):
			for g, gradient in enumerate(updates[i]):
				updates[i][g] = convert_to_tensor(gradient)

		# Sum all the gradients together
		for i in range(1, len(updates)):
			gradients = updates[i]
			for g, gradient in enumerate(gradients):
				updates[0][g] += gradient

		# Average gradients
		for g, gradient in enumerate(updates[0]):
			updates[0][g] /= len(updates)

		return [grad.numpy().tolist() for grad in updates[0]]
		
class FlowerFedAvg(AggregateClass):
	
	def __init__(self):
		self._fedavg_instance = FedAvg()
		
	def aggregate(self, updates: list):
		# This will handle the aggregation of weights based on a strategy
		if type(updates) != list or any(len(update) != 2 for update in updates):
			raise TypeError("StrategyAgg aggregate method: updates must be a list of (weights, num_samples)")
		elif len(updates) == 0:
			raise ValueError("StrategyAgg aggregate method: No updates found")

		mock_results: List[Tuple[ClientProxy, FitRes]] = []
		for i, (w, num_samples) in enumerate(updates):
			parameters = ndarrays_to_parameters(w)

			# Memory optimization
			w.clear()

			client_proxy = MockClientProxy(cid=str(i))
			fit_res = FitRes(Status(None, None), parameters, num_samples, None)
			mock_results.append((client_proxy, fit_res))

		aggregated_parameters, _ = self._fedavg_instance.aggregate_fit(
			server_round=1,
			results=mock_results,
			failures=[]
		)

		aggregated_parameters = parameters_to_ndarrays(aggregated_parameters)
		weights = [arr.tolist() for arr in aggregated_parameters]

		# Clean up
		del mock_results, aggregated_parameters, _, client_proxy, fit_res
		gc.collect()

		return weights
		
class MedianAgg(AggregateClass):
	
	def aggregate(self, updates: list):
		# This will handle the aggregation of gradients by taking the median
		if type(updates) != list:
			raise TypeError("MedianAgg aggregate method: updates must be a list")
		elif len(updates) == 0:
			raise ValueError("MedianAgg aggregate method: No updates found")
		
		# Convert all updates to numpy arrays
		for i in range(len(updates)):
			for g, gradient in enumerate(updates[i]):
				updates[i][g] = array(gradient)

		# Take the median of all the gradients
		median_result = [median(asarray(layer), axis=0) for layer in zip(*updates)]

		return [grad.tolist() for grad in median_result]
	
class TrimmedMeanAgg(AggregateClass):
	
	def __init__(self, trimming_percentage: float):
		if type(trimming_percentage) not in [int, float]:
			raise TypeError("TrimmedMeanAgg __init__ method: trimming_percentage must be a float")
		elif trimming_percentage < 0 or trimming_percentage >= 0.5:
			raise ValueError("TrimmedMeanAgg __init__ method: trimming_percentage must be between 0 and 0.5")
		self._trimming_percentage = trimming_percentage

	def aggregate(self, updates: list):
		# This will handle the aggregation of gradients by taking the trimmed mean
		if type(updates) != list:
			raise TypeError("TrimmedMeanAgg aggregate method: updates must be a list")
		elif len(updates) == 0:
			raise ValueError("TrimmedMeanAgg aggregate method: No updates found")
		
		# Convert all updates to numpy arrays
		for i in range(len(updates)):
			for g, gradient in enumerate(updates[i]):
				updates[i][g] = array(gradient)

		# Calculate the number of elements to trim
		num_elements_to_trim = int(len(updates) * self._trimming_percentage)

		if num_elements_to_trim > 0:
			if len(updates[num_elements_to_trim : -num_elements_to_trim]) == 0:
				raise ValueError("TrimmedMeanAgg aggregate method: No remaining elements after trimming")
			
			# Take the median of all the gradients
			trimmed_mean_result = [median(asarray(layer)[num_elements_to_trim : -num_elements_to_trim], axis=0) for layer in zip(*updates)]
		
		else:
			trimmed_mean_result = [median(asarray(layer), axis=0) for layer in zip(*updates)]

		return [grad.tolist() for grad in trimmed_mean_result]

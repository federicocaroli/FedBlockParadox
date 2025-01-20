import threading, numpy as np, datetime, setproctitle, time, os.path, string, json

from multiprocessing import Process, current_process
from multiprocessing.synchronize import BoundedSemaphore as BoundedSemaphoreClass
from scipy.spatial.distance import cdist

from . import utils
from .constants import DIRECTORY_WHERE_TO_STORE_TMP_FILES

class PassWeightsValidation():
	def __init__(self, validation_semaphore: BoundedSemaphoreClass, min_num_of_updates_between_aggregations: int, max_num_of_updates_between_aggregations: int, count_down_timer_to_start_aggregation: float, list_of_initial_validators: list, verbose: int = 0) -> None:
		if type(min_num_of_updates_between_aggregations) != int or type(max_num_of_updates_between_aggregations) != int or type(count_down_timer_to_start_aggregation) not in [int, float] or type(list_of_initial_validators) != list or type(verbose) != int or isinstance(validation_semaphore, BoundedSemaphoreClass) is False:
			raise TypeError("PassWeightsValidation")
		elif min_num_of_updates_between_aggregations <= 0 or max_num_of_updates_between_aggregations < min_num_of_updates_between_aggregations or len(list_of_initial_validators) == 0 or count_down_timer_to_start_aggregation < 0:
			raise ValueError("PassWeightsValidation")

		self._validation_semaphore = validation_semaphore

		self._min_num_of_updates_between_aggregations = min_num_of_updates_between_aggregations
		self._max_num_of_updates_between_aggregations = max_num_of_updates_between_aggregations
		self._count_down_timer_to_start_aggregation = count_down_timer_to_start_aggregation
		self._list_of_validators = list_of_initial_validators
		self._verbose = verbose

		self._state_variables_lock = threading.Lock()
		self._aggregation_is_ready = False
		self._timer_to_start_validation = None		

		self._updates_tables_lock = threading.Lock()
		self._count_of_completed_positive_validations = 0
		self._updates_validation_scores = {}
		self._honest_updates_and_aggregated_scores = {}

	def is_aggregation_ready(self) -> bool:
		'''
		Returns whether the aggregation is ready, based on the number of completed updates.
		'''
		with self._state_variables_lock:
			return self._aggregation_is_ready
	
	def get_honest_updates_and_aggregated_scores(self) -> dict:

		'''
		Returns the recorded update scores.
		'''
		with self._updates_tables_lock:
			return self._honest_updates_and_aggregated_scores

	def must_stop_accepting_new_updates(self) -> bool:
		with self._state_variables_lock:
			with self._updates_tables_lock:
				return self._aggregation_is_ready

	def _stop_accepting_new_update_validations(self) -> None:
		with self._state_variables_lock:
			with self._updates_tables_lock:
				if self._count_of_completed_positive_validations != self._max_num_of_updates_between_aggregations:
					if self._aggregation_is_ready or self._count_of_completed_positive_validations < self._min_num_of_updates_between_aggregations or self._count_of_completed_positive_validations > self._max_num_of_updates_between_aggregations:
						raise ValueError(f"PassWeightsValidation _stop_accepting_new_update_validations method. Is aggregation ready: {self._aggregation_is_ready}. Number of honest updates and aggregated scores: {self._count_of_completed_positive_validations}. Min number of updates between aggregations: {self._min_num_of_updates_between_aggregations}. Max number of updates between aggregations: {self._max_num_of_updates_between_aggregations}.")

					self._aggregation_is_ready = True

	def validate_update(self, model_architecture = None, model_info = None) -> tuple[float, bool] | None:
		'''
		Always returns the update as valid with its validation score, regardless of any thresholds or validators.
		Aggregation readiness is not affected by validation; it is only dependent on the number of completed updates.
		'''
		with self._state_variables_lock:
			with self._updates_tables_lock:
				if self._aggregation_is_ready or self._count_of_completed_positive_validations >= self._max_num_of_updates_between_aggregations:
					return None
		
		# Always mark the update as valid (True)
		return 1.0, True

	def handle_new_validation_result(self, positive_validation: bool, accuracy: float, update_name: str, updater_id: int, validator_id: int) -> tuple[bool, int, float, bool] | None:
		'''
		Handles the new validation result and tracks the score. If the number of updates 
		reaches the required threshold for aggregation, it triggers the aggregation.
		'''
		if type(positive_validation)!=bool or type(accuracy) not in [float, int] or type(update_name) != str or type(updater_id) != int or type(validator_id) != int:
			raise TypeError("PassWeightsValidation handle_new_validation_result method")
		elif validator_id not in self._list_of_validators:
			raise ValueError("PassWeightsValidation handle_new_validation_result method")
		elif positive_validation != (accuracy >= 1.0):
			raise ValueError("PassWeightsValidation handle_new_validation_result method")
		
		result = None

		with self._state_variables_lock:
			if self._aggregation_is_ready:
				return None

			with self._updates_tables_lock:
				if self._count_of_completed_positive_validations >= self._max_num_of_updates_between_aggregations:
					return None

				if update_name not in self._updates_validation_scores:
					self._updates_validation_scores[update_name] = {}
				elif validator_id in self._updates_validation_scores[update_name]:
					raise Exception(f"PassWeightsValidation handle_new_validation_result method. Impossible to handle new validation result. Validator ID is already in the list of validation results. Update Name: {update_name}. Validator ID: {validator_id}. Known Validators: {self._updates_validation_scores[update_name].keys()}")

				# Store the validation score
				self._updates_validation_scores[update_name][validator_id] = accuracy

				if len(self._updates_validation_scores[update_name]) == len(self._list_of_validators):
					self._count_of_completed_positive_validations += 1

					self._honest_updates_and_aggregated_scores[update_name] = 1.0
					result = True, len(self._updates_validation_scores[update_name]), 1.0, self._count_of_completed_positive_validations == self._min_num_of_updates_between_aggregations
				
					# Track the number of completed updates
					if self._count_of_completed_positive_validations == self._min_num_of_updates_between_aggregations:
						if self._timer_to_start_validation is None:
							self._timer_to_start_validation = threading.Timer(self._count_down_timer_to_start_aggregation, self._stop_accepting_new_update_validations)
							self._timer_to_start_validation.start()
						else:
							raise Exception("PassWeightsValidation handle_new_validation_result method. Timer to start validation is already set.")
						
					elif self._count_of_completed_positive_validations == self._max_num_of_updates_between_aggregations:
						if self._timer_to_start_validation is not None:
							self._timer_to_start_validation.cancel()
						
						self._aggregation_is_ready = True

					elif self._count_of_completed_positive_validations > self._max_num_of_updates_between_aggregations:
						raise Exception("PassWeightsValidation handle_new_validation_result method. Impossible to handle new validation result. Number of completed validations is greater than the maximum number of updates between aggregations")
				
				elif len(self._updates_validation_scores[update_name]) > len(self._list_of_validators):
					raise Exception("PassWeightsValidation handle_new_validation_result method. Impossible to handle new validation result. Number of validators that have validated the update is greater than the number of validators")
				
			return result

	def start_new_round(self, list_of_validators: list, honest_updates_aggregated_scores = None) -> None:
		'''
		Starts a new round of updates, resetting the internal state for the new round.
		'''
		if type(list_of_validators) != list:
			raise TypeError("PassWeightsValidation start_new_round method")
		elif len(list_of_validators) == 0:
			raise ValueError("PassWeightsValidation start_new_round method")

		if self._timer_to_start_validation is not None:
			self._timer_to_start_validation.cancel()

			while self._timer_to_start_validation.is_alive():
				time.sleep(0.1)

		with self._state_variables_lock:
			with self._updates_tables_lock:
				self._updates_validation_scores = {}
				self._honest_updates_and_aggregated_scores = {}
				self._count_of_completed_positive_validations = 0
				self._aggregation_is_ready = False
				self._list_of_validators = list_of_validators
				self._timer_to_start_validation = None				

class PassGradientsValidation():
	def __init__(self, validation_semaphore: BoundedSemaphoreClass, validator_node_id: int, min_num_of_updates_needed_to_start_validation: int, max_num_of_updates_needed_to_start_validation: int, count_down_timer_to_start_validation: float, list_of_initial_validators: list, lazy_validation: bool = True) -> None:
		if type(validator_node_id) != int or type(min_num_of_updates_needed_to_start_validation) != int or type(max_num_of_updates_needed_to_start_validation) != int or type(list_of_initial_validators) != list or type(count_down_timer_to_start_validation) not in [int, float] or type(lazy_validation) != bool or isinstance(validation_semaphore, BoundedSemaphoreClass) is False:
			raise TypeError("PassGradientsValidation")
		elif min_num_of_updates_needed_to_start_validation <= 0 or max_num_of_updates_needed_to_start_validation < min_num_of_updates_needed_to_start_validation or len(list_of_initial_validators) == 0 or count_down_timer_to_start_validation < 0:
			raise ValueError("PassGradientsValidation")
		
		self._validation_semaphore = validation_semaphore

		self._validator_node_id = validator_node_id
		self._min_num_of_updates_needed_to_start_validation = min_num_of_updates_needed_to_start_validation
		self._max_num_of_updates_needed_to_start_validation = max_num_of_updates_needed_to_start_validation
		self._count_down_timer_to_start_validation = count_down_timer_to_start_validation
		self._list_of_update_names_to_validate = []
		self._list_of_update_names_to_validate_lock = threading.Lock()
		self._count_of_updates_to_validate = 0

		self._state_variables_lock = threading.Lock()
		self._must_stop_accepting_new_updates = False											# Boolean to indicate if the validator must stop accepting new updates. It is used when lazy validation is enabled and the validator is waiting for the votes from the validators with lower node_id than its node_id. In this case, the validator must stop accepting new updates and cannot start the validation until it receives the votes from all the validators with lower node_id than its node_id.
		self._is_validation_ready_to_start = False												# Boolean to indicate if the Pass validation is ready to start, that means that the validator has received the minimum number of updates needed to start the validation and the timer to start the validation has expired
		self._is_validation_in_progress = False													# Boolean to indicate if the Pass validation is in progress
		self._is_validation_completed = False													# Boolean to indicate if the Pass validation is completed
		self._aggregation_is_ready = False														# Boolean to indicate if the aggregation of the validation results is ready, that means that the validator has received the validation results from all the validators
		self._timer_to_start_validation = None													# When this timer expires, the validator stops accepting new updates and it is ready to start the validation

		self._validation_results = {}
		self._validation_results_lock = threading.Lock()
		self._list_of_validators = list_of_initial_validators

		self._lazy_validation = lazy_validation													# Boolean to indicate if the validation is lazy. If it is lazy, then the validator will be able to start the validation process only when it has received votes from all the validators with lower node_id than its node_id. So, there is a period of time where the validator doesn't accept new updates but cannot even start the validation process because it is waiting for the votes from the validators with lower node_id than its node_id.

	def get_min_num_of_updates_needed_to_start_validation(self) -> int:
		return self._min_num_of_updates_needed_to_start_validation

	def get_list_of_update_names_to_validate(self) -> list:
		with self._list_of_update_names_to_validate_lock:
			return self._list_of_update_names_to_validate
		
	def get_list_of_validators(self) -> list:
		with self._state_variables_lock:
			return self._list_of_validators
		
	def is_lazy_validation(self) -> bool:
		return self._lazy_validation
	
	def must_stop_accepting_new_updates(self) -> bool:
		with self._state_variables_lock:
			return self._must_stop_accepting_new_updates
	
	def is_validation_ready_to_start(self) -> bool:
		with self._state_variables_lock:
			return self._is_validation_ready_to_start
		
	def is_validation_in_progress(self) -> bool:
		with self._state_variables_lock:
			return self._is_validation_in_progress
		
	def is_validation_completed(self) -> bool:
		with self._state_variables_lock:
			return self._is_validation_completed
		
	def is_aggregation_ready(self) -> bool:
		with self._state_variables_lock:
			return self._aggregation_is_ready
		
	def get_validation_results(self) -> dict:
		with self._validation_results_lock:
			return self._validation_results
	
	def _stop_accepting_new_updates_for_validation(self) -> None:
		with self._state_variables_lock:
			if self._count_of_updates_to_validate != self._max_num_of_updates_needed_to_start_validation:
				if self._is_validation_ready_to_start or self._must_stop_accepting_new_updates or self._is_validation_in_progress or self._is_validation_completed or self._aggregation_is_ready or self._count_of_updates_to_validate < self._min_num_of_updates_needed_to_start_validation:
					raise ValueError(f"PassGradientsValidation _stop_accepting_new_updates_for_validation method. Is validation ready to start: {self._is_validation_ready_to_start}. Must stop accepting new updates: {self._must_stop_accepting_new_updates}. Is validation in progress: {self._is_validation_in_progress}. Is validation completed: {self._is_validation_completed}. Is aggregation ready: {self._aggregation_is_ready}.")

				self._must_stop_accepting_new_updates = True

				# If the validation is not lazy, then the validation is ready to start without waiting for the votes from the validators with lower node_id than its node_id
				if self._lazy_validation is False:
					self._is_validation_ready_to_start = True
				else:
					# If the validation is lazy, then the validator must wait for the votes from the validators with lower node_id than its node_id before starting the validation
					with self._validation_results_lock:
						if all(validator_id in self._validation_results for validator_id in self._list_of_validators if validator_id < self._validator_node_id):
							self._is_validation_ready_to_start = True
			
	def add_update_name_to_validate(self, update_name: str) -> bool | None:
		'''
		Add update name to the list of update names to validate. If the number of update names to validate is equal to the minimum number of updates needed to start validation, then start the timer to start validation and return True. Otherwise, return False. If the update name is not a string, raise a TypeError.
		'''

		if type(update_name) != str:
			raise TypeError("PassGradientsValidation add_update_name_to_validate method")
	
		with self._state_variables_lock:
			if self._must_stop_accepting_new_updates:
				return None

			with self._list_of_update_names_to_validate_lock:
				if self._count_of_updates_to_validate >= self._max_num_of_updates_needed_to_start_validation:
					return None
				
				self._list_of_update_names_to_validate.append(update_name)
				self._count_of_updates_to_validate += 1

				if self._count_of_updates_to_validate == self._min_num_of_updates_needed_to_start_validation:
					
					if self._timer_to_start_validation is None:
						self._timer_to_start_validation = threading.Timer(self._count_down_timer_to_start_validation, self._stop_accepting_new_updates_for_validation)
						self._timer_to_start_validation.start()
					else:
						raise Exception("PassGradientsValidation add_update_name_to_validate method. Timer to start validation is already set.")

					return True

				elif self._count_of_updates_to_validate == self._max_num_of_updates_needed_to_start_validation:
					if self._timer_to_start_validation is not None:
						self._timer_to_start_validation.cancel()
					
					self._must_stop_accepting_new_updates = True

					# If the validation is not lazy, then the validation is ready to start without waiting for the votes from the validators with lower node_id than its node_id
					if self._lazy_validation is False:
						self._is_validation_ready_to_start = True
					else:
						# If the validation is lazy, then the validator must wait for the votes from the validators with lower node_id than its node_id before starting the validation
						with self._validation_results_lock:
							if all(validator_id in self._validation_results for validator_id in self._list_of_validators if validator_id < self._validator_node_id):
								self._is_validation_ready_to_start = True

				elif self._count_of_updates_to_validate > self._max_num_of_updates_needed_to_start_validation:
					raise Exception("PassGradientsValidation add_update_name_to_validate method. Impossible to add update name to validate. Number of involved update names to validate is greater than the maximum number of updates needed to start validation.")

				return False

	def start_new_round(self, list_of_validators: list) -> None:
		if type(list_of_validators) != list:
			raise TypeError("PassGradientsValidation start_new_round method")
		elif len(list_of_validators) == 0:
			raise ValueError("PassGradientsValidation start_new_round method")

		with self._state_variables_lock:
			with self._list_of_update_names_to_validate_lock:
				with self._validation_results_lock:

					if self._is_validation_in_progress:
						raise Exception("PassGradientsValidation reset_validation method. Impossible to reset validation. Validation is in progress.")

					self._must_stop_accepting_new_updates = False
					self._is_validation_ready_to_start = False
					self._is_validation_in_progress = False
					self._is_validation_completed = False
					self._aggregation_is_ready = False
					self._timer_to_start_validation = None

					self._list_of_update_names_to_validate = []
					self._count_of_updates_to_validate = 0
					self._validation_results = {}

					self._list_of_validators = list_of_validators

	def perform_validation(self, updates: list) -> tuple[list[float], list[int]]:
		if type(updates) != list:
			raise TypeError("PassGradientsValidation perform_validation method")
		elif len(updates) == 0:
			raise ValueError("PassGradientsValidation perform_validation method")

		with self._state_variables_lock:
			if self._is_validation_in_progress:
				raise Exception("PassGradientsValidation perform_validation method. Impossible to perform validation. Validation is already in progress.")
			elif self._is_validation_completed:
				raise Exception("PassGradientsValidation perform_validation method. Impossible to perform validation. Validation is already completed.")
			elif not self._is_validation_ready_to_start:
				raise Exception("PassGradientsValidation perform_validation method. Impossible to perform validation. Validation is not ready to start.")
			elif not self._must_stop_accepting_new_updates:
				raise Exception("PassGradientsValidation perform_validation method. Impossible to perform validation. Validator must have already stopped accepting new updates.")

			self._is_validation_in_progress = True
			self._is_validation_ready_to_start = False

		try:
			with self._list_of_update_names_to_validate_lock:
				if self._count_of_updates_to_validate < self._min_num_of_updates_needed_to_start_validation:
					raise Exception("PassGradientsValidation perform_validation method. Impossible to perform validation. Number of involved update names to validate is less than the minimum number of updates needed to start validation.")
				elif self._count_of_updates_to_validate > self._max_num_of_updates_needed_to_start_validation:
					raise Exception("PassGradientsValidation perform_validation method. Impossible to perform validation. Number of involved update names to validate is greater than the maximum number of updates needed to start validation.")
				elif len(updates) != self._count_of_updates_to_validate:
					raise Exception("PassGradientsValidation perform_validation method. Impossible to perform validation. Number of updates is not equal to the number of update names to validate.")
			
				good_idx = np.arange(len(updates))
				scores = [1.0 for _ in range(len(updates))]
				
				return scores, good_idx.tolist()
	   
		except Exception as e:
			raise Exception(f"PassGradientsValidation perform_validation method. Exception: {type(e)}:{str(e)}")
		
		finally:
			with self._state_variables_lock:
				self._is_validation_in_progress = False
				self._is_validation_completed = True

	def handle_new_validation_results(self, results: dict, validator_id: int) -> None:
		if type(results) != dict or type(validator_id) != int:
			raise TypeError("PassGradientsValidation _handle_new_validation_results method")
		
		with self._state_variables_lock:
			if self._aggregation_is_ready:
				raise Exception("PassGradientsValidation _handle_new_validation_results method. Impossible to handle new validation results. Aggregation is already ready.")
			
			with self._validation_results_lock:
				if validator_id not in self._list_of_validators:
					raise Exception("PassGradientsValidation _handle_new_validation_results method. Impossible to handle new validation results. Validator ID is not in the list of validators.")
				elif validator_id in self._validation_results:
					raise Exception("PassGradientsValidation _handle_new_validation_results method. Impossible to handle new validation results. Validator ID is already in the list of validation results.")
				
				self._validation_results[validator_id] = results
		
				# If the number of validation results is equal to the number of validators, then the aggregation is ready
				if len(self._validation_results) == len(self._list_of_validators):
					self._aggregation_is_ready = True
				
				# If the validation is lazy, then the validator must wait for the votes from the validators with lower node_id than its node_id before starting the validation
				elif self._lazy_validation and all(validator_id in self._validation_results for validator_id in self._list_of_validators if validator_id < self._validator_node_id):
					# To set the validation ready to start, the validator must have stopped accepting new updates, the validator must have not started or completed the validation yet
					if self._must_stop_accepting_new_updates and not self._is_validation_ready_to_start and not self._is_validation_in_progress and not self._is_validation_completed:
						self._is_validation_ready_to_start = True

class KrumValidation():
	def __init__(self, validation_semaphore: BoundedSemaphoreClass, validator_node_id: int, min_num_of_updates_needed_to_start_validation: int, max_num_of_updates_needed_to_start_validation: int, num_of_updates_to_validate_negatively: int, count_down_timer_to_start_validation: float, distance_function_type: str, list_of_initial_validators: list, lazy_validation: bool = True) -> None:
		if type(validator_node_id) != int or type(min_num_of_updates_needed_to_start_validation) != int or type(num_of_updates_to_validate_negatively) != int or type(distance_function_type) != str or type(list_of_initial_validators) != list or type(count_down_timer_to_start_validation) not in [int, float] or type(lazy_validation) != bool or type(max_num_of_updates_needed_to_start_validation) != int or isinstance(validation_semaphore, BoundedSemaphoreClass) is False:
			raise TypeError("KrumValidation")
		elif min_num_of_updates_needed_to_start_validation <= 0 or num_of_updates_to_validate_negatively <= 0 or min_num_of_updates_needed_to_start_validation < num_of_updates_to_validate_negatively or len(list_of_initial_validators) == 0 or count_down_timer_to_start_validation < 0 or max_num_of_updates_needed_to_start_validation < min_num_of_updates_needed_to_start_validation:
			raise ValueError("KrumValidation")

		self._validation_semaphore = validation_semaphore

		self._validator_node_id = validator_node_id
		self._min_num_of_updates_needed_to_start_validation = min_num_of_updates_needed_to_start_validation
		self._max_num_of_updates_needed_to_start_validation = max_num_of_updates_needed_to_start_validation
		self._num_of_updates_to_validate_negatively = num_of_updates_to_validate_negatively
		self._count_down_timer_to_start_validation = count_down_timer_to_start_validation
		self._distance_function_type = distance_function_type
		self._list_of_update_names_to_validate = []
		self._count_of_updates_to_validate = 0
		self._list_of_update_names_to_validate_lock = threading.Lock()

		self._state_variables_lock = threading.Lock()
		self._must_stop_accepting_new_updates = False											# Boolean to indicate if the validator must stop accepting new updates. It is used when lazy validation is enabled and the validator is waiting for the votes from the validators with lower node_id than its node_id. In this case, the validator must stop accepting new updates and cannot start the validation until it receives the votes from all the validators with lower node_id than its node_id.
		self._is_validation_ready_to_start = False												# Boolean to indicate if the Krum validation is ready to start, that means that the validator has received the minimum number of updates needed to start the validation and the timer to start the validation has expired
		self._is_validation_in_progress = False													# Boolean to indicate if the Krum validation is in progress
		self._is_validation_completed = False													# Boolean to indicate if the Krum validation is completed
		self._aggregation_is_ready = False														# Boolean to indicate if the aggregation of the validation results is ready, that means that the validator has received the validation results from all the validators
		self._timer_to_start_validation = None													# When this timer expires, the validator stops accepting new updates and it is ready to start the validation

		self._validation_results = {}
		self._validation_results_lock = threading.Lock()
		self._list_of_validators = list_of_initial_validators

		self._lazy_validation = lazy_validation													# Boolean to indicate if the validation is lazy. If it is lazy, then the validator will be able to start the validation process only when it has received votes from all the validators with lower node_id than its node_id. So, there is a period of time where the validator doesn't accept new updates but cannot even start the validation process because it is waiting for the votes from the validators with lower node_id than its node_id.

	def get_min_num_of_updates_needed_to_start_validation(self) -> int:
		return self._min_num_of_updates_needed_to_start_validation
	
	def get_num_of_updates_to_validate_negatively(self) -> int:
		return self._num_of_updates_to_validate_negatively

	def get_list_of_update_names_to_validate(self) -> list:
		with self._list_of_update_names_to_validate_lock:
			return self._list_of_update_names_to_validate
		
	def get_list_of_validators(self) -> list:
		with self._state_variables_lock:
			return self._list_of_validators
		
	def is_lazy_validation(self) -> bool:
		return self._lazy_validation
	
	def must_stop_accepting_new_updates(self) -> bool:
		with self._state_variables_lock:
			return self._must_stop_accepting_new_updates
	
	def is_validation_ready_to_start(self) -> bool:
		with self._state_variables_lock:
			return self._is_validation_ready_to_start
		
	def is_validation_in_progress(self) -> bool:
		with self._state_variables_lock:
			return self._is_validation_in_progress
		
	def is_validation_completed(self) -> bool:
		with self._state_variables_lock:
			return self._is_validation_completed
		
	def is_aggregation_ready(self) -> bool:
		with self._state_variables_lock:
			return self._aggregation_is_ready
		
	def get_validation_results(self) -> dict:
		with self._validation_results_lock:
			return self._validation_results
	
	def _stop_accepting_new_updates_for_validation(self) -> None:
		with self._state_variables_lock:
			if self._count_of_updates_to_validate != self._max_num_of_updates_needed_to_start_validation:
				if self._is_validation_ready_to_start or self._must_stop_accepting_new_updates or self._is_validation_in_progress or self._is_validation_completed or self._aggregation_is_ready or self._count_of_updates_to_validate < self._min_num_of_updates_needed_to_start_validation:
					raise ValueError(f"KrumValidation _stop_accepting_new_updates_for_validation method. Is validation ready to start: {self._is_validation_ready_to_start}. Must stop accepting new updates: {self._must_stop_accepting_new_updates}. Is validation in progress: {self._is_validation_in_progress}. Is validation completed: {self._is_validation_completed}. Is aggregation ready: {self._aggregation_is_ready}.")

				self._must_stop_accepting_new_updates = True

				# If the validation is not lazy, then the validation is ready to start without waiting for the votes from the validators with lower node_id than its node_id
				if self._lazy_validation is False:
					self._is_validation_ready_to_start = True
				else:
					# If the validation is lazy, then the validator must wait for the votes from the validators with lower node_id than its node_id before starting the validation
					with self._validation_results_lock:
						if all(validator_id in self._validation_results for validator_id in self._list_of_validators if validator_id < self._validator_node_id):
							self._is_validation_ready_to_start = True
			
	def add_update_name_to_validate(self, update_name: str) -> bool | None:
		'''
		Add update name to the list of update names to validate. If the number of update names to validate is equal to the minimum number of updates needed to start validation, then start the timer to start validation and return True. Otherwise, return False. If the update name is not a string, raise a TypeError.
		'''

		if type(update_name) != str:
			raise TypeError("KrumValidation add_update_name_to_validate method")
	
		with self._state_variables_lock:
			if self._must_stop_accepting_new_updates:
				return None

			with self._list_of_update_names_to_validate_lock:
				if self._count_of_updates_to_validate >= self._max_num_of_updates_needed_to_start_validation:
					return None

				self._list_of_update_names_to_validate.append(update_name)
				self._count_of_updates_to_validate += 1

				if self._count_of_updates_to_validate == self._min_num_of_updates_needed_to_start_validation:
					
					if self._timer_to_start_validation is None:
						self._timer_to_start_validation = threading.Timer(self._count_down_timer_to_start_validation, self._stop_accepting_new_updates_for_validation)
						self._timer_to_start_validation.start()
					else:
						raise Exception("KrumValidation add_update_name_to_validate method. Timer to start validation is already set.")

					return True

				elif self._count_of_updates_to_validate == self._max_num_of_updates_needed_to_start_validation:
					if self._timer_to_start_validation is not None:
						self._timer_to_start_validation.cancel()
					
					self._must_stop_accepting_new_updates = True

					# If the validation is not lazy, then the validation is ready to start without waiting for the votes from the validators with lower node_id than its node_id
					if self._lazy_validation is False:
						self._is_validation_ready_to_start = True
					else:
						# If the validation is lazy, then the validator must wait for the votes from the validators with lower node_id than its node_id before starting the validation
						with self._validation_results_lock:
							if all(validator_id in self._validation_results for validator_id in self._list_of_validators if validator_id < self._validator_node_id):
								self._is_validation_ready_to_start = True
					
				elif self._count_of_updates_to_validate > self._max_num_of_updates_needed_to_start_validation:
					raise Exception("KrumValidation add_update_name_to_validate method. Impossible to add update name to validate. Number of involved update names to validate is greater than the maximum number of updates needed to start validation.")

				return False

	def start_new_round(self, list_of_validators: list) -> None:
		if type(list_of_validators) != list:
			raise TypeError("KrumValidation start_new_round method")
		elif len(list_of_validators) == 0:
			raise ValueError("KrumValidation start_new_round method")

		with self._state_variables_lock:
			with self._list_of_update_names_to_validate_lock:
				with self._validation_results_lock:

					if self._is_validation_in_progress:
						raise Exception("KrumValidation reset_validation method. Impossible to reset validation. Validation is in progress.")

					self._must_stop_accepting_new_updates = False
					self._is_validation_ready_to_start = False
					self._is_validation_in_progress = False
					self._is_validation_completed = False
					self._aggregation_is_ready = False
					self._timer_to_start_validation = None

					self._list_of_update_names_to_validate = []
					self._count_of_updates_to_validate = 0
					self._validation_results = {}

					self._list_of_validators = list_of_validators

	def perform_validation(self, updates: list) -> tuple[list[float], list[int]]:
		if type(updates) != list:
			raise TypeError("KrumValidation perform_validation method")
		elif len(updates) == 0:
			raise ValueError("KrumValidation perform_validation method")

		with self._state_variables_lock:
			if self._is_validation_in_progress:
				raise Exception("KrumValidation perform_validation method. Impossible to perform validation. Validation is already in progress.")
			elif self._is_validation_completed:
				raise Exception("KrumValidation perform_validation method. Impossible to perform validation. Validation is already completed.")
			elif not self._is_validation_ready_to_start:
				raise Exception("KrumValidation perform_validation method. Impossible to perform validation. Validation is not ready to start.")
			elif not self._must_stop_accepting_new_updates:
				raise Exception("KrumValidation perform_validation method. Impossible to perform validation. Validator must have already stopped accepting new updates.")

			self._is_validation_in_progress = True
			self._is_validation_ready_to_start = False

		try:
			with self._list_of_update_names_to_validate_lock:
				if self._count_of_updates_to_validate < self._min_num_of_updates_needed_to_start_validation:
					raise Exception("KrumValidation perform_validation method. Impossible to perform validation. Number of involved update names to validate is less than the minimum number of updates needed to start validation.")
				elif self._count_of_updates_to_validate > self._max_num_of_updates_needed_to_start_validation:
					raise Exception("KrumValidation perform_validation method. Impossible to perform validation. Number of involved update names to validate is greater than the maximum number of updates needed to start validation.")
				elif len(updates) != self._count_of_updates_to_validate:
					raise Exception("KrumValidation perform_validation method. Impossible to perform validation. Number of updates is not equal to the number of update names to validate.")
			
				with self._validation_semaphore:
					filename = os.path.join(DIRECTORY_WHERE_TO_STORE_TMP_FILES, ''.join(np.random.choice(list(string.ascii_lowercase + string.digits), size=24)) + ".json")

					validation_process = Process(target=KrumValidation._validation, args=(True, updates, self._num_of_updates_to_validate_negatively, self._distance_function_type, filename), name= f"FedBlockSimulator - validator_{self._validator_node_id}_krum_validation", daemon= True)
					validation_process.start()
					validation_process.join()

				with open(filename, "r") as file:
					result = json.load(file)

				os.remove(filename)

				if type(result) != dict:
					raise Exception("Result from subprocess is not dict")
				elif "error" in result:
					raise Exception(f"Error while performing krum validation. Error: {result['error']}")

				return result["scores"], result["good_indexes"]
	   
		except Exception as e:
			raise Exception(f"KrumValidation perform_validation method. Exception: {type(e)}:{str(e)}")
		
		finally:
			with self._state_variables_lock:
				self._is_validation_in_progress = False
				self._is_validation_completed = True

	def handle_new_validation_results(self, results: dict, validator_id: int) -> None:
		if type(results) != dict or type(validator_id) != int:
			raise TypeError("KrumValidation _handle_new_validation_results method")
		
		with self._state_variables_lock:
			if self._aggregation_is_ready:
				raise Exception("KrumValidation _handle_new_validation_results method. Impossible to handle new validation results. Aggregation is already ready.")
			
			with self._validation_results_lock:
				if validator_id not in self._list_of_validators:
					raise Exception("KrumValidation _handle_new_validation_results method. Impossible to handle new validation results. Validator ID is not in the list of validators.")
				elif validator_id in self._validation_results:
					raise Exception("KrumValidation _handle_new_validation_results method. Impossible to handle new validation results. Validator ID is already in the list of validation results.")
				
				self._validation_results[validator_id] = results
		
				# If the number of validation results is equal to the number of validators, then the aggregation is ready
				if len(self._validation_results) == len(self._list_of_validators):
					self._aggregation_is_ready = True
				
				# If the validation is lazy, then the validator must wait for the votes from the validators with lower node_id than its node_id before starting the validation
				elif self._lazy_validation and all(validator_id in self._validation_results for validator_id in self._list_of_validators if validator_id < self._validator_node_id):
					# To set the validation ready to start, the validator must have stopped accepting new updates, the validator must have not started or completed the validation yet
					if self._must_stop_accepting_new_updates and not self._is_validation_ready_to_start and not self._is_validation_in_progress and not self._is_validation_completed:
						self._is_validation_ready_to_start = True

	@staticmethod
	def _validation(is_subprocess: bool, updates: list, num_of_updates_to_validate_negatively: int, distance_function_type: str, filename: str | None = None):
		result = None
		
		try:
			if type(is_subprocess) != bool or type(updates) != list or type(num_of_updates_to_validate_negatively) != int or type(distance_function_type) != str or (is_subprocess and (filename is None or type(filename) != str)):
				raise TypeError("KrumValidation _validation method")

			if is_subprocess:
				setproctitle.setproctitle(f"{current_process().name}")

			updates = [KrumValidation._flatten_list(update) for update in updates]
			scores, good_indexes = KrumValidation._krum(distance_function_type, updates, num_of_updates_to_validate_negatively)

			result = {"scores": scores, "good_indexes": good_indexes}

		except Exception as e:
			if is_subprocess:
				result = {"error": f"{type(e)}:{str(e)}"}
			else:
				raise e

		if is_subprocess:
			with open(filename, "w") as file:
				json.dump(result, file)
		else:
			return result

	@staticmethod
	def _flatten_list(nested_list: list) -> list:
		if type(nested_list) != list:
			raise TypeError("KrumValidation _flatten_list method")
		
		flattened = []
		for item in nested_list:
			if isinstance(item, list):
				flattened.extend(KrumValidation._flatten_list(item))
			else:
				flattened.append(item)
		return flattened

	# KRUM Algorithm Implementation
	@staticmethod
	def _get_krum_scores(distance_function_type, X, groupsize):
		krum_scores = np.zeros(len(X))
		distances = cdist(X, X, metric=distance_function_type)
		groupsize = max(1, groupsize)
		for i in range(len(X)):
			krum_scores[i] = np.sum(np.sort(distances[i])[1:(groupsize+1)])
		return krum_scores
	
	@staticmethod
	def _krum(distance_function_type, deltas, clip) -> tuple[list[float], list[int]]:
		n = len(deltas)
		deltas = np.array(deltas)
		scores = KrumValidation._get_krum_scores(distance_function_type, deltas, n-clip-2)
		good_idx = np.argpartition(scores, n - clip)[:(n - clip)]

		print(f"{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')} Available scores: {scores}. Selected Indices of Reliable Updates: {good_idx}")

		if 0 in scores:
			print(f"{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')} Warning: Krum algorithm has compute one or more scores equal to 0. This happens when the number of honest updates involved in the validation process (total number of updates involved - number of updates to consider negatively) is equal or lower than 2.")

		return scores.tolist(), good_idx.tolist()

class LocalDatasetValidation():
	def __init__(self, validation_semaphore: BoundedSemaphoreClass, dataset_quanta_paths: list, min_update_validation_score_first_round: float, min_num_of_updates_between_aggregations: int, max_num_of_updates_between_aggregations: int, count_down_timer_to_start_aggregation: float, list_of_initial_validators: list, positive_threshold_to_pass_validation: int, lazy_loading_of_validation_set: bool, batch_size: int, verbose: int = 0, testing_ops: bool = False) -> None:
		if type(min_update_validation_score_first_round) not in [float, int] or type(dataset_quanta_paths) != list or type(batch_size) != int or type(min_num_of_updates_between_aggregations) != int or type(max_num_of_updates_between_aggregations) != int or type(count_down_timer_to_start_aggregation) not in [float, int] or type(lazy_loading_of_validation_set) != bool or type(verbose) != int or type(testing_ops) != bool or type(list_of_initial_validators) != list or type(positive_threshold_to_pass_validation) != int or isinstance(validation_semaphore, BoundedSemaphoreClass) is False:
			raise TypeError("LocalDatasetValidation")
		elif min_update_validation_score_first_round < 0 or min_num_of_updates_between_aggregations <= 0 or max_num_of_updates_between_aggregations < min_num_of_updates_between_aggregations or len(list_of_initial_validators) == 0 or positive_threshold_to_pass_validation < 0 or count_down_timer_to_start_aggregation < 0 or batch_size <= 0:
			raise ValueError("LocalDatasetValidation")

		self._validation_semaphore = validation_semaphore

		self._min_update_validation_score = min_update_validation_score_first_round
		self._min_num_of_updates_between_aggregations = min_num_of_updates_between_aggregations
		self._max_num_of_updates_between_aggregations = max_num_of_updates_between_aggregations
		self._count_down_timer_to_start_aggregation = count_down_timer_to_start_aggregation
		self._positive_threshold_to_pass_validation = positive_threshold_to_pass_validation
		self._lazy_loading_of_validation_set = lazy_loading_of_validation_set
		self._dataset_quanta_paths = dataset_quanta_paths
		self._verbose = verbose
		self._testing_ops = testing_ops
		self._batch_size = batch_size
		
		self._state_variables_lock = threading.Lock()
		self._aggregation_is_ready = False
		self._timer_to_start_validation = None

		self._updates_validation_scores = {}
		self._honest_updates_and_aggregated_scores = {}
		self._count_of_completed_positive_validations = 0
		self._updates_tables_lock = threading.Lock()

		self._list_of_validators = list_of_initial_validators

	def get_min_update_validation_score(self) -> int:
		return self._min_update_validation_score
		
	def is_aggregation_ready(self) -> bool:
		with self._state_variables_lock:
			return self._aggregation_is_ready
	
	def must_stop_accepting_new_updates(self) -> bool:
		with self._state_variables_lock:
			with self._updates_tables_lock:
				return self._aggregation_is_ready

	def get_honest_updates_and_aggregated_scores(self) -> dict:
		with self._updates_tables_lock:
			return self._honest_updates_and_aggregated_scores

	def _stop_accepting_new_update_validations(self) -> None:
		with self._state_variables_lock:
			with self._updates_tables_lock:
				if self._count_of_completed_positive_validations != self._max_num_of_updates_between_aggregations:
					if self._aggregation_is_ready or self._count_of_completed_positive_validations < self._min_num_of_updates_between_aggregations or self._count_of_completed_positive_validations > self._max_num_of_updates_between_aggregations:
						raise ValueError(f"LocalDatasetValidation _stop_accepting_new_update_validations method. Is aggregation ready: {self._aggregation_is_ready}. Number of honest updates and aggregated scores: {self._count_of_completed_positive_validations}. Min number of updates between aggregations: {self._min_num_of_updates_between_aggregations}. Max number of updates between aggregations: {self._max_num_of_updates_between_aggregations}.")

					self._aggregation_is_ready = True

	def validate_update(self, model_architecture: dict, model_weights: list) -> tuple[float, bool] | None:
	
		if type(model_architecture) != dict or type(model_weights) != list:
			raise TypeError("LocalDatasetValidation validate_update method")

		with self._state_variables_lock:
			with self._updates_tables_lock:
				if self._aggregation_is_ready or self._count_of_completed_positive_validations >= self._max_num_of_updates_between_aggregations:
					return None
			
		try:
			if self._testing_ops:
				validation_score = np.random.uniform(0.75, 1.0)
			
			else:
				with self._validation_semaphore:
					filename = os.path.join(DIRECTORY_WHERE_TO_STORE_TMP_FILES, ''.join(np.random.choice(list(string.ascii_lowercase + string.digits), size=24)) + ".json")

					validation_process = Process(target=LocalDatasetValidation._validation, args=(True, self._lazy_loading_of_validation_set, model_architecture, model_weights, self._dataset_quanta_paths, self._batch_size, self._verbose, filename), name= f"FedBlockSimulator - local_dataset_validation", daemon= True)
					validation_process.start()
					validation_process.join()

				with open(filename, "r") as file:
					result = json.load(file)

				os.remove(filename)
				
				if type(result) != dict:
					raise Exception("Result from subprocess is not dict")
				elif "error" in result:
					raise Exception(f"Error while performing local dataset validation. Error: {result['error']}")
				
				validation_score = result["validation_score"]

			with self._state_variables_lock:
				with self._updates_tables_lock:
					if self._aggregation_is_ready or self._count_of_completed_positive_validations >= self._max_num_of_updates_between_aggregations:
						return None

			return validation_score, validation_score >= self._min_update_validation_score
		except Exception as e:
			raise Exception(f"LocalDatasetValidation validate_update method. Exception: {type(e)}:{str(e)}")

	@staticmethod
	def _validation(is_subprocess: bool, lazy_loading: bool, model_architecture: dict, model_weights: list, dataset_quanta_paths: list, batch_size: int, verbose: int, filename: str | None = None):
		result = None
		
		try:
			if type(is_subprocess) != bool or type(lazy_loading) != bool or type(model_architecture) != dict or type(model_weights) != list or type(dataset_quanta_paths) != list or type(batch_size) != int or type(verbose) != int or (is_subprocess and (filename is None or type(filename) != str)):
				raise TypeError("LocalDatasetValidation _validation method")

			if is_subprocess:
				setproctitle.setproctitle(f"{current_process().name}")

			model = utils.build_model_from_architecture_and_weights(model_architecture, model_weights)

			validation_set = utils.get_training_set(dataset_quanta_paths, lazy_loading, batch_size)

			if lazy_loading:
				_, validation_score = model.evaluate(validation_set, verbose=verbose)
			else:
				_, validation_score = model.evaluate(validation_set['img'], validation_set['label'], verbose=verbose)

			result = {"validation_score": validation_score}

		except Exception as e:
			if is_subprocess:
				result = {"error": f"{type(e)}:{str(e)}"}
			else:
				raise e

		if is_subprocess:
			with open(filename, "w") as file:
				json.dump(result, file)

		else:
			return result

	def handle_new_validation_result(self, positive_validation: bool, accuracy: float, update_name: str, updater_id: int, validator_id: int) -> tuple[bool, int, float, bool] | None:
		'''
		Handle new validation result. If the number of validators that have validated the update is equal to the number of validators, then check if the number of validators that have validated the update positively is greater than or equal to the positive threshold to pass validation. If it is, then return True. Otherwise, return False. If the update is uncompleted, then return None.
		If the number of updates completed validated with a number of positive votes greater then the threshold is equal to the number of updates between aggregations, then set the aggregation to be ready.
		'''

		if type(positive_validation) != bool or type(accuracy) not in [float, int] or type(update_name) != str or type(updater_id) != int or type(validator_id) != int:
			raise TypeError("LocalDatasetValidation handle_new_validation_result method")
		elif validator_id not in self._list_of_validators:
			raise ValueError("LocalDatasetValidation handle_new_validation_result method")
		elif positive_validation != (accuracy >= self._min_update_validation_score):
			raise ValueError("LocalDatasetValidation handle_new_validation_result method")

		result = None

		with self._state_variables_lock:
			# It may happen that the aggregation is ready before the validator has received the votes from all the validators. In this case, the validator must reject the new validation results
			if self._aggregation_is_ready:
				return None
				
			with self._updates_tables_lock:
				if self._count_of_completed_positive_validations >= self._max_num_of_updates_between_aggregations:
					return None
				
				if update_name not in self._updates_validation_scores:
					self._updates_validation_scores[update_name] = {}
				elif validator_id in self._updates_validation_scores[update_name]:
					raise Exception(f"LocalDatasetValidation handle_new_validation_result method. Impossible to handle new validation result. Validator ID is already in the list of validation results. Update Name: {update_name}. Validator ID: {validator_id}. Known Validators: {self._updates_validation_scores[update_name].keys()}")
				
				self._updates_validation_scores[update_name][validator_id] = accuracy

				if len(self._updates_validation_scores[update_name]) == len(self._list_of_validators):
					
					list_of_positive_scores = [accuracy for accuracy in self._updates_validation_scores[update_name].values() if accuracy >= self._min_update_validation_score]

					# Check if the update has been validated by all validators and if the number of positive votes is greater than or equal to the positive threshold to pass validation
					if len(list_of_positive_scores) >= self._positive_threshold_to_pass_validation:
						if update_name in self._honest_updates_and_aggregated_scores:
							raise Exception(f"LocalDatasetValidation handle_new_validation_result method. Impossible to handle new validation result. Update Name is already in the list of honest updates and aggregated scores. Update Name: {update_name}. Known Updates: {self._honest_updates_and_aggregated_scores.keys()}")
						
						self._count_of_completed_positive_validations += 1
						self._honest_updates_and_aggregated_scores[update_name] = self._aggregate_validation_scores(list_of_positive_scores)
						result = True, len(list_of_positive_scores), self._honest_updates_and_aggregated_scores[update_name], self._count_of_completed_positive_validations == self._min_num_of_updates_between_aggregations

						if self._count_of_completed_positive_validations == self._min_num_of_updates_between_aggregations:
							if self._timer_to_start_validation is None:
								self._timer_to_start_validation = threading.Timer(self._count_down_timer_to_start_aggregation, self._stop_accepting_new_update_validations)
								self._timer_to_start_validation.start()
							else:
								raise Exception("LocalDatasetValidation handle_new_validation_result method. Timer to start validation is already set.")
						
						elif self._count_of_completed_positive_validations == self._max_num_of_updates_between_aggregations:
							if self._timer_to_start_validation is not None:
								self._timer_to_start_validation.cancel()

							self._aggregation_is_ready = True

						elif self._count_of_completed_positive_validations > self._max_num_of_updates_between_aggregations:
							raise Exception("LocalDatasetValidation handle_new_validation_result method. Impossible to handle new validation result. Number of completed validations is greater than the maximum number of updates between aggregations")

					else:
						result = False, len(list_of_positive_scores), 0, False
				
				elif len(self._updates_validation_scores[update_name]) > len(self._list_of_validators):
					raise Exception("LocalDatasetValidation handle_new_validation_result method. Impossible to handle new validation result. Number of validators that have validated the update is greater than the number of validators")
					
			return result
			
	def start_new_round(self, list_of_validators: list, honest_updates_aggregated_scores: list = None) -> None:
		if type(list_of_validators) != list or (honest_updates_aggregated_scores is not None and type(honest_updates_aggregated_scores) != list):
			raise TypeError("LocalDatasetValidation start_new_round method")
		elif len(list_of_validators) == 0 or (honest_updates_aggregated_scores is not None and (len(honest_updates_aggregated_scores) < self._min_num_of_updates_between_aggregations or len(honest_updates_aggregated_scores) > self._max_num_of_updates_between_aggregations)) or (honest_updates_aggregated_scores is not None and not all(type(score) in [int, float] for score in honest_updates_aggregated_scores)):
			raise ValueError("LocalDatasetValidation start_new_round method")	

		if self._timer_to_start_validation is not None:
			self._timer_to_start_validation.cancel()

			while self._timer_to_start_validation.is_alive():
				time.sleep(0.1)

		with self._state_variables_lock:
			with self._updates_tables_lock:
				if honest_updates_aggregated_scores is None:
					self._define_minimum_update_validation_score_for_the_next_round(list(self._honest_updates_and_aggregated_scores.values()))
				else:
					self._define_minimum_update_validation_score_for_the_next_round(honest_updates_aggregated_scores)

				self._aggregation_is_ready = False
				self._updates_validation_scores = {}
				self._count_of_completed_positive_validations = 0
				self._honest_updates_and_aggregated_scores = {}
				self._list_of_validators = list_of_validators
				self._timer_to_start_validation = None

	def _aggregate_validation_scores(self, scores: list):
		"""
		Method to aggregate the scores of the updates. The aggregation is done by taking the average of the scores.
		"""

		if type(scores) != list or not all(type(score) in [int, float] for score in scores):
			raise TypeError("LocalDatasetValidation _aggregate_validation_scores method")
		elif len(scores) == 0:
			raise ValueError("LocalDatasetValidation _aggregate_validation_scores method")

		agg = sum(scores) / len(scores)
		return agg
	
	def _define_minimum_update_validation_score_for_the_next_round(self, updaters_scores: list):

		if type(updaters_scores) != list:
			raise TypeError("LocalDatasetValidation _define_minimum_update_validation_score_for_the_next_round method")
		elif len(updaters_scores) == 0:
			raise ValueError("LocalDatasetValidation _define_minimum_update_validation_score_for_the_next_round method")
				
		# Sort the scores in descending order. I decide to order the scores in descending order because in situations where the num of scores is even the median is the lower central value (Ex: [4,3,2,1] --> Median is 2)
		updaters_scores.sort(reverse=True)

		if self._testing_ops:
			self._min_update_validation_score = updaters_scores[-1]
		else:
			position_index = int(len(updaters_scores) * 0.75)
			self._min_update_validation_score = updaters_scores[position_index]	
			self._min_update_validation_score -= self._min_update_validation_score * 0.05

			if self._min_update_validation_score < 0:
				self._min_update_validation_score = 0	

class TrimmedMeanValidation():
	def __init__(self, validation_semaphore: BoundedSemaphoreClass, validator_node_id: int, min_num_of_updates_needed_to_start_validation: int, max_num_of_updates_needed_to_start_validation: int, trimming_percentage: float, count_down_timer_to_start_validation: float, distance_function_type: str, list_of_initial_validators: list, lazy_validation: bool = True) -> None:
		if type(validator_node_id) != int or type(min_num_of_updates_needed_to_start_validation) != int or type(max_num_of_updates_needed_to_start_validation) != int or type(trimming_percentage) not in [float, int] or type(distance_function_type) != str or type(list_of_initial_validators) != list or type(count_down_timer_to_start_validation) not in [int, float] or type(lazy_validation) != bool or isinstance(validation_semaphore, BoundedSemaphoreClass) is False:
			raise TypeError("TrimmedMeanValidation")
		elif min_num_of_updates_needed_to_start_validation <= 0 or max_num_of_updates_needed_to_start_validation < min_num_of_updates_needed_to_start_validation or trimming_percentage < 0 or trimming_percentage >= 0.5 or len(list_of_initial_validators) == 0 or count_down_timer_to_start_validation < 0:
			raise ValueError("TrimmedMeanValidation")
		
		self._validation_semaphore = validation_semaphore

		self._validator_node_id = validator_node_id
		self._min_num_of_updates_needed_to_start_validation = min_num_of_updates_needed_to_start_validation
		self._max_num_of_updates_needed_to_start_validation = max_num_of_updates_needed_to_start_validation
		self._trimming_percentage = trimming_percentage
		self._count_down_timer_to_start_validation = count_down_timer_to_start_validation
		self._distance_function_type = distance_function_type
		self._list_of_update_names_to_validate = []
		self._list_of_update_names_to_validate_lock = threading.Lock()
		self._count_of_updates_to_validate = 0

		self._state_variables_lock = threading.Lock()
		self._must_stop_accepting_new_updates = False											# Boolean to indicate if the validator must stop accepting new updates. It is used when lazy validation is enabled and the validator is waiting for the votes from the validators with lower node_id than its node_id. In this case, the validator must stop accepting new updates and cannot start the validation until it receives the votes from all the validators with lower node_id than its node_id.
		self._is_validation_ready_to_start = False												# Boolean to indicate if the Krum validation is ready to start, that means that the validator has received the minimum number of updates needed to start the validation and the timer to start the validation has expired
		self._is_validation_in_progress = False													# Boolean to indicate if the Krum validation is in progress
		self._is_validation_completed = False													# Boolean to indicate if the Krum validation is completed
		self._aggregation_is_ready = False														# Boolean to indicate if the aggregation of the validation results is ready, that means that the validator has received the validation results from all the validators
		self._timer_to_start_validation = None													# When this timer expires, the validator stops accepting new updates and it is ready to start the validation

		self._validation_results = {}
		self._validation_results_lock = threading.Lock()
		self._list_of_validators = list_of_initial_validators

		self._lazy_validation = lazy_validation													# Boolean to indicate if the validation is lazy. If it is lazy, then the validator will be able to start the validation process only when it has received votes from all the validators with lower node_id than its node_id. So, there is a period of time where the validator doesn't accept new updates but cannot even start the validation process because it is waiting for the votes from the validators with lower node_id than its node_id.

	def get_min_num_of_updates_needed_to_start_validation(self) -> int:
		return self._min_num_of_updates_needed_to_start_validation
	
	def get_trimming_percentage(self) -> float:
		return self._trimming_percentage

	def get_list_of_update_names_to_validate(self) -> list:
		with self._list_of_update_names_to_validate_lock:
			return self._list_of_update_names_to_validate
		
	def get_list_of_validators(self) -> list:
		with self._state_variables_lock:
			return self._list_of_validators
		
	def is_lazy_validation(self) -> bool:
		return self._lazy_validation
	
	def must_stop_accepting_new_updates(self) -> bool:
		with self._state_variables_lock:
			return self._must_stop_accepting_new_updates
	
	def is_validation_ready_to_start(self) -> bool:
		with self._state_variables_lock:
			return self._is_validation_ready_to_start
		
	def is_validation_in_progress(self) -> bool:
		with self._state_variables_lock:
			return self._is_validation_in_progress
		
	def is_validation_completed(self) -> bool:
		with self._state_variables_lock:
			return self._is_validation_completed
		
	def is_aggregation_ready(self) -> bool:
		with self._state_variables_lock:
			return self._aggregation_is_ready
		
	def get_validation_results(self) -> dict:
		with self._validation_results_lock:
			return self._validation_results
	
	def _stop_accepting_new_updates_for_validation(self) -> None:
		with self._state_variables_lock:
			if self._count_of_updates_to_validate != self._max_num_of_updates_needed_to_start_validation:
				if self._is_validation_ready_to_start or self._must_stop_accepting_new_updates or self._is_validation_in_progress or self._is_validation_completed or self._aggregation_is_ready or self._count_of_updates_to_validate < self._min_num_of_updates_needed_to_start_validation:
					raise ValueError(f"TrimmedMeanValidation _stop_accepting_new_updates_for_validation method. Is validation ready to start: {self._is_validation_ready_to_start}. Must stop accepting new updates: {self._must_stop_accepting_new_updates}. Is validation in progress: {self._is_validation_in_progress}. Is validation completed: {self._is_validation_completed}. Is aggregation ready: {self._aggregation_is_ready}.")

				self._must_stop_accepting_new_updates = True

				# If the validation is not lazy, then the validation is ready to start without waiting for the votes from the validators with lower node_id than its node_id
				if self._lazy_validation is False:
					self._is_validation_ready_to_start = True
				else:
					# If the validation is lazy, then the validator must wait for the votes from the validators with lower node_id than its node_id before starting the validation
					with self._validation_results_lock:
						if all(validator_id in self._validation_results for validator_id in self._list_of_validators if validator_id < self._validator_node_id):
							self._is_validation_ready_to_start = True
			
	def add_update_name_to_validate(self, update_name: str) -> bool | None:
		'''
		Add update name to the list of update names to validate. If the number of update names to validate is equal to the minimum number of updates needed to start validation, then start the timer to start validation and return True. Otherwise, return False. If the update name is not a string, raise a TypeError.
		'''

		if type(update_name) != str:
			raise TypeError("TrimmedMeanValidation add_update_name_to_validate method")
	
		with self._state_variables_lock:
			if self._must_stop_accepting_new_updates:
				return None

			with self._list_of_update_names_to_validate_lock:
				if self._count_of_updates_to_validate >= self._max_num_of_updates_needed_to_start_validation:
					return None

				self._list_of_update_names_to_validate.append(update_name)
				self._count_of_updates_to_validate += 1

				if self._count_of_updates_to_validate == self._min_num_of_updates_needed_to_start_validation:
					
					if self._timer_to_start_validation is None:
						self._timer_to_start_validation = threading.Timer(self._count_down_timer_to_start_validation, self._stop_accepting_new_updates_for_validation)
						self._timer_to_start_validation.start()
					else:
						raise Exception("TrimmedMeanValidation add_update_name_to_validate method. Timer to start validation is already set.")

					return True

				elif self._count_of_updates_to_validate == self._max_num_of_updates_needed_to_start_validation:
					if self._timer_to_start_validation is not None:
						self._timer_to_start_validation.cancel()

					self._must_stop_accepting_new_updates = True

					# If the validation is not lazy, then the validation is ready to start without waiting for the votes from the validators with lower node_id than its node_id
					if self._lazy_validation is False:
						self._is_validation_ready_to_start = True
					else:
						# If the validation is lazy, then the validator must wait for the votes from the validators with lower node_id than its node_id before starting the validation
						with self._validation_results_lock:
							if all(validator_id in self._validation_results for validator_id in self._list_of_validators if validator_id < self._validator_node_id):
								self._is_validation_ready_to_start = True
					
				elif self._count_of_updates_to_validate > self._max_num_of_updates_needed_to_start_validation:
					raise Exception("TrimmedMeanValidation add_update_name_to_validate method. Impossible to add update name to validate. Number of involved update names to validate is greater than the maximum number of updates needed to start validation.")

				return False

	def start_new_round(self, list_of_validators: list) -> None:
		if type(list_of_validators) != list:
			raise TypeError("TrimmedMeanValidation start_new_round method")
		elif len(list_of_validators) == 0:
			raise ValueError("TrimmedMeanValidation start_new_round method")

		with self._state_variables_lock:
			with self._list_of_update_names_to_validate_lock:
				with self._validation_results_lock:

					if self._is_validation_in_progress:
						raise Exception("TrimmedMeanValidation reset_validation method. Impossible to reset validation. Validation is in progress.")

					self._must_stop_accepting_new_updates = False
					self._is_validation_ready_to_start = False
					self._is_validation_in_progress = False
					self._is_validation_completed = False
					self._aggregation_is_ready = False
					self._timer_to_start_validation = None

					self._list_of_update_names_to_validate = []
					self._count_of_updates_to_validate = 0
					self._validation_results = {}

					self._list_of_validators = list_of_validators

	def perform_validation(self, updates: list) -> tuple[list[float], list[int]]:
		if type(updates) != list:
			raise TypeError("TrimmedMeanValidation perform_validation method")

		with self._state_variables_lock:
			if self._is_validation_in_progress:
				raise Exception("TrimmedMeanValidation perform_validation method. Impossible to perform validation. Validation is already in progress.")
			elif self._is_validation_completed:
				raise Exception("TrimmedMeanValidation perform_validation method. Impossible to perform validation. Validation is already completed.")
			elif not self._is_validation_ready_to_start:
				raise Exception("TrimmedMeanValidation perform_validation method. Impossible to perform validation. Validation is not ready to start.")
			elif not self._must_stop_accepting_new_updates:
				raise Exception("TrimmedMeanValidation perform_validation method. Impossible to perform validation. Validator must have already stopped accepting new updates.")

			self._is_validation_in_progress = True
			self._is_validation_ready_to_start = False

		try:
			with self._list_of_update_names_to_validate_lock:
				if self._count_of_updates_to_validate < self._min_num_of_updates_needed_to_start_validation:
					raise Exception("TrimmedMeanValidation perform_validation method. Impossible to perform validation. Number of involved update names to validate is less than the minimum number of updates needed to start validation.")
				elif self._count_of_updates_to_validate > self._max_num_of_updates_needed_to_start_validation:
					raise Exception("TrimmedMeanValidation perform_validation method. Impossible to perform validation. Number of involved update names to validate is greater than the maximum number of updates needed to start validation.")
				elif len(updates) != self._count_of_updates_to_validate :
					raise Exception("TrimmedMeanValidation perform_validation method. Impossible to perform validation. Number of updates is not equal to the number of update names to validate.")
			
				with self._validation_semaphore:
					filename = os.path.join(DIRECTORY_WHERE_TO_STORE_TMP_FILES, ''.join(np.random.choice(list(string.ascii_lowercase + string.digits), size=24)) + ".json")

					validation_process = Process(target=TrimmedMeanValidation._validation, args=(True, updates, self._trimming_percentage, self._distance_function_type, filename), name= f"FedBlockSimulator - trimmed_mean_validation", daemon= True)
					validation_process.start()
					validation_process.join()

				with open(filename, "r") as file:
					result = json.load(file)

				os.remove(filename)

				if type(result) != dict:
					raise Exception("Result from subprocess is not dict")
				elif "error" in result:
					raise Exception(f"Error while performing trimmed mean validation. Error: {result['error']}")
				
				return result["scores"], result["good_indexes"]
	   
		except Exception as e:
			raise Exception(f"TrimmedMeanValidation perform_validation method. Exception: {type(e)}:{str(e)}")
		
		finally:
			with self._state_variables_lock:
				self._is_validation_in_progress = False
				self._is_validation_completed = True

	def handle_new_validation_results(self, results: dict, validator_id: int) -> None:
		if type(results) != dict or type(validator_id) != int:
			raise TypeError("TrimmedMeanValidation _handle_new_validation_results method")
		
		with self._state_variables_lock:
			if self._aggregation_is_ready:
				raise Exception("TrimmedMeanValidation _handle_new_validation_results method. Impossible to handle new validation results. Aggregation is already ready.")
			
			with self._validation_results_lock:
				if validator_id not in self._list_of_validators:
					raise Exception("TrimmedMeanValidation _handle_new_validation_results method. Impossible to handle new validation results. Validator ID is not in the list of validators.")
				elif validator_id in self._validation_results:
					raise Exception("TrimmedMeanValidation _handle_new_validation_results method. Impossible to handle new validation results. Validator ID is already in the list of validation results.")
				
				self._validation_results[validator_id] = results
		
				# If the number of validation results is equal to the number of validators, then the aggregation is ready
				if len(self._validation_results) == len(self._list_of_validators):
					self._aggregation_is_ready = True
				
				# If the validation is lazy, then the validator must wait for the votes from the validators with lower node_id than its node_id before starting the validation
				elif self._lazy_validation and all(validator_id in self._validation_results for validator_id in self._list_of_validators if validator_id < self._validator_node_id):
					# To set the validation ready to start, the validator must have stopped accepting new updates, the validator must have not started or completed the validation yet
					if self._must_stop_accepting_new_updates and not self._is_validation_ready_to_start and not self._is_validation_in_progress and not self._is_validation_completed:
						self._is_validation_ready_to_start = True

	@staticmethod
	def _validation(is_subprocess: bool, updates: list, trim_perc: float, distance_function_type: str, filename: str | None = None):
		result = None
		
		try:
			if type(is_subprocess) != bool or type(updates) != list or type(trim_perc) not in [float, int] or type(distance_function_type) != str or (is_subprocess and (filename is None or type(filename) != str)):
				raise TypeError("TrimmedMeanValidation _validation method")

			if is_subprocess:
				setproctitle.setproctitle(f"{current_process().name}")

			updates = [TrimmedMeanValidation._flatten_list(update) for update in updates]
			trimmed_mean = TrimmedMeanValidation._calculate_trimmed_mean(updates, trim_perc)
			scores, good_indexes = TrimmedMeanValidation._find_valid_updates(distance_function_type, trim_perc, updates, trimmed_mean)

			result = {"scores": scores, "good_indexes": good_indexes}

		except Exception as e:
			if is_subprocess:
				result = {"error": f"{type(e)}:{str(e)}"}
			else:
				raise e

		if is_subprocess:
			with open(filename, "w") as file:
				json.dump(result, file)
		else:
			return result

	@staticmethod
	def _calculate_trimmed_mean(updates: list, trim_perc: float) -> list[float]:
		"""
		Compute the trimmed mean along axis=0.
		Based on the scipy implementation.
		"""
		updates = np.array(updates)
		n = len(updates)
		lowercut = int(trim_perc * n)

		if lowercut > 0:

			if len(updates[lowercut : -lowercut]) == 0:
				raise ValueError("TrimmedMeanValidation _calculate_trimmed_mean method. Trimming percentage too large.")

			sorted_updates = np.sort(updates, axis=0)
			trimmed_updates = sorted_updates[lowercut : -lowercut]

		else:
			trimmed_updates = updates

		return np.mean(trimmed_updates, axis=0).tolist()

	@staticmethod
	def _find_valid_updates(distance_function_type: str, trim_perc: float, updates: list, trimmed_mean: list) -> tuple[list[float], list[int]]:
		"""
		Calculate the distance between each update and the trimmed mean.
		Return the scores (distances) and the indices of valid updates (the closest ones).
		"""
		updates = np.array(updates)
		trimmed_mean = np.array(trimmed_mean)

		distances = cdist(updates, trimmed_mean.reshape(1, -1), metric=distance_function_type).flatten()

		# Find the indices of valid updates (closest to the trimmed mean)
		valid_indices = np.argsort(distances).tolist()

		# Filter out the outliers based on the trimming percentage
		num_valid = int(len(updates) * (1 - 2 * trim_perc))
		valid_indices = valid_indices[:num_valid]

		# Get the valid scores (distances) for the valid updates
		scores = distances.tolist()

		return scores, valid_indices

	@staticmethod
	def _flatten_list(nested_list: list) -> list:
		if type(nested_list) != list:
			raise TypeError("TrimmedMeanValidation _flatten_list method")
		
		flattened = []
		for item in nested_list:
			if isinstance(item, list):
				flattened.extend(TrimmedMeanValidation._flatten_list(item))
			else:
				flattened.append(item)
		return flattened
	
class GlobalDatasetValidation():
	def __init__(self, validation_semaphore: BoundedSemaphoreClass, validation_set_path: str, min_update_validation_score_first_round: float, min_num_of_updates_between_aggregations: int, max_num_of_updates_between_aggregations: int, count_down_timer_to_start_aggregation: float, list_of_initial_validators: list, positive_threshold_to_pass_validation: int, lazy_loading_of_validation_set: bool, batch_size: int, verbose: int = 0, testing_ops: bool = False) -> None:
		if type(min_update_validation_score_first_round) not in [float, int] or type(validation_set_path) != str or type(min_num_of_updates_between_aggregations) != int or type(max_num_of_updates_between_aggregations) != int or type(count_down_timer_to_start_aggregation) not in [float, int] or type(lazy_loading_of_validation_set) != bool or type(verbose) != int or type(testing_ops) != bool or type(list_of_initial_validators) != list or type(positive_threshold_to_pass_validation) != int or type(batch_size) != int:
			raise TypeError("GlobalDatasetValidation")
		elif min_update_validation_score_first_round < 0 or min_num_of_updates_between_aggregations <= 0 or max_num_of_updates_between_aggregations < min_num_of_updates_between_aggregations or len(list_of_initial_validators) == 0 or positive_threshold_to_pass_validation < 0 or count_down_timer_to_start_aggregation < 0 or not isinstance(validation_semaphore, BoundedSemaphoreClass) or batch_size <= 0:
			raise ValueError("GlobalDatasetValidation")
		
		self._validation_semaphore = validation_semaphore

		self._min_update_validation_score_first_round = min_update_validation_score_first_round
		self._min_update_validation_score = min_update_validation_score_first_round
		self._min_num_of_updates_between_aggregations = min_num_of_updates_between_aggregations
		self._max_num_of_updates_between_aggregations = max_num_of_updates_between_aggregations
		self._count_down_timer_to_start_aggregation = count_down_timer_to_start_aggregation
		self._positive_threshold_to_pass_validation = positive_threshold_to_pass_validation
		self._lazy_loading_of_validation_set = lazy_loading_of_validation_set
		self._validation_set_path = validation_set_path
		self._verbose = verbose
		self._testing_ops = testing_ops
		self._batch_size = batch_size
		
		self._state_variables_lock = threading.Lock()
		self._aggregation_is_ready = False
		self._timer_to_start_validation = None

		self._updates_validation_scores = {}
		self._honest_updates_and_aggregated_scores = {}
		self._count_of_completed_positive_validations = 0
		self._updates_tables_lock = threading.Lock()

		self._list_of_validators = list_of_initial_validators

	def get_min_update_validation_score(self) -> int:
		return self._min_update_validation_score
		
	def is_aggregation_ready(self) -> bool:
		with self._state_variables_lock:
			return self._aggregation_is_ready
		
	def get_honest_updates_and_aggregated_scores(self) -> dict:
		with self._updates_tables_lock:
			return self._honest_updates_and_aggregated_scores

	def must_stop_accepting_new_updates(self) -> bool:
		with self._state_variables_lock:
			with self._updates_tables_lock:
				return self._aggregation_is_ready

	def _stop_accepting_new_update_validations(self) -> None:
		with self._state_variables_lock:
			with self._updates_tables_lock:
				if self._count_of_completed_positive_validations != self._max_num_of_updates_between_aggregations:
					if self._aggregation_is_ready or self._count_of_completed_positive_validations < self._min_num_of_updates_between_aggregations or self._count_of_completed_positive_validations > self._max_num_of_updates_between_aggregations:
						raise ValueError(f"GlobalDatasetValidation _stop_accepting_new_update_validations method. Is aggregation ready: {self._aggregation_is_ready}. Number of honest updates and aggregated scores: {self._count_of_completed_positive_validations}. Min number of updates between aggregations: {self._min_num_of_updates_between_aggregations}. Max number of updates between aggregations: {self._max_num_of_updates_between_aggregations}.")

					self._aggregation_is_ready = True

	def validate_update(self, model_architecture: dict, model_weights: list) -> tuple[float, bool] | None:
	
		if type(model_architecture) != dict or type(model_weights) != list:
			raise TypeError("GlobalDatasetValidation validate_update method")

		with self._state_variables_lock:
			if self._aggregation_is_ready or self._count_of_completed_positive_validations >= self._max_num_of_updates_between_aggregations:
				return None
			
		try:
			if self._testing_ops:
				validation_score = np.random.uniform(0.75, 1.0)
			
			else:
				with self._validation_semaphore:
					filename = os.path.join(DIRECTORY_WHERE_TO_STORE_TMP_FILES, ''.join(np.random.choice(list(string.ascii_lowercase + string.digits), size=24)) + ".json")

					validation_process = Process(target=GlobalDatasetValidation._validation, args=(True, self._lazy_loading_of_validation_set, model_architecture, model_weights, self._validation_set_path, self._batch_size, self._verbose, filename), name= f"FedBlockSimulator - global_dataset_validation", daemon= True)
					validation_process.start()
					validation_process.join()

				with open(filename, "r") as file:
					result = json.load(file)

				os.remove(filename)

				if type(result) != dict:
					raise Exception("Result from subprocess is not dict")
				elif "error" in result:
					raise Exception(f"Error while validating update. Error: {result['error']}")
				
				validation_score = result["validation_score"]

				with self._state_variables_lock:
					if self._aggregation_is_ready or self._count_of_completed_positive_validations >= self._max_num_of_updates_between_aggregations:
						return None

			return validation_score, validation_score >= self._min_update_validation_score
		except Exception as e:
			raise Exception(f"GlobalDatasetValidation validate_update method. Exception: {type(e)}:{str(e)}")
	
	@staticmethod
	def _validation(is_subprocess: bool, lazy_loading: bool, model_architecture: dict, model_weights: list, validation_set_path: str, batch_size: int, verbose: int, filename: str | None = None):
		result = None
		
		try:
			if type(is_subprocess) != bool or type(lazy_loading) != bool or type(model_architecture) != dict or type(model_weights) != list or type(validation_set_path) != str or type(batch_size) != int or type(verbose) != int or (is_subprocess and (filename is None or type(filename) != str)):
				raise TypeError("GlobalDatasetValidation _validation method")

			if is_subprocess:
				setproctitle.setproctitle(f"{current_process().name}")

			model = utils.build_model_from_architecture_and_weights(model_architecture, model_weights)

			validation_set = utils.get_dataset(batch_size, validation_set_path, lazy_loading)

			if lazy_loading:
				_, validation_score = model.evaluate(validation_set, verbose=verbose)
			else:
				_, validation_score = model.evaluate(validation_set['img'], validation_set['label'], verbose=verbose)

			result = {"validation_score": validation_score}

		except Exception as e:
			if is_subprocess:
				result = {"error": f"{type(e)}:{str(e)}"}
			else:
				raise e

		if is_subprocess:
			with open(filename, "w") as file:
				json.dump(result, file)
		else:
			return result

	def handle_new_validation_result(self, positive_validation: bool, accuracy: float, update_name: str, updater_id: int, validator_id: int) -> tuple[bool, int, float, bool] | None:
		'''
		Handle new validation result. If the number of validators that have validated the update is equal to the number of validators, then check if the number of validators that have validated the update positively is greater than or equal to the positive threshold to pass validation. If it is, then return True. Otherwise, return False. If the update is uncompleted, then return None.
		If the number of updates completed validated with a number of positive votes greater then the threshold is equal to the number of updates between aggregations, then set the aggregation to be ready.
		'''

		if type(positive_validation) != bool or type(accuracy) not in [float, int] or type(update_name) != str or type(updater_id) != int or type(validator_id) != int:
			raise TypeError("GlobalDatasetValidation handle_new_validation_result method")
		elif validator_id not in self._list_of_validators:
			raise ValueError("GlobalDatasetValidation handle_new_validation_result method")
		elif positive_validation != (accuracy >= self._min_update_validation_score):
			raise ValueError("GlobalDatasetValidation handle_new_validation_result method")

		result = None

		with self._state_variables_lock:
			# It may happen that the aggregation is ready before the validator has received the votes from all the validators. In this case, the validator must reject the new validation results
			if self._aggregation_is_ready:
				return None
				
			with self._updates_tables_lock:
				if self._count_of_completed_positive_validations >= self._max_num_of_updates_between_aggregations:
					return None

				if update_name not in self._updates_validation_scores:
					self._updates_validation_scores[update_name] = {}
				elif validator_id in self._updates_validation_scores[update_name]:
					raise Exception(f"GlobalDatasetValidation handle_new_validation_result method. Impossible to handle new validation result. Validator ID is already in the list of validation results. Update Name: {update_name}. Validator ID: {validator_id}. Known Validators: {self._updates_validation_scores[update_name].keys()}")
				
				self._updates_validation_scores[update_name][validator_id] = accuracy

				if len(self._updates_validation_scores[update_name]) == len(self._list_of_validators):
					
					list_of_positive_scores = [accuracy for accuracy in self._updates_validation_scores[update_name].values() if accuracy >= self._min_update_validation_score]

					# Check if the update has been validated by all validators and if the number of positive votes is greater than or equal to the positive threshold to pass validation
					if len(list_of_positive_scores) >= self._positive_threshold_to_pass_validation:
						if update_name in self._honest_updates_and_aggregated_scores:
							raise Exception(f"GlobalDatasetValidation handle_new_validation_result method. Impossible to handle new validation result. Update Name is already in the list of honest updates and aggregated scores. Update Name: {update_name}. Known Updates: {self._honest_updates_and_aggregated_scores.keys()}")
						
						self._count_of_completed_positive_validations += 1
						self._honest_updates_and_aggregated_scores[update_name] = self._aggregate_validation_scores(list_of_positive_scores)
						result = True, len(list_of_positive_scores), self._honest_updates_and_aggregated_scores[update_name], self._count_of_completed_positive_validations == self._min_num_of_updates_between_aggregations

						if self._count_of_completed_positive_validations == self._min_num_of_updates_between_aggregations:
							if self._timer_to_start_validation is None:
								self._timer_to_start_validation = threading.Timer(self._count_down_timer_to_start_aggregation, self._stop_accepting_new_update_validations)
								self._timer_to_start_validation.start()
							else:
								raise Exception("GlobalDatasetValidation handle_new_validation_result method. Timer to start validation is already set.")
				
						elif self._count_of_completed_positive_validations == self._max_num_of_updates_between_aggregations:
							if self._timer_to_start_validation is not None:
								self._timer_to_start_validation.cancel()
							
							self._aggregation_is_ready = True

						elif self._count_of_completed_positive_validations > self._max_num_of_updates_between_aggregations:
							raise Exception("GlobalDatasetValidation handle_new_validation_result method. Impossible to handle new validation result. Number of completed validations is greater than the maximum number of updates between aggregations")
					else:
						result = False, len(list_of_positive_scores), 0, False
				
				elif len(self._updates_validation_scores[update_name]) > len(self._list_of_validators):
					raise Exception("GlobalDatasetValidation handle_new_validation_result method. Impossible to handle new validation result. Number of validators that have validated the update is greater than the number of validators")
			
			return result
			
	def start_new_round(self, list_of_validators: list, honest_updates_aggregated_scores: list = None) -> None:
		if type(list_of_validators) != list:
			raise TypeError("GlobalDatasetValidation start_new_round method")
		elif len(list_of_validators) == 0 or (honest_updates_aggregated_scores is not None and (len(honest_updates_aggregated_scores) < self._min_num_of_updates_between_aggregations or len(honest_updates_aggregated_scores) > self._max_num_of_updates_between_aggregations)) or (honest_updates_aggregated_scores is not None and not all(type(score) in [int, float] for score in honest_updates_aggregated_scores)):
			raise ValueError("GlobalDatasetValidation start_new_round method")

		if self._timer_to_start_validation is not None:
			self._timer_to_start_validation.cancel()

			while self._timer_to_start_validation.is_alive():
				time.sleep(0.1)

		with self._state_variables_lock:
			with self._updates_tables_lock:
				if honest_updates_aggregated_scores is None:
					self._define_minimum_update_validation_score_for_the_next_round(list(self._honest_updates_and_aggregated_scores.values()))
				else:
					self._define_minimum_update_validation_score_for_the_next_round(honest_updates_aggregated_scores)

				self._aggregation_is_ready = False
				self._updates_validation_scores = {}
				self._count_of_completed_positive_validations = 0
				self._honest_updates_and_aggregated_scores = {}
				self._list_of_validators = list_of_validators
				self._timer_to_start_validation = None

	def _aggregate_validation_scores(self, scores: list):
		"""
		Method to aggregate the scores of the updates. The aggregation is done by taking the average of the scores.
		"""

		if type(scores) != list or not all(type(score) in [int, float] for score in scores):
			raise TypeError("GlobalDatasetValidation _aggregate_validation_scores method")
		elif len(scores) == 0:
			raise ValueError("GlobalDatasetValidation _aggregate_validation_scores method")

		agg = sum(scores) / len(scores)
		return agg
	
	def _define_minimum_update_validation_score_for_the_next_round(self, updaters_scores: list):

		if type(updaters_scores) != list:
			raise TypeError("GlobalDatasetValidation _define_minimum_update_validation_score_for_the_next_round method")
		elif len(updaters_scores) == 0:
			raise ValueError("GlobalDatasetValidation _define_minimum_update_validation_score_for_the_next_round method")
				

		threshold_num_of_scores = int(self._max_num_of_updates_between_aggregations * 0.50)

		if threshold_num_of_scores <= self._min_num_of_updates_between_aggregations:
			threshold_num_of_scores = self._min_num_of_updates_between_aggregations + 15
			
		if threshold_num_of_scores > self._max_num_of_updates_between_aggregations:
			threshold_num_of_scores = self._max_num_of_updates_between_aggregations
		
		# Sort the scores in descending order. I decide to order the scores in descending order because in situations where the num of scores is even the median is the lower central value (Ex: [4,3,2,1] --> Median is 2)
		updaters_scores.sort(reverse=True)

		index = int(len(updaters_scores) * 0.75)
		num_of_updates_before_index = index + 1

		if self._testing_ops:
			self._min_update_validation_score = updaters_scores[-1]
		elif num_of_updates_before_index >= threshold_num_of_scores:
			self._min_update_validation_score = updaters_scores[index]
		elif num_of_updates_before_index < threshold_num_of_scores:
			if threshold_num_of_scores <= len(updaters_scores):
				self._min_update_validation_score = updaters_scores[threshold_num_of_scores - 1]
			else:
				self._min_update_validation_score = updaters_scores[-1] - 0.05
		
		if self._min_update_validation_score < 0:
			self._min_update_validation_score = 0
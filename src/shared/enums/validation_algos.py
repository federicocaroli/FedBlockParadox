from .common import ConfigGenericValidationAlgorithmParams

class PassWeightsValidationParams(ConfigGenericValidationAlgorithmParams):
    '''Enum for Pass Validation Parameters'''
    MINIMUM_NUM_OF_UPDATES_BETWEEN_AGGREGATIONS = "min_num_of_updates_between_aggregations"
    MAXIMUM_NUM_OF_UPDATES_BETWEEN_AGGREGATIONS = "max_num_of_updates_between_aggregations"
    COUNT_DOWN_TIMER_TO_START_AGGREGATION = "count_down_timer_to_start_aggregation"

class LocalDatasetUsedForValidationParams(ConfigGenericValidationAlgorithmParams):
    '''Enum for Local Dataset Used For Validation'''
    MIN_UPDATE_VALIDATION_SCORE_FIRST_ROUND = "min_update_validation_score_first_round"
    MINIMUM_NUM_OF_UPDATES_BETWEEN_AGGREGATIONS = "min_num_of_updates_between_aggregations"
    MAXIMUM_NUM_OF_UPDATES_BETWEEN_AGGREGATIONS = "max_num_of_updates_between_aggregations"
    COUNT_DOWN_TIMER_TO_START_AGGREGATION = "count_down_timer_to_start_aggregation"

class GlobalDatasetUsedForValidationParams(ConfigGenericValidationAlgorithmParams):
    '''Enum for Local Dataset Used For Validation'''
    MIN_UPDATE_VALIDATION_SCORE_FIRST_ROUND = "min_update_validation_score_first_round"
    MINIMUM_NUM_OF_UPDATES_BETWEEN_AGGREGATIONS = "min_num_of_updates_between_aggregations"
    MAXIMUM_NUM_OF_UPDATES_BETWEEN_AGGREGATIONS = "max_num_of_updates_between_aggregations"
    COUNT_DOWN_TIMER_TO_START_AGGREGATION = "count_down_timer_to_start_aggregation"

class PassGradientsValidationParams(ConfigGenericValidationAlgorithmParams):
    '''Enum for Pass Gradient Validation Parameters'''
    MINIMUM_NUM_OF_UPDATES_NEEDED_TO_START_VALIDATION = "min_num_of_updates_needed_to_start_validation"
    MAXIMUM_NUM_OF_UPDATES_NEEDED_TO_START_VALIDATION = "max_num_of_updates_needed_to_start_validation"
    COUNT_DOWN_TIMER_TO_START_VALIDATION = "count_down_timer_to_start_validation"

class KrumValidationParams(ConfigGenericValidationAlgorithmParams):
    '''Enum for Krum Validation Parameters'''
    MINIMUM_NUM_OF_UPDATES_NEEDED_TO_START_VALIDATION = "min_num_of_updates_needed_to_start_validation"
    MAXIMUM_NUM_OF_UPDATES_NEEDED_TO_START_VALIDATION = "max_num_of_updates_needed_to_start_validation"
    NUM_OF_UPDATES_TO_VALIDATE_NEGATIVELY = "num_of_updates_to_validate_negatively"
    COUNT_DOWN_TIMER_TO_START_VALIDATION = "count_down_timer_to_start_validation"
    DISTANCE_FUNCTION = "distance_function"

class TrimmedMeanValidationParams(ConfigGenericValidationAlgorithmParams):
    '''Enum for Trimmed Mean Validation Parameters'''
    MINIMUM_NUM_OF_UPDATES_NEEDED_TO_START_VALIDATION = "min_num_of_updates_needed_to_start_validation"
    MAXIMUM_NUM_OF_UPDATES_NEEDED_TO_START_VALIDATION = "max_num_of_updates_needed_to_start_validation"
    TRIMMING_PERCENTAGE = "trimming_percentage"
    COUNT_DOWN_TIMER_TO_START_VALIDATION = "count_down_timer_to_start_validation"
    DISTANCE_FUNCTION = "distance_function"
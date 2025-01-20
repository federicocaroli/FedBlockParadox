import os, logging, copy
from threading import Lock

# Log levels constants
DEBUG = logging.DEBUG
INFO = logging.INFO
WARNING = logging.WARNING
ERROR = logging.ERROR
CRITICAL = logging.CRITICAL

class Diagnostic():
	'''Provides a common interface to record important events'''

	_VALID_LOG_LEVELS = [DEBUG, INFO, WARNING, ERROR, CRITICAL]

	def __init__(self, path: str, logLevel: int, maxLogRecordLength: int = 500, logger_name: (str | None) = None) -> object:

		if type(path) != str or type(logLevel) != int or type(maxLogRecordLength) != int or (logger_name != None and type(logger_name) != str):
			raise TypeError("Diagnostic __init__ method")

		if logLevel not in self._VALID_LOG_LEVELS:
			raise ValueError("Diagnostic __init__ method")

		self._status = True
		self._path = path
		self._logLevel = logLevel
		self._maxLogRecordLength = maxLogRecordLength
		self._logger_name = logger_name
		self._logger = None

		if logger_name is not None:
			self._logger = logging.getLogger(logger_name)
		else:
			self._logger = logging.getLogger()

		self._lock = Lock()                                                                                         	# Sync operations on logging module
		self._fileHandler = logging.FileHandler(filename=path, mode='a', encoding='utf-8')                                     	# Creates the log file's instance
		self._fileHandler.setFormatter(logging.Formatter("%(levelname)s:%(asctime)s,%(message)s", datefmt="%Y/%m/%d %H:%M:%S")) # Sets the format of the log message
		self._logger.setLevel(INFO)
		self._logger.addHandler(self._fileHandler)                                                                       # Adds the log file's handler between those handled by logging module
		self.record(msg = "Starting program", logLevel = INFO, identifier= self._logger_name)
		self._logger.setLevel(logLevel)                                                                                  # Sets the minimum log level to save the message on the file
		

	@property
	def validLogLevels(self) -> list:
		'''Get a list of supported log levels'''
		return copy.deepcopy(self._VALID_LOG_LEVELS)

	@property
	def activedLogLevel(self) -> list:
		'''Get the log level in use'''
		return self._logLevel

	# Change file location
	def updateParam(self, path: str, logLevel: int) -> bool:
		'''
		Change the path of logger file. 
		'''
		
		if type(path) != str or type(logLevel) != int:
			raise TypeError("Diagnostic updateParam method")

		if logLevel not in self._VALID_LOG_LEVELS:
			raise ValueError("Diagnostic updateParam method")

		if self._status == True:
			with self._lock:
				try:
					self._logger.removeHandler(self._fileHandler)
					self._fileHandler = logging.FileHandler(filename = path, mode = 'a', encoding = 'utf-8')
					self._fileHandler.setFormatter(logging.Formatter("%(levelname)s:%(asctime)s,%(message)s", datefmt="%Y/%m/%d %H:%M:%S")) # Sets the format of the log message
					self._logger.addHandler(self._fileHandler)      
					self._path = path
				except Exception as e:
					return False

				try:
					self._logger.setLevel(logLevel)
					self._logLevel = logLevel
				except Exception as e:
					return False
		
		return True

	# Truncate files to reduce their demension
	@staticmethod
	def truncate(filename: str) -> None:
		'''
		Truncate the file to fix its dimension to 50 rows (most recent rows are preserved)
		'''
		if type(filename) != str:
			raise TypeError("Diagnostic truncate method")
		if os.path.isfile(filename) == False:
			raise ValueError("Diagnostic truncate method")

		os.system('echo "$(tail -n 100000 ' + filename + ' )" > ' + filename)
	
	# Record an event into logger file
	def record(self, msg: str, logLevel: int, identifier: str, exc: (Exception | None) = None, skipLengthTruncation: bool = False) -> None:
		'''
		Record a message in the logger file. If the logger file is too big, it is truncated.
		'''

		if type(msg) != str or type(logLevel) != int or type(identifier) != str or (exc != None and isinstance(exc, Exception) == False) or type(skipLengthTruncation) != bool:
			raise TypeError("Diagnostic record method")
		if logLevel not in self._VALID_LOG_LEVELS:
			raise ValueError("Diagnostic record method")

		if self._status == True:
			with self._lock:

				# Check the logger file's size
				if (os.path.getsize(self._path) >= 200000000):
					self.truncate(filename = self._path)

				msg = repr(msg)[1:-1]

				if len(msg) > self._maxLogRecordLength and skipLengthTruncation == False:
					msg = msg[:self._maxLogRecordLength] + " ..."

				logMsg = "[{identifier}] {msg}".format(identifier = identifier, msg = msg)

				if exc != None:
					exceptionPartOfMsg = repr(". {type}:{exception}".format(type= type(exc), exception = str(exc)))[1:-1]

					if len(exceptionPartOfMsg) > self._maxLogRecordLength and skipLengthTruncation == False:
						exceptionPartOfMsg = exceptionPartOfMsg[:self._maxLogRecordLength] + " ..."

					logMsg += exceptionPartOfMsg

				self._logger.log(logLevel, logMsg)
				self._logger.handlers[0].flush()

	def clear(self) -> None:
		'''
		Clear log file.
		'''

		with self._lock:
			os.system('echo "$(tail -n 10 ' + self._path + ' )" > ' + self._path)

	def readLog(self) -> str:
		'''
		Read log file.
		'''

		contents = ""
		with self._lock:
			with open(self._path, 'r') as log_file:
				contents = log_file.read()
		return contents

	# Close all handler    
	def shutdown(self):
		'''
		Close all handlers handle from logging module.
		'''
		
		if self._status == True:
			with self._lock:
				logging.shutdown()
				self._status = False
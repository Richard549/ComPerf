import logging
import sys
import fcntl
import time
import errno
import os

from enum import Enum, auto

class LogLevelOption(Enum):
	INFO = auto()
	DEBUG = auto()
	TRACE = auto()

log = logging.getLogger(__name__)
log_level_value = 1

def debug_mode():
	global log_level_value
	if log_level_value > 1:
		return True
	else:
		return False

def acquire_lock(mutex_file):
	while True:
		try:
			fcntl.flock(mutex_file, fcntl.LOCK_EX | fcntl.LOCK_NB)
			break
		except IOError as e:
			# raise on unrelated IOErrors
			if e.errno != errno.EAGAIN:
				raise
			else:
				time.sleep(0.1)

def release_lock(mutex_file):
	fcntl.flock(mutex_file, fcntl.LOCK_UN)

def initialise_logging(log_file, log_level_option, tee=False):

	file_handler = logging.FileHandler(filename=log_file)
	handlers = [file_handler]

	if tee:
		stdout_handler = logging.StreamHandler(sys.stdout)
		handlers.append(stdout_handler)

	# Add TRACE: a log level more low-level than debug

	TRACE_LOG_LEVEL = logging.DEBUG -1

	def logForLevel(self, message, *args, **kwargs):
		if self.isEnabledFor(TRACE_LOG_LEVEL):
			self._log(TRACE_LOG_LEVEL, message, args, **kwargs)
	def logToRoot(message, *args, **kwargs):
		logging.log(TRACE_LOG_LEVEL, message, *args, **kwargs)

	logging.addLevelName(TRACE_LOG_LEVEL, "TRACE")
	setattr(logging, "TRACE", TRACE_LOG_LEVEL)
	setattr(logging.getLoggerClass(), "trace", logForLevel)
	setattr(logging, "trace", logToRoot)

	# configure the log level requested by the user

	log_level = None
	if log_level_option == LogLevelOption.INFO:
		log_level = logging.INFO
	elif log_level_option == LogLevelOption.DEBUG:
		log_level = logging.DEBUG
	elif log_level_option == LogLevelOption.TRACE:
		log_level = logging.TRACE
	else:
		print("Log level", log_level_option, "not supported.")
		raise NotImplementedError()

	global log_level_value
	log_level_value = log_level_option.value

	logging.basicConfig(
		level=log_level,
		format='[%(asctime)s] {%(filename)s:%(lineno)d} %(levelname)s - %(message)s',
		handlers=handlers
	)

	stdout_logger = logging.getLogger('STDOUT')
	sl = StreamToLogger(stdout_logger, logging.DEBUG)
	sys.stdout = sl

	stderr_logger = logging.getLogger('STDERR')
	sl = StreamToLogger(stderr_logger, logging.ERROR)
	sys.stderr = sl

	#logging.getLogger('matplotlib.font_manager').disabled = True
	mpl_logger = logging.getLogger('matplotlib')
	mpl_logger.setLevel(logging.WARNING)

	log.debug("Initialised logging.")

class StreamToLogger(object):
	"""
	Fake file-like stream object that redirects writes to a logger instance.
	"""
	def __init__(self, logger, log_level=logging.INFO):
		self.logger = logger
		self.log_level = log_level
		self.linebuf = ""

	def write(self, buf):

		for line in buf.rstrip().splitlines():
			self.logger.log(self.log_level, "%s", str(line.rstrip()))

	def flush(self):
		pass

def ensure_exists(folder):
	# No validation - better hope the config file isn't malicious! 
	os.system("mkdir -p " + folder)

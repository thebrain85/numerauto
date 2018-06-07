"""
Numerauto example that calls a command line when a new round has started.
"""

import logging
import sys

from numerauto import Numerauto
from numerauto.eventhandlers import CommandlineExecutor


# Set up logging to file and stdout
log_format = "%(asctime)s [%(levelname)8s] %(name)s: %(message)s"
logging.basicConfig(format=log_format, level=logging.DEBUG,
                    handlers=[logging.FileHandler('debug.log'),
                              logging.StreamHandler(sys.stdout)])

# Create Numerauto instance and add Commandline executor
na = Numerauto()
na.add_event_handler(CommandlineExecutor('executor',
                                         on_new_tournament_commandline='echo New round: %round% Data: %dataset_path%'))

try:
    na.run(single_run=True)
except Exception as e:
    logging.exception(e)

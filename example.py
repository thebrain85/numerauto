"""
Basic Numerauto example.

Trains, applies and uploads a logistic regression model.

Note: Replace publickey and secretkey with your own API keys to prevent a
NumerAPIAuthorizationError.
"""

import logging
import sys

from sklearn.linear_model import LogisticRegression

from numerauto import Numerauto
from numerauto.eventhandlers import SKLearnModelTrainer, PredictionUploader


# Set up logging to file and stdout
log_format = "%(asctime)s [%(levelname)8s] %(name)s: %(message)s"
logging.basicConfig(format=log_format, level=logging.DEBUG,
                    handlers=[logging.FileHandler('debug.log'),
                              logging.StreamHandler(sys.stdout)])

# Create Numerauto instance and add event handlers
# Note that the event handlers are processed in the order they are added
na = Numerauto()
na.add_event_handler(SKLearnModelTrainer('logistic_regression',
                                         lambda: LogisticRegression()))
na.add_event_handler(PredictionUploader('logistic_regression_uploader',
                                        'logistic_regression.csv',
                                        'insert your publickey here',
                                        'insert your secretkey here'))

try:
    na.run()
except Exception as e:
    logging.exception(e)

"""
Basic Numerauto example.

Trains, applies and uploads a linear regression model.

Note: Replace publickey and secretkey with your own API keys to prevent a
NumerAPIAuthorizationError.
"""

import logging
import sys

from sklearn.linear_model import LinearRegression

from numerauto import Numerauto
from numerauto.eventhandlers import SKLearnModelTrainer, PredictionUploader
from numerauto.eventhandlers import PredictionStatisticsGenerator, BasicReportWriter, BasicReportEmailer


# Set up logging to file and stdout
log_format = "%(asctime)s [%(levelname)8s] %(name)s: %(message)s"
fh = logging.FileHandler('numerauto.log')
fh.setLevel(logging.INFO)
logging.basicConfig(format=log_format, level=logging.DEBUG,
                    handlers=[logging.FileHandler('debug.log'),
                              fh,
                              logging.StreamHandler(sys.stdout)])

# Create Numerauto instance and add event handlers
# Note that the event handlers are processed in the order they are added
na = Numerauto()

# Model trainer
# The tournament ID defaults to that of the Numerauto instance (which defaults to 8)

# Models are stored in ./models/tournament_<name>/round_<num>/<name>.p
# Predictions are stored in ./predictions/tournament_<name>/round_<num>/<name>.csv
na.add_event_handler(SKLearnModelTrainer('linear_regression',
                                         lambda: LinearRegression()))

# Generate statistics for the prediction in the numerauto report dictionary
na.add_event_handler(PredictionStatisticsGenerator('gen1', 'linear_regression.csv'))

# Prediction uploader
na.add_event_handler(PredictionUploader('linear_regression_uploader',
                                        'linear_regression.csv',
                                        'insert your publickey here',
                                        'insert your secretkey here'))

# Report handlers: Write a simple report to file and email it
na.add_event_handler(BasicReportWriter('writer'))
na.add_event_handler(BasicReportEmailer('emailer', 'smtp.gmail.com', 587, 'user', 'api-key', 'from@email', 'to@email'))


try:
    na.run()
except Exception as e:
    logging.exception(e)

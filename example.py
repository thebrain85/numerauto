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
from numerauto.eventhandlers import SKLearnModelTrainer, PredictionUploader, CommandlineExecutor


# Set up logging to file and stdout
log_format = "%(asctime)s [%(levelname)8s] %(name)s: %(message)s"
logging.basicConfig(format=log_format, level=logging.DEBUG,
                    handlers=[logging.FileHandler('debug.log'),
                              logging.StreamHandler(sys.stdout)])

# Create Numerauto instance and add event handlers
# Note that the event handlers are processed in the order they are added
na = Numerauto()

# Model trainers: One for each tournament ID
# The tournament ID defaults to that of the Numerauto instance (which defaults to 1)

# Models are stored in ./models/tournament_<name>/round_<num>/<name>.p
# Predictions are stored in ./predictions/tournament_<name>/round_<num>/<name>.csv
na.add_event_handler(SKLearnModelTrainer('logistic_regression1',
                                         lambda: LogisticRegression()))
na.add_event_handler(SKLearnModelTrainer('logistic_regression2',
                                         lambda: LogisticRegression(),
                                         tournament_id=2))
na.add_event_handler(SKLearnModelTrainer('logistic_regression3',
                                         lambda: LogisticRegression(),
                                         tournament_id=3))
na.add_event_handler(SKLearnModelTrainer('logistic_regression4',
                                         lambda: LogisticRegression(),
                                         tournament_id=4))
na.add_event_handler(SKLearnModelTrainer('logistic_regression5',
                                         lambda: LogisticRegression(),
                                         tournament_id=5))

# Prediction uploaders: One for each tournament
na.add_event_handler(PredictionUploader('logistic_regression_uploader1',
                                        'logistic_regression1.csv',
                                        'insert your publickey here',
                                        'insert your secretkey here'))
na.add_event_handler(PredictionUploader('logistic_regression_uploader2',
                                        'logistic_regression2.csv',
                                        'insert your publickey here',
                                        'insert your secretkey here',
                                        tournament_id=2))
na.add_event_handler(PredictionUploader('logistic_regression_uploader3',
                                        'logistic_regression3.csv',
                                        'insert your publickey here',
                                        'insert your secretkey here',
                                        tournament_id=3))
na.add_event_handler(PredictionUploader('logistic_regression_uploader4',
                                        'logistic_regression4.csv',
                                        'insert your publickey here',
                                        'insert your secretkey here',
                                        tournament_id=4))
na.add_event_handler(PredictionUploader('logistic_regression_uploader5',
                                        'logistic_regression5.csv',
                                        'insert your publickey here',
                                        'insert your secretkey here',
                                        tournament_id=5))
try:
    na.run(single_run=True)
except Exception as e:
    logging.exception(e)

"""
Numerauto event handlers module
"""

from pathlib import Path
import pickle
import logging

import pandas as pd

from numerapi.utils import ensure_directory_exists
from .robust_numerapi import RobustNumerAPI, NumerAPIError


logger = logging.getLogger(__name__)


class EventHandler:
    """
    Base Numerauto event handler.

    This event handler defines the events that are triggered by Numerauto.
    Subclasses of EventHandler can override one or more of these events and
    implement custom code to execute when the event triggers.

    Attributes:
        name: Name of the event handler
        numerauto: Numerauto instance this handler is added to (None if not added)
    """

    def __init__(self, name):
        """
        Creates a new EventHandler instance.

        Args:
            name: Event handler name.
        """

        if name == '':
            raise ValueError('Name can not be empty')

        self.name = name
        self.numerauto = None

    def on_start(self):
        """ Triggered when the Numerauto daemon starts """
        pass

    def on_shutdown(self):
        """ Triggered when the Numerauto daemon shuts down """
        pass

    def on_round_begin(self, round_number):
        """ Triggered when a new Numerai round is detected """
        pass

    def on_new_training_data(self, round_number):
        """ Triggered when new training data is detected """
        pass

    def on_new_tournament_data(self, round_number):
        """
        Triggered when new tournament data is detected. Currently this triggers
        for every new round.
        """
        pass


class SKLearnModelTrainer(EventHandler):
    """
    Event handler that trains and applies models that adhere to the sklearn API.
    The model must implement the 'fit' and 'predict_proba' methods, and must be
    able to be written to file using pickle.

    Each time the model is trained, it is saved to the ./models directory.
    Each time the model is applied, predictions are written to the ./predictions
    directory.
    """

    def __init__(self, name, model_factory):
        """
        Creates a new SKLearnModelTrainer instance.

        Args:
            name: Event handler name.
            model_factory: Function that creates a new model instance.
                           The function must take no arguments.
        """

        super().__init__(name)
        self.model_factory = model_factory

    def on_new_training_data(self, round_number):
        train_x = pd.read_csv(self.numerauto.get_dataset_path(round_number) / 'numerai_training_data.csv', header=0)

        train_y = train_x['target'].as_matrix()
        train_x = train_x.drop({'id', 'target', 'era', 'data_type'}, axis=1).as_matrix()

        logger.info('SKLearnModelTrainer(%s): Fitting model for round %d', self.name, round_number)
        model = self.model_factory()
        model.fit(train_x, train_y)

        ensure_directory_exists(Path('./models/round_{}'.format(round_number)))
        model_filename = Path('./models/round_{}/{}.p'.format(round_number, self.name))
        pickle.dump(model, open(model_filename, 'wb'))

    def on_new_tournament_data(self, round_number):
        test_x = pd.read_csv(self.numerauto.get_dataset_path(round_number) / 'numerai_tournament_data.csv', header=0)
        test_ids = test_x['id']
        test_x = test_x.drop({'id', 'target', 'era', 'data_type'}, axis=1).as_matrix()

        logger.info('SKLearnModelTrainer(%s): Applying model for round %d',
                    self.name, round_number)
        model_filename = Path('./models/round_{}/{}.p'.format(
            self.numerauto.persistent_state['last_round_trained'], self.name))
        model = pickle.load(open(model_filename, 'rb'))
        predictions = model.predict_proba(test_x)[:, 1]

        df = pd.DataFrame(predictions, columns=['probability'], index=test_ids)
        ensure_directory_exists(Path('./predictions/round_{}'.format(round_number)))
        df.to_csv(Path('./predictions/round_{}/{}.csv'.format(round_number, self.name)),
                  index_label='id', float_format='%.8f')


class PredictionUploader(EventHandler):
    """
    Event handler that uploads a predictions file from the ./predictions directory
    using the Numerai API.
    """

    def __init__(self, name, filename, public_id, secret_key):
        """
        Creates a new PredictionUploader instance.

        Args:
            name: Event handler name.
            filename: Filename of the predictions file.
            public_id: Numerai public API key for the account the prediction is uploaded to.
            secret_key: Numerai secret API key for the account the prediction is uploaded to.
        """
        super().__init__(name)
        self.filename = filename
        self.public_id = public_id
        self.secret_key = secret_key

    def on_new_tournament_data(self, round_number):
        logger.info('PredictionUploader(%s): Uploading predictions for round %d: %s',
                    self.name, round_number, self.filename)
        napi = RobustNumerAPI(public_id=self.public_id, secret_key=self.secret_key)
        try:
            prediction_path = Path('./predictions/round_{}/'.format(round_number))
            napi.upload_predictions(prediction_path / self.filename)
        except NumerAPIError as e:
            logger.error('PredictionUploader(%s): NumerAPI exception in round %d: %s',
                         self.name, round_number, e)
            logger.error('PredictionUploader(%s): Predictions not uploaded successfully, '
                         'please upload %s manually, or remove state.pickle and restart '
                         'Numerauto to process this round again', prediction_path / self.filename)

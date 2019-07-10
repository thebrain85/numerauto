"""
Numerauto event handlers module
"""

import os
from pathlib import Path
import pickle
import logging
import collections
import smtplib

import pandas as pd
from scipy.stats import spearmanr

from numerapi.utils import ensure_directory_exists
from .robust_numerapi import RobustNumerAPI, NumerAPIError
from .utils import wait_for_retry


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
    
    def on_cleanup(self, round_number):
        """ Triggered at the end of processing the current round """
        pass


class SKLearnModelTrainer(EventHandler):
    """
    Event handler that trains and applies models that adhere to the sklearn API.
    The model must implement the 'fit' and 'predict' methods, and must be
    able to be written to file using pickle.

    Each time the model is trained, it is saved to the
    numerauto.config['model_directory'] directory (defaults to ./models):
        ./models/tournament_<name>/round_<num>/<name>.p
    Each time the model is applied, predictions are written to the
    numerauto.config['prediction_directory'] directory (defaults to ./predictions):
        ./predictions/tournament_<name>/round_<num>/<name>.csv
    """

    def __init__(self, name, model_factory, tournament_id=None):
        """
        Creates a new SKLearnModelTrainer instance.

        Args:
            name: Event handler name.
            model_factory: Function that creates a new model instance.
                           The function must take no arguments.
            tournament_id: ID of the tournament to upload predictions to. The default None will copy the tournament id of the Numerauto instance
        """

        super().__init__(name)
        self.model_factory = model_factory
        self.tournament_id = tournament_id

    def on_start(self):
        if self.tournament_id is None:
            self.tournament_id = self.numerauto.tournament_id

        # Set default configuration
        if 'prediction_directory' not in self.numerauto.config:
            self.numerauto.config['prediction_directory'] = './predictions'
        if 'model_directory' not in self.numerauto.config:
            self.numerauto.config['model_directory'] = './models'
            
        # Turn model and prediction directory into pathlib Path
        self.numerauto.config['prediction_directory'] = Path(self.numerauto.config['prediction_directory'])
        self.numerauto.config['model_directory'] = Path(self.numerauto.config['model_directory'])

    def on_new_training_data(self, round_number):
        tournament_name = self.numerauto.tournaments[self.tournament_id]
        
        train_x = pd.read_csv(self.numerauto.get_dataset_path(round_number) / 'numerai_training_data.csv', header=0)
        target_columns = set([x for x in list(train_x) if x[0:7] == 'target_'])

        train_y = train_x['target_' + tournament_name].values
        train_x = train_x.drop({'id', 'era', 'data_type'} | target_columns, axis=1).values

        logger.info('SKLearnModelTrainer(%s): Fitting model for tournament %s round %d',
                    self.name, tournament_name, round_number)
        model = self.model_factory()
        model.fit(train_x, train_y)

        ensure_directory_exists(self.numerauto.config['model_directory'] / 'tournament_{}/round_{}'.format(tournament_name, round_number))
        model_filename = self.numerauto.config['model_directory'] / 'tournament_{}/round_{}/{}.p'.format(tournament_name, round_number, self.name)
        pickle.dump(model, open(model_filename, 'wb'))
        
        self.numerauto.report['training'][tournament_name][self.name]['filename'] = model_filename


    def on_new_tournament_data(self, round_number):
        tournament_name = self.numerauto.tournaments[self.tournament_id]

        test_x = pd.read_csv(self.numerauto.get_dataset_path(round_number) / 'numerai_tournament_data.csv', header=0)
        target_columns = set([x for x in list(test_x) if x[0:7] == 'target_'])

        test_ids = test_x['id']
        test_x = test_x.drop({'id', 'era', 'data_type'} | target_columns, axis=1).values

        logger.info('SKLearnModelTrainer(%s): Applying model for tournament %s round %d',
                    self.name, tournament_name, round_number)
        model_filename = self.numerauto.config['model_directory'] / 'tournament_{}/round_{}/{}.p'.format(
            tournament_name, self.numerauto.persistent_state['last_round_trained'], self.name)
        model = pickle.load(open(model_filename, 'rb'))
        predictions = model.predict(test_x)

        df = pd.DataFrame(predictions, columns=['predict_' + tournament_name], index=test_ids)
        ensure_directory_exists(self.numerauto.config['prediction_directory'] / 'tournament_{}/round_{}'.format(tournament_name, round_number))
        df.to_csv(self.numerauto.config['prediction_directory'] / 'tournament_{}/round_{}/{}.csv'.format(tournament_name, round_number, self.name),
                  index_label='id', float_format='%.8f')
        
        self.numerauto.report['predictions'][tournament_name][self.name + '.csv']['filename'] = self.numerauto.config['prediction_directory'] / 'tournament_{}/round_{}/{}.csv'.format(tournament_name, round_number, self.name)


class PredictionUploader(EventHandler):
    """
    Event handler that uploads a predictions file from the
    numerauto.config['prediction_directory'] directory (defaults to ./predictions)
    using the Numerai API.
    """

    def __init__(self, name, filename, public_id, secret_key, tournament_id=None, verify_upload=True):
        """
        Creates a new PredictionUploader instance.

        Args:
            name: Event handler name.
            filename: Filename of the predictions file.
            public_id: Numerai public API key for the account the prediction is uploaded to.
            secret_key: Numerai secret API key for the account the prediction is uploaded to.
            tournament_id: ID of the tournament to upload predictions to. The default None will copy the tournament id of the Numerauto instance
        """
        super().__init__(name)
        self.filename = filename
        self.public_id = public_id
        self.secret_key = secret_key
        self.tournament_id = tournament_id
        self.verify_upload = verify_upload

    def on_start(self):
        if self.tournament_id is None:
            self.tournament_id = self.numerauto.tournament_id
        
        # Set default configuration
        if 'upload_verify_wait_schedule' not in self.numerauto.config:
            self.numerauto.config['upload_verify_wait_schedule'] = [10, 10, 10, 10, 10, 10, 60, 60, 60, 60, 60, 600, 3600]
        
        if 'prediction_directory' not in self.numerauto.config:
            self.numerauto.config['prediction_directory'] = './predictions'
        
        # Turn prediction directory in pathlib Path
        self.numerauto.config['prediction_directory'] = Path(self.numerauto.config['prediction_directory'])

    def on_new_tournament_data(self, round_number):
        logger.info('PredictionUploader(%s): Uploading predictions for round %d: %s',
                    self.name, round_number, self.filename)
        napi = RobustNumerAPI(public_id=self.public_id, secret_key=self.secret_key,
                              retry_wait_schedule=self.numerauto.config['napi_wait_schedule'])

        tournament_name = self.numerauto.tournaments[self.tournament_id]

        try:
            prediction_path = self.numerauto.config['prediction_directory'] / 'tournament_{}/round_{}/'.format(tournament_name, round_number)
            submission_id = napi.upload_predictions(prediction_path / self.filename, tournament=self.tournament_id)
            print(submission_id)
            
            if self.verify_upload:
                status = napi.submission_status(submission_id=submission_id)

                attempts = 0
                while status['concordance'] is None or \
                      status['concordance']['pending']:
                    wait_for_retry(attempts, self.numerauto.config['upload_verify_wait_schedule'])
                    attempts += 1
                    status = napi.submission_status(submission_id=submission_id)
                
                logger.info('PredictionUploader(%s): Upload verified: Correlation: %.4f Consistency: %.1f Concordance: %r',
                            self.name, status['validationCorrelation'], status['consistency'], status['concordance']['value'])
                
                self.numerauto.report['submissions'][tournament_name][self.filename] = {
                        'submission_id': submission_id,
                        'filename': prediction_path / self.filename,
                        'validationCorrelation': status['validationCorrelation'],
                        'consistency': status['consistency'],
                        'concordance': status['concordance']['value']}
            else:
                self.numerauto.report['submissions'][tournament_name][self.filename] = {
                    'submission_id': submission_id,
                    'filename': prediction_path / self.filename}
                
        except NumerAPIError as e:
            logger.error('PredictionUploader(%s): NumerAPI exception in tournament %s round %d: %s',
                         self.name, tournament_name, round_number, e)
            logger.error('PredictionUploader(%s): Predictions not uploaded successfully, '
                         'please upload %s manually, or remove state.pickle and restart '
                         'Numerauto to process this round again', self.name, prediction_path / self.filename)



class CommandlineExecutor(EventHandler):
    """
    Event handler that executes a command line on new training and/or tournament
    data.
    """

    def __init__(self, name, on_new_training_commandline=None, on_new_tournament_commandline=None):
        """
        Creates a new CommandlineExecutor instance.
        The command lines provided in the arguments will have the substring
        %round% replaced by the current round number and %dataset_path% by the
        full path to the new unzipped dataset.

        Args:
            name: Event handler name.
            on_new_training_commandline: Command line to execute when new training data is available.
            on_new_tournament_commandline: Command line to execute when new tournament data is available.
        """
        super().__init__(name)
        self.on_new_training_commandline = on_new_training_commandline
        self.on_new_tournament_commandline = on_new_tournament_commandline

    def on_new_training_data(self, round_number):
        if self.on_new_training_commandline:
            cmdline = self.on_new_training_commandline
            cmdline = cmdline.replace('%round%', str(round_number))
            cmdline = cmdline.replace('%dataset_path%', str(self.numerauto.get_dataset_path(round_number).absolute()))

            logger.info('CommandlineExecutor(%s): Executing command: %s', self.name, cmdline)
            os.system(cmdline)

    def on_new_tournament_data(self, round_number):
        if self.on_new_tournament_commandline:
            cmdline = self.on_new_tournament_commandline
            cmdline = cmdline.replace('%round%', str(round_number))
            cmdline = cmdline.replace('%dataset_path%', str(self.numerauto.get_dataset_path(round_number).absolute()))

            logger.info('CommandlineExecutor(%s): Executing command: %s', self.name, cmdline)
            os.system(cmdline)



class PredictionStatisticsGenerator(EventHandler):
    """
    Event handler that generates statistics for a given prediction filename and
    stores them in the numerauto report dictionary.
    """
    
    def __init__(self, name, filename, tournament_id=None):
        super().__init__(name)
        self.filename = filename
        self.tournament_id = tournament_id
        
    def on_start(self):
        if self.tournament_id is None:
            self.tournament_id = self.numerauto.tournament_id
        
    def on_new_tournament_data(self, round_number):
        tournament_name = self.numerauto.tournaments[self.tournament_id]

        test_df = pd.read_csv(self.numerauto.get_dataset_path(round_number) / 'numerai_tournament_data.csv', header=0)
        
        prediction_path = self.numerauto.config['prediction_directory'] / 'tournament_{}/round_{}/'.format(tournament_name, round_number)
        p_df = pd.read_csv(prediction_path / self.filename, header=0)
        
        val_eras = test_df[test_df['data_type'] == 'validation']['era'].unique()
        
        # TODO: sort by id
        d = self.numerauto.report['predictions'][tournament_name][self.filename]
        
        consistency = 0
        for e in val_eras:
            d['validationCorrelation'][e] = spearmanr(test_df[test_df['era'] == e]['target_' + tournament_name],
                                                  p_df[test_df['era'] == e]['predict_' + tournament_name])[0]
            consistency += d['validationCorrelation'][e] > 0

        d['validationCorrelation']['overall'] = spearmanr(test_df[test_df['data_type'] == 'validation']['target_' + tournament_name],
                                                      p_df[test_df['data_type'] == 'validation']['predict_' + tournament_name])[0]
        d['consistency'] = consistency / len(val_eras)



class BasicReportWriter(EventHandler):
    """
    Event handler that writes the numerauto report dictionary to a basic report
    file.
    """
    
    def on_start(self):
        if 'report_directory' not in self.numerauto.config:
            self.numerauto.config['report_directory'] = './reports'

        # Turn report directory in pathlib Path
        self.numerauto.config['report_directory'] = Path(self.numerauto.config['report_directory'])

    def on_cleanup(self, round_number):
        # Function to turn nested_defaultdict back into normal dictionaries
        to_dict = lambda x: {y: to_dict(x[y]) for y in x} if type(x) == collections.defaultdict else x
        
        ensure_directory_exists(self.numerauto.config['report_directory'])
        filename = self.numerauto.config['report_directory'] / 'round_{}.txt'.format(round_number)
        with open(filename, 'w') as f:
            # Recurse through dictionary structure and write to file
            def write_dict(d, indent=0):
                for x in d:
                    if type(d[x]) == dict:
                        f.write('  '*indent + str(x) + ':\n')
                        write_dict(d[x], indent+1)
                    else:
                        f.write('  '*indent + str(x) + ': ' + str(d[x]) + '\n')
                        
            logger.debug('BasicReportWriter(%s): Writing report to file: %s', self.name, filename)
            write_dict(to_dict(self.numerauto.report))


class BasicReportEmailer(EventHandler):
    """
    Event handler that emails the numerauto report dictionary as an email with
    simple formatting.
    """
    
    def __init__(self, name, smtp_server, smtp_port, smtp_user, smtp_password, email_from, email_to, smtp_tls=True):
        super().__init__(name)
        
        self.smtp_server = smtp_server
        self.smtp_port = smtp_port
        self.smtp_user = smtp_user
        self.smtp_password = smtp_password
        self.email_from = email_from
        self.email_to = email_to
        self.smtp_tls = smtp_tls

    def on_cleanup(self, round_number):
        # Function to turn nested_defaultdict back into normal dictionaries
        to_dict = lambda x: {y: to_dict(x[y]) for y in x} if type(x) == collections.defaultdict else x
        
        # Recurse through dictionary structure and write to report to a string
        def convert_dict(d, indent=0):
            report = ''
            for x in d:
                if type(d[x]) == dict:
                    report += '  '*indent + str(x) + ':\n'
                    report += convert_dict(d[x], indent+1)
                else:
                    report += '  '*indent + str(x) + ': ' + str(d[x]) + '\n'
            return report
        
        report = convert_dict(to_dict(self.numerauto.report))

        try:
            s = smtplib.SMTP(host=self.smtp_server, port=self.smtp_port)
            
            if self.smtp_tls:
                s.starttls()
    
            s.login(self.smtp_user, self.smtp_password)
                    
            message_subject = 'Numerauto report: round {}'.format(round_number)
    
            message = "From: %s\n" % self.email_from \
                    + "To: %s\n" % self.email_to \
                    + "Subject: %s\n" % message_subject \
                    + "\n"  \
                    + report
            logger.debug('BasicReportEmailer(%s): Sending report email', self.name)
            s.sendmail(self.email_from, self.email_to, message)
        except smtplib.SMTPException as e:
            logger.error('BasicReportEmailer(%s): SMTP exception while attempting to send report email: %s',
                         self.name, round_number, e)
            

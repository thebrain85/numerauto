"""
Main Numerauto module
"""

import pickle
import datetime
import signal
import sys
import os
import shutil
import collections
from pathlib import Path
import logging

import requests
import pytz
import dateutil

from .robust_numerapi import RobustNumerAPI
from .utils import check_dataset
from .utils import wait, wait_until

logger = logging.getLogger(__name__)

# Defaultdict that will fill missing values with itself, allowing nested
# dictionary queries without first creating the keys.
nested_defaultdict = lambda: collections.defaultdict(nested_defaultdict)


class InterruptedException(Exception):
    """ Exception that is raised by our signal handler. """
    pass

def signal_handler(signum, frame):
    """ SIGINT/SIGTERM handler """

    logger.info('Signal received, exiting!')
    raise InterruptedException()


class Numerauto:
    """
    Numerai daemon.

    The Numerauto class implements a daemon that automatically detects the
    start of new Numerai rounds. Custom event handlers can be added to the
    Numerauto instance to add custom code that trains and applies models to new
    data, and uploads predictions for each round.

    See numerauto.handlers for some basic event handlers.

    Attributes:
        tournament_id: Numerai tournament id for which this instance will download data.
        napi: A robust version of NumerAPI (note that no API keys are supplied)
        event_handlers: List of event handlers that are bound to this instance.
        persistent_state: Internal storage of the current state of the daemon.
        round_number: Current round number.
        tournaments: Dictionary mapping tournament ID to tournament name
        report: Dictionary that event handlers can write to during round processing.
        config: Dictionary that contains all Numerauto configuration entries
    """

    def __init__(self, tournament_id=8, config={}):
        """
        Creates a Numerauto instance.

        Args:
            tournament_id: Numerai tournament id for which this instance will download data.
            config: Dictionary containing configuration entries to replace the default values
        """
        self.tournament_id = tournament_id
        self.event_handlers = []
        self.persistent_state = None
        self.round_number = None
        self.tournaments = None
        self.report = None
        
        self.config = {
                # Directory to store data
                'data_directory': './data',
                # Include validation data when checking for new training data
                'check_validation_data': True,
                # Seconds before planned round start to wake up and start checking
                # if new round has started.
                'wakeup_time': 360,
                # Seconds to wait between each check for the new round.
                'round_wait_interval': 60,
                # If a dataset was downloaded that was not new, wait this many seconds
                # before downloading the dataset again.
                'invalid_dataset_waittime': 600,
                # In single_run mode, maximum seconds to wait for a new round
                'single_run_max_wait': 86400,
                # Incremental waiting times for failed RobustNumerAPI queries (5x 1 minute, 3x 10 minutes, 3x 1 hour)
                'napi_wait_schedule': [60, 60, 60, 60, 60, 600, 600, 600, 3600, 3600, 3600]
                }
        
        # Add/replace user-defined config entries
        self.config = {**self.config, **config}
        
        # Change data directory into a pathlib Path
        self.config['data_directory'] = Path(self.config['data_directory'])
        
        self.napi = RobustNumerAPI(verbosity='warning', show_progress_bars=False,
                                   retry_wait_schedule=self.config['napi_wait_schedule'])


    def add_event_handler(self, handler):
        """
        Add an event handler to this instance.

        Args:
            handler: Event handler to add.
        """

        self.event_handlers.append(handler)
        handler.numerauto = self

    def remove_event_handler(self, handler_name):
        """
        Remove an event handler from this instance by its name.

        Args:
            handler_name: Name of the event handler to remove.
        """

        for h in self.event_handlers:
            if h.name == handler_name:
                h.numerauto = None

        self.event_handlers = [h for h in self.event_handlers if h.name != handler_name]

    def _on_start(self):
        """ Internal event on daemon start """

        logger.debug('on_start')
        for h in self.event_handlers:
            h.on_start()

    def _on_shutdown(self):
        """ Internal event on daemon shutdown """

        logger.debug('on_shutdown')
        for h in self.event_handlers:
            h.on_shutdown()

    def _on_round_begin(self, round_number):
        """ Internal event on round start """

        logger.debug('on_round_begin(%d)', round_number)
        for h in self.event_handlers:
            h.on_round_begin(round_number)

    def _on_new_training_data(self, round_number):
        """ Internal event on detection of new training data """

        logger.debug('on_new_training_data(%d)', round_number)
        for h in self.event_handlers:
            h.on_new_training_data(round_number)

    def _on_new_tournament_data(self, round_number):
        """ Internal event on detection of new tournament data """

        logger.debug('on_new_tournament_data(%d)', round_number)
        for h in self.event_handlers:
            h.on_new_tournament_data(round_number)

    def _on_cleanup(self, round_number):
        """ Internal event on end of round processing """

        logger.debug('on_cleanup(%d)', round_number)
        for h in self.event_handlers:
            h.on_cleanup(round_number)

    def _check_new_training_data(self, round_number):
        """
        Internal function to check if the newly downloaded dataset contains
        new training data.
        """

        logger.debug('check_new_training_data(%d)', round_number)
        if self.persistent_state['last_round_trained'] is None:
            logger.info('check_new_training_data: last_round_trained not set, '
                        'treating training data as new')
            return True

        # Check if validation data has changed
        if self.config['check_validation_data']:
            filename_old = self.get_dataset_path(self.persistent_state['last_round_trained']) / 'numerai_tournament_data.csv'
            filename_new = self.get_dataset_path(round_number) / 'numerai_tournament_data.csv'
    
            if check_dataset(filename_old, filename_new, data_type='validation'):
                return True

        filename_old = self.get_dataset_path(self.persistent_state['last_round_trained']) / 'numerai_training_data.csv'
        filename_new = self.get_dataset_path(round_number) / 'numerai_training_data.csv'

        return check_dataset(filename_old, filename_new)

    def _get_tournaments (self):
        tournaments = self.napi.get_tournaments()
        self.tournaments = {x['tournament']: x['name'] for x in tournaments}

    def _on_round_begin_internal(self, round_number):
        """ Internal event on round start """

        logger.debug('on_round_begin_internal(%d)', round_number)
        
        # Update the ID to tournament name dictionary, do this every round
        # in case of renaming of tournaments
        self._get_tournaments()
        
        # Initialize round report dictionary
        self.report = nested_defaultdict()
        self.report['round'] = round_number
        self.report['round_processing_start_time'] = datetime.datetime.now()
        
        self._on_round_begin(round_number)

        # Check if training is needed, if so call on_new_training_data
        if self._check_new_training_data(round_number):
            # Signal new training data
            self._on_new_training_data(round_number)
            self.persistent_state['last_round_trained'] = round_number

            # Immediately save state to prevent retraining if other event handlers fail
            self.save_state()

        # Signal new tournament data
        self._on_new_tournament_data(round_number)
        
        self.report['round_processing_end_time'] = datetime.datetime.now()
        
        # Signal end of round
        self._on_cleanup(round_number)
        
        print(self.report)
        
        # Reset report dictionary
        self.report = None


    def wait_till_next_round(self):
        """
        Wait until a new Numerai round is detected. Will wait until 5 minutes
        before the closing time of the current round, as reported by the
        Numerai API. Then the current round number is requested every minute
        until a new round number is received.

        Returns:
            Dictionary with the new round information.
        """

        logger.debug('wait_till_next_round')

        round_info = self.napi.get_current_round_details(tournament=self.tournament_id)
        dt_round_close = dateutil.parser.parse(round_info['closeTime'])
        dt_round_close_atstart = dt_round_close

        new_round_info = round_info

        dt_now = datetime.datetime.utcnow().replace(tzinfo=pytz.utc)
        logger.info('Waiting for round %d. Time to next round: %.1f hours',
                    self.persistent_state['last_round_processed'] + 1,
                    (dt_round_close - dt_now).total_seconds() / 3600)

        # Loop until the API reports a new round number
        while new_round_info['number'] == round_info['number']:
            dt_now = datetime.datetime.utcnow().replace(tzinfo=pytz.utc)
            
            # Update round close time in case of delays
            # Report any delays to the log file
            dt_round_close = dateutil.parser.parse(new_round_info['closeTime'])
            if dt_round_close != dt_round_close_atstart:
                logger.info('Round close time changed. Round %d got delayed by %1.f minutes',
                            self.persistent_state['last_round_processed'] + 1,
                            (dt_round_close - dt_round_close_atstart).total_seconds() / 60)
                dt_round_close_atstart = dt_round_close
            
            seconds_wait = (dt_round_close - dt_now).total_seconds() + 5

            if seconds_wait > self.config['wakeup_time']:
                # Wait till 'wakeup_time' seconds before round start
                wait_until(dt_round_close - datetime.timedelta(seconds=self.config['wakeup_time'] - 5))
            else:
                # Then query round information every 'round_wait_interval' seconds until round has started
                wait(min(seconds_wait, self.config['round_wait_interval']))

            new_round_info = self.napi.get_current_round_details(tournament=self.tournament_id)
            dt_now = datetime.datetime.utcnow().replace(tzinfo=pytz.utc)
            logger.info('Periodic check before planned round start. Current '
                        'round: %d. Time to next round: %.1f minutes',
                        new_round_info['number'],
                        (dt_round_close - dt_now).total_seconds() / 60)

        return new_round_info


    def get_dataset_path(self, round_number):
        """
        Get the base path for the data for a given round number.

        Args:
            round_number: Number of the round for which the path is requested.

        Returns:
            pathlib Path for the dataset of the requested round.
        """

        return self.config['data_directory'] / 'numerai_dataset_{}'.format(round_number)


    def _download_and_check(self):
        """
        Download a new dataset and check whether it contains new tournament
        data.

        Returns:
            True if the new dataset was validated as new data. False otherwise.
        """

        logger.debug('download_and_check')
        try:
            logger.info('Downloading dataset')
            dataset_path = self.napi.download_current_dataset(dest_path=self.config['data_directory'],
                                                                   unzip=True,
                                                                   tournament=self.tournament_id)

            filename_old = self.get_dataset_path(self.round_number - 1) / 'numerai_tournament_data.csv'
            filename_new = self.get_dataset_path(self.round_number) / 'numerai_tournament_data.csv'

            valid = check_dataset(filename_old, filename_new, data_type='live')
            
            if not valid:
                # Remove downloaded and unzipped files if dataset not new
                os.remove(dataset_path)
                if os.path.isdir(dataset_path[:-4]):
                    shutil.rmtree(dataset_path[:-4])
                    
        except requests.RequestException:
            import traceback
            msg = traceback.format_exc()
            logging.warning('Request exception: %s', msg)

            valid = False

        return valid


    def _run_new_round(self):
        """
        Internal function that downloads and verifies a new dataset and calls
        the internal event handlers.
        """

        logger.debug('run_new_round')

        # Download data. If data is not valid, wait 10 minutes and try again.
        valid = self._download_and_check()

        while not valid:
            logger.info('run_new_round: New dataset is not valid, retrying in %.1f minutes',
                        self.config['invalid_dataset_waittime']/60)

            wait(self.config['invalid_dataset_waittime'])
            valid = self._download_and_check()

        # Call round begin event
        self._on_round_begin_internal(self.round_number)

        # Save current round as the last round processed
        self.persistent_state['last_round_processed'] = self.round_number

        # Save persistent state (in case of any crash)
        self.save_state()


    def load_state(self):
        """ Load the internal state from file using pickle. """

        logger.debug('load_state')

        # Try loading the state from file
        try:
            with open('state.pickle', 'rb') as fp:
                self.persistent_state = pickle.load(fp)
        except FileNotFoundError:
            self.persistent_state = {}
        except EOFError:
            self.persistent_state = {}

        # Set last round processed and trained if it does not exist
        if 'last_round_processed' not in self.persistent_state:
            self.persistent_state['last_round_processed'] = None

        if 'last_round_trained' not in self.persistent_state:
            self.persistent_state['last_round_trained'] = None

        logger.debug('load_state: last_round_processed = %s',
                     self.persistent_state['last_round_processed'])
        logger.debug('load_state: last_round_trained = %s',
                     self.persistent_state['last_round_trained'])


    def save_state(self):
        """ Save the internal state to file using pickle. """

        logger.debug('save_state')
        logger.debug('save_state: last_round_processed = %s',
                     self.persistent_state['last_round_processed'])
        logger.debug('save_state: last_round_trained = %s',
                     self.persistent_state['last_round_trained'])

        with open('state.pickle', 'wb') as fp:
            pickle.dump(self.persistent_state, fp)


    # Run Numerauto in daemon mode
    def run(self, single_run=False):
        """
        Start the Numerauto daemon. Will process Numerai rounds until
        interrupted.

        Args:
            single_run: Indicates whether this function should only process one
            round after catching up to the current round. It will exit if the
            waiting time until the next round exceeds 24 hours. This can be used
            to start Numerauto from a task scheduler (at most 24 hours before the
            round start).
        """
        logger.debug('run')

        # Set up signal handlers to gracefully exit
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)

        # Load internal state
        self.load_state()

        # Trigger start event
        self._on_start()

        try:
            self.round_number = self.napi.get_current_round(tournament=self.tournament_id)
            if (self.persistent_state['last_round_processed'] is None or
                    self.persistent_state['last_round_trained'] is None or
                    self.round_number > self.persistent_state['last_round_processed']):
                logger.info('Current round (%d) does not appear to be processed',
                            self.round_number)
                self._run_new_round()
    
            logger.info('Entering daemon loop')
        
            while True:
                self.round_number = self.napi.get_current_round(tournament=self.tournament_id)
                # Check if we didn't already pass into the next round
                if self.round_number == self.persistent_state['last_round_processed']:
                    # In case of a single run, check whether we're not going to wait
                    # too long (> 24 hours) for the next round
                    if single_run:
                        round_info = self.napi.get_current_round_details(tournament=self.tournament_id)
    
                        dt_round_close = dateutil.parser.parse(round_info['closeTime'])
                        dt_now = datetime.datetime.utcnow().replace(tzinfo=pytz.utc)

                        if (dt_round_close - dt_now).total_seconds() > self.config['single_run_max_wait']:
                            logger.info('Single run stopping because new round is more than 1 day in the future')
                            break
    
                    # Wait till next round starts
                    round_info = self.wait_till_next_round()

                self.round_number = round_info['number']
                self._run_new_round()
    
                if single_run:
                    # Stop after processing one round
                    logger.info('Exiting daemon loop because of single_run')
                    break

        except InterruptedException:
            logger.info('Exiting daemon loop because of interrupt')
        
        # Trigger shutdown event
        self._on_shutdown()

        # Save internal state
        self.save_state()


# Make this file runnable as a standalone test without event handlers.
# Will download data and wait for each round.
if __name__ == "__main__":
    log_format = "%(asctime)s [%(levelname)8s] %(name)s: %(message)s"
    logging.basicConfig(format=log_format, level=logging.DEBUG,
                        handlers=[logging.FileHandler('debug.log'),
                                  logging.StreamHandler(sys.stdout)])
    try:
        Numerauto().run()
    except Exception as e:
        logging.exception(e)

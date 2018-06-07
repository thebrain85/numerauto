"""
Utilities for Numerauto
"""

import os
import logging
import time
import datetime
import signal

import pandas
import pytz


logger = logging.getLogger(__name__)


def check_dataset(filename_old, filename_new, data_type=None):
    """
    Checks whether two Numerai datasets are the same. Optionally it can check
    only rows with a specified data_type.

    Args:
        filename_old: Filename of the first (old) dataset
        filename_new: Filename of the second (new) dataset
        data_type: Data type of the rows to check (default: None, i.e. all rows)
    """

    logger.debug('check_dataset(%s, %s)', filename_old, filename_new)

    # Load dataset from last round and current round (if available)
    if not os.path.isfile(filename_new):
        logger.error('check_dataset: New data could not be loaded')
        return False

    if not os.path.isfile(filename_old):
        logger.info('check_dataset: No previous dataset available. Skipping check.')
        return True

    logger.info('check_dataset: Checking %s vs %s', filename_old, filename_new)

    # Read datasets
    old_dataset = pandas.read_csv(filename_old)
    new_dataset = pandas.read_csv(filename_new)

    if data_type is not None:
        # Filter only data_type from datasets
        old_dataset = old_dataset[old_dataset['data_type'] == data_type]
        new_dataset = new_dataset[new_dataset['data_type'] == data_type]

    # If the number of elements is not the same, the data is different
    if old_dataset.shape != new_dataset.shape:
        logger.debug('check_dataset: Number of elements changed')
        return True

    # Fix dataset indexing
    old_dataset.index = old_dataset.id
    new_dataset.index = new_dataset.id

    old_dataset = old_dataset.sort_index()
    new_dataset = new_dataset.sort_index()

    # Check if values are the not same
    if not new_dataset.equals(old_dataset):
        logger.debug('check_dataset: new dataset does not equal old dataset')
        return True

    # Data does not appear to have changed
    logger.debug('check_dataset: No change detected')
    return False



def wait(seconds):
    """
    Helper function that waits for a given number of seconds while checking
    the exit_requested attribute. If exit_requested is set to True, this
    function will return.

    Args:
        seconds: Number of seconds to wait.
    """

    logger.debug('wait(%d)', seconds)
    dt_now = datetime.datetime.utcnow().replace(tzinfo=pytz.utc) + datetime.timedelta(seconds=seconds)
    wait_until(dt_now)
    

def wait_until(timestamp):
    """
    Helper function that waits until a given datetime timestamp is reached,
    while checking the exit_requested attribute. If exit_requested is set
    to True, this function will return.

    Args:
        timestamp: datetime object indicating the date and time that should
                   be waited until.
    """
    logger.debug('wait_until(%s)', timestamp)
    dt_now = datetime.datetime.utcnow().replace(tzinfo=pytz.utc)

    while (timestamp - dt_now).total_seconds() > 0:
        time.sleep(min(1,(timestamp - dt_now).total_seconds()))
        dt_now = datetime.datetime.utcnow().replace(tzinfo=pytz.utc)


def wait_for_retry(attempt_number):
    logger.debug('wait_for_retry(%d)', attempt_number)

    # Hardcoded retry schedule:
    # 5x 1 minute
    # 3x 10 minutes
    # 3x 1 hour
    # Fail afterwards (3 hours 35 minutes of retrying)
    waiting_schedule = [60, 60, 60, 60, 60, 600, 600, 600, 3600, 3600, 3600]

    if attempt_number >= len(waiting_schedule):
        raise RuntimeError('Request failed too many times')

    wait(waiting_schedule[attempt_number])

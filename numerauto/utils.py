"""
Utilities for Numerauto
"""

import os
import logging

import pandas


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

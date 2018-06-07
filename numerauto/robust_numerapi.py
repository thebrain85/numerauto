"""
Module containing a robust implementation of NumerAPI.
"""

import logging

import requests
from requests.exceptions import RequestException

import numerapi

from .utils import wait_for_retry

logger = logging.getLogger(__name__)


API_TOURNAMENT_URL = 'https://api-tournament.numer.ai'


class NumerAPIAuthorizationError(Exception):
    """ Error that is raised if authorization using the Numerai API fails. """
    pass

class NumerAPIError(Exception):
    """
    General error class for failure of the Numerai API.

    Attributes:
        errors: Error dictionary returned by the Numerai API.
    """

    def __init__(self, message, errors):
        super().__init__(message)
        self.errors = errors

class RobustNumerAPI(numerapi.NumerAPI):
    """
    Robust implementation of NumerAPI.

    Checks for failure of requests and retries the requests until they succeed.
    """

    def __raw_query_patched(self, query, variables=None, authorization=False):
        """
        NumerAPI raw_query modified to not raise ValueErrors. Instead,
        NumerAPIError and NumerAPIAuthorizationError are raised.
        """

        body = {'query': query,
                'variables': variables}
        headers = {'Content-type': 'application/json',
                   'Accept': 'application/json'}
        if authorization:
            if self.token:
                public_id, secret_key = self.token
                headers['Authorization'] = \
                    'Token {}${}'.format(public_id, secret_key)
            else:
                raise NumerAPIAuthorizationError("API keys required for this action.")
        r = requests.post(API_TOURNAMENT_URL, json=body, headers=headers)
        
        # Ensure any 4xx and 5xx return codes raise an HTTPError
        r.raise_for_status()
        
        result = r.json()
        if "errors" in result:
            err = self._handle_call_error(result['errors'])
            # fail!
            raise NumerAPIError(err, result['errors'])

        return result

    def raw_query(self, query, variables=None, authorization=False):
        """
        Robust implementation of raw_query. Will retry the query if a
        RequestException is intercepted.
        """

        attempt_number = 0
        while True:
            try:
                return self.__raw_query_patched(query, variables=variables,
                                                authorization=authorization)
            except RequestException as e:
                logger.error('Request failed: %s', e)
                wait_for_retry(attempt_number)
                attempt_number += 1

                # TODO: See if we need to re-raise some request exceptions

    def upload_predictions(self, file_path, tournament=1):
        """
        Robust implementation of upload_predictions. Will retry the upload if a
        RequestException is intercepted.
        """
        
        attempt_number = 0
        while True:
            try:
                return super().upload_predictions(file_path, tournament=tournament)
            except RequestException as e:
                logger.error('Upload request failed: %s', e)
                wait_for_retry(attempt_number)
                attempt_number += 1

                # TODO: See if we need to re-raise some request exceptions

    def get_current_round_details(self, tournament=1):
        """
        Requests time details about the current round.

        Returns:
            Dictionary containing round details.
        """
        query = '''
            query($tournament: Int!) {
              rounds(tournament: $tournament
                     number: 0) {
                number
                openTime
                closeTime
                resolveTime
              }
            }
        '''
        arguments = {'tournament': tournament}

        raw = self.raw_query(query, arguments)

        # TODO: Find out if this can actually return None, and why.
        #       Ideally we don't want it to do so
        if raw is None:
            logger.error('get_current_round_details returned None')
            raise RuntimeError('get_current_round_details returned None')

        return raw['data']['rounds'][0]

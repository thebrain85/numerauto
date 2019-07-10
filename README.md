[![PyPI](https://img.shields.io/pypi/v/numerauto.svg)](https://pypi.python.org/pypi/numerauto)

# Numerauto
Numerauto is a Python daemon that facilates automatic weekly participation in
the Numerai machine learning competition (http://numer.ai).

Users can implement custom event handlers to handle new training data, apply
models to new tournament data when a new round starts, and more. Example
event handlers are included for training SKLearn classifiers and uploading
predictions.

If you encounter a problem or have suggestions, feel free to open an issue.

# Installation
To install the latest release:

`pip install --upgrade numerauto`

If you prefer to use the latest development version, clone this github reposistory:

`git clone https://github.com/thebrain85/numerauto`

And then run the following command in the numerauto directory.

`python setup.py install`


# Usage

## Example
Although Numerauto can be run without any event handlers for testing, it will
not do anything except detecting round changes and downloading the new dataset
every week.

See `example.py` for a basic example that trains a scikit-learn logistic
regression model and uploads its predictions.

The example uses the `PredictionUploader` event handler to upload predictions
to Numerai. This requires you to register an API key in your Numerai account.
This can be done in Account settings -> Your API Keys -> Add. Select
'Upload submissions' to be able to upload predictions using this API key. Replace
'insert your publickey/secretkey here' in the code with your own API public/secret
API key pair to allow the example code to upload the prediction to your account.

See `example2.py` for an example that uses the `CommandlineExecutor` event
handler to call a custom commandline once a new round is detected. You can
modify the command line to execute your own code, e.g. `python myscript.py`,
like you would do manually. This is an easy and quick way to make use of
Numerauto's automatic detection of new rounds in Numerai.

## Custom event handlers
Implementing your own event handler is easy. Simply create a subclass of
numerauto.eventhandlers.EventHandler and overload the on_* methods that you
need to implement your own functionality. The event handler can then be added
to a Numerauto instance using the `add_event_handler` method. Then start the
main loop of the Numerauto daemon by calling the `run` method.

Currently these events are supported:
- `def on_start(self)`: Called when the daemon starts.
- `def on_shutdown(self)`: Called when the daemon shuts down.
- `def on_round_begin(self, round_number)`: Called when a new round has started.
- `def on_new_training_data(self, round_number)`: Called when the daemon has detected that new training data is available.
- `def on_new_tournament_data(self, round_number)`: Called every round to signal that there is new tournament data.

Note that event handlers are called in the order they are added to the
Numerauto instance. Also note that all handlers for one event are called before
the next event is handled, keep this in mind when designing event handlers that
interact with one another, or that keep large amounts of data in memory.
Ideally, the handler should clean up memory in `on_new_tournament_data` to
prevent memory being used while the daemon is idle and waiting for the next
round.

## Running Numerauto
By default, the `run` method of Numerauto will keep running indefinitely until
interrupted using a SIGINT (ctrl-c) or SIGTERM signal. This way, you only have
to start your Numerauto script once, for example on startup, or running inside
[screen](https://www.gnu.org/software/screen/manual/screen.html) on linux.
`example.py` runs Numerauto this way.

Alternatively, you can provide the argument `single_run` to the `run` method of
Numerauto. This will cause Numerauto to only wait once for a new round (and with
a maximum of 24 hours). This is ideally suited for calling Numerauto from a
task scheduler, such as cron on Linux or the Windows Task Scheduler. Make sure
you schedule Numerauto to run a little before the official round start time,
it will wait and run as soon as the dataset is available.
`example2.py` runs Numerauto this way.

## Persistent state: state.pickle
Numerauto stores a persistent state in the `state.pickle` file in the directory
from which the daemon is being run. By default, the Numerauto daemon stores
the last round number that was processed (`last_round_processed`) and the last
round number on which training was performed (`last_round_trained`). You can
force the system to reprocess and retrain by stopping the daemon and removing
the state.pickle file.

Custom event handlers can store persistent information in the `persistent_state`
dictionary of the Numerauto instance.

## Round report
Event handlers have access to a special dictionary in the numerauto instance
via `self.numerauto.report`. This dictionary is reset every round and can be
used to report values that are relevant to the event handler. For example,
`SKLearnModelTrainer` reports back the filenames of the trained model (if
applicable) and the generated predictions, and `PredictionStatisticsGenerator`
reports the validation metrics of a prediction.

The report can be written to file every round with `BasicReportWriter`, or
emailed with `BasicReportEmailer`, both using only simple formatting.

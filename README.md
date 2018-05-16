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
`pip install --upgrade numerauto`

# Usage
Although Numerauto can be run without any event handlers for testing, it will
not do anything except detecting round changes and downloading the new dataset
every week.

See `example.py` for a basic example that trains a scikit-learn logistic
regression model and uploads its predictions.

Implementing your own event handler is easy. Simply create a subclass of
numerauto.eventhandlers.EventHandler and overload the on_* methods that you
need to implement your own functionality. The event handler can then be added
to a Numerauto instance using the `add_event_handler` method. Then start the
main loop of the Numerauto daemon by calling the `run` method. Note that this
will keep running indefinitely or until interrupted using a SIGINT (ctrl-c) or
SIGTERM signal.

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

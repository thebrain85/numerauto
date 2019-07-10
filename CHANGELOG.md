# Changelog

- v0.3.1
    * Added support for the new kazutsugi tournament

- v0.3.0
    * Added configuration entries to the Numerauto class to allow changing certain constants
    * Added report structure to Numerauto that allows event handlers to report back values, which can be written to a report file or emailed.
    * Numerauto now checks for changes in round close time, which should allow it to detect delays in round start time
    * Various bug fixes

- v0.2.0
    * Modified event handlers to support multiple tournaments.
    * Improved handling of API failures (will now follow a schedule with increasing wait times after repeated failures).
    * Improved handling of interrupts.
    * Added `CommandlineExecutor` event handler that executes a user-supplied commandline on new training and/or tournament data.
    * Added `single_run` argument to Numerauto `run` function. This will cause Numerauto to exit after handling one round, in order to support running from task schedulers.
    * Included new example script to demonstrate `CommandlineExecutor` and `single_run` features.

- v0.1.0
    * Initial release.
    * Supports detection of new Numerai rounds and the basic infrastructure for custom event handlers.

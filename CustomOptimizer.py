import keras_tuner as kt
from keras_tuner.engine import trial as trial_module, tuner_utils
import warnings
import tensorflow as tf

class CustomOptimizer(kt.BayesianOptimization):


    def search(self, *fit_args, **fit_kwargs):
        """Performs a search for best hyperparameter configuations.

        Args:
            *fit_args: Positional arguments that should be passed to
              `run_trial`, for example the training and validation data.
            **fit_kwargs: Keyword arguments that should be passed to
              `run_trial`, for example the training and validation data.
        """
        if "verbose" in fit_kwargs:
            self._display.verbose = fit_kwargs.get("verbose")
        self.on_search_begin()
        while True:
            trial = self.oracle.create_trial(self.tuner_id)
            if trial.status == trial_module.TrialStatus.STOPPED:
                # Oracle triggered exit.
                tf.get_logger().info("Oracle triggered exit")
                break
            if trial.status == trial_module.TrialStatus.IDLE:
                # Oracle is calculating, resend request.
                continue

            self.on_trial_begin(trial)

            try:
                results = self.run_trial(trial, *fit_args, **fit_kwargs)

                # `results` is None indicates user updated oracle in `run_trial()`.
                if results is None:
                    warnings.warn(
                        "`Tuner.run_trial()` returned None. It should return one of "
                        "float, dict, keras.callbacks.History, or a list of one "
                        "of these types. The use case of calling "
                        "`Tuner.oracle.update_trial()` in `Tuner.run_trial()` is "
                        "deprecated, and will be removed in the future.",
                        DeprecationWarning,
                        stacklevel=2,
                    )
                else:
                    self.oracle.update_trial(
                        trial.trial_id,
                        # Convert to dictionary before calling `update_trial()`
                        # to pass it from gRPC.
                        tuner_utils.convert_to_metrics_dict(
                            results, self.oracle.objective, "Tuner.run_trial()"
                        ),
                    )
            except Exception as e:
                print(e)
                fake_results = {
                    'loss': 100.0,
                    'accuracy': 0.0,
                    'val_loss': 100.0,
                    'val_accuracy': 0.0
                }
                self.oracle.update_trial(
                    trial.trial_id,
                    tuner_utils.convert_to_metrics_dict(
                        fake_results, self.oracle.objective, "Tuner.run_trial()"
                    ),
                )
            self.on_trial_end(trial)
        self.on_search_end()
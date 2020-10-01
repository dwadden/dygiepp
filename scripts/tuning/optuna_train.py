#!/usr/bin/env python3
"""
Tune hyperparameter search with [Optuna](https://optuna.org).

Requirements:
- `sqlite3` for storage of trials.
If do not install `sqlite3`, you can use in-memory storage: change `storage` to `None` in `optuna.create_study` (not recommended).

Usage:
- Place Jsonnet configuration file in `./training_config/` dir, see example `./training_config/ace05_event_optuna.jsonnet`:
  - Mask values of hyperparameters with Jsonnet method calling `std.extVar('{param_name}')` with `std.parseInt` for integer or `std.parseJson` for floating-point and other types.
  - Override nested default template values with `+:`, see [config.md](`./doc/config.md`).
- Edit the `objective`-function in `./scripts/tuning/optuna_train.py`:
  - Add [trial suggestions with `suggest` functions](https://optuna.readthedocs.io/en/stable/reference/generated/optuna.trial.Trial.html).
  - Change `metrics`-argument off `executor` to the relevant optimization goal.
- With the `dygiepp` modeling env activated, run: `python optuna_train.py <CONFIG_NAME>`
- The best config will be dumped at `./training_config/best_<CONFIG_NAME>.json`.

For more details see [Optuna blog](https://medium.com/optuna/hyperparameter-optimization-for-allennlp-using-optuna-54b4bfecd78b).
"""

import optuna
import argparse


def objective(trial: optuna.Trial) -> float:
    trial.suggest_int("max_span_width", 6, 10)
    trial.suggest_float("lossw_events", 0.75, 1.0)
    trial.suggest_float("lossw_relation", 0.25, 0.75)
    trial.suggest_float("lossw_ner", 0.25, 0.75)
    trial.suggest_int("feature_size", 10, 30)
    trial.suggest_int("ffwd_num_layers", 1, 3)
    trial.suggest_int("ffwd_hidden_dims", 100, 300)
    trial.suggest_float("ffwd_dropout", 0.2, 0.6)
    trial.suggest_float("relation_spans_per_word", 0.3, 0.7)
    trial.suggest_float("events_trigger_spans_per_word", 0.1, 0.8)
    trial.suggest_float("events_argument_spans_per_word", 0.4, 1.0)
    trial.suggest_float("events_lossw_trigger", 0.1, 0.4)
    trial.suggest_float("events_lossw_arguments", 0.8, 1.0)

    executor = optuna.integration.allennlp.AllenNLPExecutor(
        trial=trial,  # trial object
        config_file=f"./training_config/{args.config_name}.jsonnet",  # jsonnet path
        serialization_dir=f"./models/optuna/{trial.number}",  # directory for snapshots and logs
        metrics="best_validation_MEAN__trig_class_f1",
        include_package="dygie",
    )
    return executor.run()


if __name__ == "__main__":

    # parse console arguments for setting config filename
    parser = argparse.ArgumentParser(
        description="Hyperparameter tuning with optuna script."
    )
    parser.add_argument(
        "config_name",
        help="Filename of config inside ./training_config_dir",
    )
    args = parser.parse_args()
    global CONSOLE_ARGUMENTS
    CONSOLE_ARGUMENTS = args

    # create Optuna hyperparameter search study
    study = optuna.create_study(
        storage="sqlite:///trial.db",  # save results in DB
        sampler=optuna.samplers.TPESampler(seed=24),
        study_name=args.config_name,
        direction="maximize",
        pruner=optuna.pruners.HyperbandPruner(),
        load_if_exists=True,
    )

    # Run the search
    timeout = 60 * 60 * 10  # timeout (sec): 60*60*10 sec => 10 hours
    study.optimize(
        objective,
        n_jobs=1,  # number of processes in parallel execution
        n_trials=3,  # number of trials to train a model
        timeout=timeout,  # threshold for executing time (sec)
    )

    # Write best config found
    optuna.integration.allennlp.dump_best_config(
        f"./training_config/{args.config_name}.jsonnet",
        f"best_{args.config_name}.json",
        study,
    )

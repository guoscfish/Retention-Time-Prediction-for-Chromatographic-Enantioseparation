# ONE_ROUND_TRIAL_COMPLETE

Completed the requested one-round trial: L333 → L365 (32 acquisitions).

All 4,114 candidates were supplied in one global-selection request. The sealed successful selection was reused for this training run. Scratch training completed in 206.04 seconds, stopped at epoch 304, and retained the best checkpoint from epoch 204.

Validation RMSE: 8.043997 → 7.850580 (2.40% reduction). No test labels were accessed.

Stopped after one round as requested; the next round has not started. The registered protocol permits six rounds, but this is only a one-round result.

See `results/one_round_trial.json` for metrics, provenance and verification. Repeating `advance --round 0` verifies and reuses the completed artifacts without fitting again.

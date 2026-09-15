# Example configuration

The process loads YAML through `pyrdp_commons.cli.setup_app`. The CLI default path is `config.yml` in the working directory (`python -m regularizer -c config.yml`).

Copy the example below and adjust hosts and credentials. `temperature` bootstraps from TimescaleDB; `setpoint` is live-only.

`imputer` accepts `default`, `const_fill`, `linear`, `daily_naive`, `knn`, or `soft_threshold_svd`; `forecaster` accepts the same names except `linear`. The last three reshape the window into a day-by-slot matrix, so they need an `update_interval` that divides a day evenly and a `window` spanning several days.

```yaml
version: 1

logging:
  loggers:
    rdp-regularizer:
      level: INFO
      handlers: ["console"]
      propagate: false

redis:
  host: localhost
  port: 6379
  db: 0

timescale:
  host: localhost
  port: 5432
  db: rdp
  user: rdp
  password: changeme

channels:
  temperature:
    input_stream: measurements.temperature
    output_stream: regularized.temperature
    polling_interval: 1m
    update_interval: 1m
    window: 1h
    jitter_tolerance: 30s
    lag_time: 10s
    offset: 0s
    output_maxlen: 200
    data_provider_name: rdp-regularizer
    imputer: default
    forecaster: default
    history_provider:
      dp_name: temperature
      dp_location_code: building-a
      dp_unit: degC
      dp_data_provider: crawler
      dp_device_id: sensor-01
      init_when_source_available: true
      bootstrap_delay: 10s

  setpoint:
    input_stream: measurements.setpoint
    output_stream: regularized.setpoint
    polling_interval: 1m
    update_interval: 1m
    window: 1h
    jitter_tolerance: 30s
    lag_time: 10s
    offset: 0s
```

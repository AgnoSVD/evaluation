## Benchmark for prediction accuracy and performance comparision

```shell
cd benchmark
python python benchmark.py -s default
```

Plot results:

```shell
cd benchmark/eval/default
python plot_default.py
```


```shell
python plot_comparison.py
```

## Benchmark for evolving dataset

```shell
cd benchmark
python benchmark_evolving_ds.py -s medium
```

Plot results:

```shell
cd benchmark/eval/evolving_ds
python plot_evolving_dataset.py -s medium
```

## Benchmark for cost restriction

```shell
cd benchmark
python benchmark_cost_restriction.py
```

Plot results:

```shell
cd benchmark/eval/cost_restriction
python plot_cost_restriction.py
```

## Benchmark for number of training workloads

```shell
cd benchmark
python python benchmark.py -s n_jobs
```

```shell
cd benchmark/eval/n_jobs
python plot_job_n.py
```

## Benchmark for row density for a number of workloads

### Steady state: all 90 workloads 
```shell
cd benchmark
python python benchmark.py -s steady_state
```

```shell
cd benchmark/eval/row_density
python plot_row_den.py -s steady_state
```

### Cold start: R=2.5 or 16*2.5=40 workloads 
```shell
cd benchmark
python python benchmark.py -s cold_start
```

```shell
cd benchmark/eval/row_density
python plot_row_den.py -s cold_start
```
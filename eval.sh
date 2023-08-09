#!/bin/bash



function benchmarkPredictionAccuracy() {
    echo "======================================= Running benchmark for prediction accuracy ======================================="
    python benchmark_performance.py -s accuracy
    cd eval/default
    echo "Plotting prediction accuracy"
    python plot_default.py
    mv fig_pred_acc_* ../eval_plots/
    cd ../..
    echo "========================================================================================================================="
}
function benchmarkCostDecreaseSpeedUp() {
    echo "====================================== Running benchmark for speed up and cost decrease ======================================"
    python benchmark_cost_decrease_speedup.py
    echo "==========================================================================================================================="
}
function benchmarkPerformanceComparison() {
    echo "====================================== Running benchmark for performance comparision ======================================"
    python benchmark_performance.py -s accuracy
    cd eval/default
    echo "Plotting performance comparison"
    python plot_comparison.py
    mv fig-comparison.png ../eval_plots/
    cd ../..
    echo "==========================================================================================================================="
}
function benchmarkEvolvingDs() {
    echo "======================================= Running benchmark for evolving dataset ======================================="
    python benchmark_evolving_ds.py -s medium
    cd eval/evolving_ds
    echo "Plotting performance on evolving dataset"
    python plot_evolving_dataset.py -s medium
    mv fig_evolving_ds_* ../eval_plots/
    cd ../..
    echo "====================================================================================================================="
}
function benchmarkCostRestriction() {
    echo "======================================= Running benchmark for cost restriction ======================================="
    python benchmark_cost_restriction.py
    cd eval/cost_restriction
    echo "Plotting performance with cost restriction"
    python plot_cost_restriction.py
    mv fig_cost_restriction.png ../eval_plots/
    cd ../..
    echo "====================================================================================================================="
}
function benchmarkNumJobs() {
    echo "======================================= Running benchmark for number of training workloads ======================================="
    python benchmark_performance.py -s n_jobs
    cd eval/n_jobs
    echo "Plotting performance with number of training workloads"
    python plot_job_n.py
    mv fig_n_jobs.png ../eval_plots/
    cd ../..
    echo "=================================================================================================================================="
}
function benchmarkSteadyState() {
    echo "======================================= Running benchmark for steady-state ======================================="
    python benchmark_performance.py -s steady_state
    cd eval/row_density
    echo "Plotting performance for steady-state"
    python plot_row_den.py -s steady_state
    mv fig_row_den_99.png ../eval_plots/fig_row_den_steady.png
    cd ../..
    echo "=================================================================================================================="
}
function benchmarkColdStart() {
    echo "======================================= Running benchmark for cold-start ======================================="
    python benchmark_performance.py -s cold_start
    cd eval/row_density
    echo "Plotting performance for cold-start"
    python plot_row_den.py -s cold_start
    mv fig_row_den_40.png ../eval_plots/fig_row_den_cold.png
    cd ../..
    echo "================================================================================================================"
}

function usage() {
    echo -e "Usage: $0 [--all,--pred-acc,--perf-comp,--speedup-savings,--evolving-ds,--cost-rest,--n-jobs,--steady-state,--cold-start"]
}

if [[ $# -lt 1 ]]; then
    usage
else
    case "$1" in
    "--all" )
    benchmarkPredictionAccuracy > PredictionAccuracyBenchmark.log &
    benchmarkPerformanceComparison > PerformanceComparisonBenchmark.log &
    benchmarkCostDecreaseSpeedUp > CostDecreaseSpeedUpBenchmark.log &
    benchmarkNumJobs > NumJobsBenchmark.log &
    benchmarkSteadyState > SteadyStateBenchmark.log &
    benchmarkColdStart > ColdStartBenchmark.log &
    benchmarkEvolvingDs > EvolvingDsBenchmark.log &
    benchmarkCostRestriction > CostRestrictionBenchmark.log &
    wait
    ;;
    "--pred-acc" )
    benchmarkPredictionAccuracy
    ;;
    "--perf-comp" )
    benchmarkPerformanceComparison
    ;;
    "--evolving-ds" )
    benchmarkEvolvingDs
    ;;
    "--cost-rest" )
    benchmarkCostRestriction
    ;;
    "--n-jobs" )
    benchmarkNumJobs
    ;;
    "--steady-state" )
    benchmarkSteadyState
    ;;
    "--cold-start" )
    benchmarkColdStart
    ;;
    * )
    usage
    ;;
    esac
fi
# TODO: remove rmse and time from log
# echo "serverless applications result: "
# cat $result


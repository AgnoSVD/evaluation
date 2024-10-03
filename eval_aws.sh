#!/bin/bash



function benchmarkPredictionAccuracy() {
    echo "======================================= Running benchmark for prediction accuracy ======================================="
    python benchmark_performance_aws.py -s accuracy -al random
    cd eval/default
    echo "Plotting prediction accuracy"
    python plot_default.py 32
    mv fig_pred_acc_als.pdf ../eval_plots/fig_pred_acc_als_aws.pdf
    mv fig_pred_acc_sgd.pdf ../eval_plots/fig_pred_acc_sgd_aws.pdf
    cd ../..
    echo "========================================================================================================================="
}
function benchmarkPerformanceComparison() {
    echo "====================================== Running benchmark for performance comparision ======================================"
    python benchmark_performance_aws.py -s accuracy -al random
    cd eval/default
    echo "Plotting performance comparison"
    python plot_comparison.py 32
    mv fig-comparison.pdf ../eval_plots/fig-comparison_aws.pdf
    cd ../..
    echo "==========================================================================================================================="
}
function benchmarkNumJobs() {
    echo "======================================= Running benchmark for number of training workloads ======================================="
    python benchmark_performance_aws.py -s n_jobs -al $1
    cd eval/n_jobs
    echo "Plotting performance with number of training workloads"
    python plot_job_n.py -n 32 -al $1
    mv fig_n_jobs_$1.pdf ../eval_plots/fig_n_jobs_$1_aws.pdf
    cd ../..
    echo "=================================================================================================================================="
}
function benchmarkSteadyState() {
    echo "======================================= Running benchmark for steady-state ======================================="
    python benchmark_performance_aws.py -s steady_state -al random
    cd eval/row_density
    echo "Plotting performance for steady-state"
    python plot_row_den.py -s steady_state -al random
    mv fig_row_den_99.png ../eval_plots/fig_row_den_steady.png
    cd ../..
    echo "=================================================================================================================="
}
function benchmarkColdStart() {
    echo "======================================= Running benchmark for cold-start ======================================="
    python benchmark_performance_aws.py -s cold_start -al $1
    cd eval/row_density
    echo "Plotting performance for cold-start"
    python plot_row_den.py -s cold_start -al $1
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
    benchmarkNumJobs variance> NumJobsBenchmark.log &
    benchmarkSteadyState > SteadyStateBenchmark.log &
    benchmarkColdStart > ColdStartBenchmark.log &
    wait
    ;;
    "--pred-acc" )
    benchmarkPredictionAccuracy
    ;;
    "--perf-comp" )
    benchmarkPerformanceComparison
    ;;
    "--n-jobs" )
    benchmarkNumJobs variance
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


#!/bin/bash



function benchmarkPredictionAccuracy() {
    echo "======================================= Running benchmark for prediction accuracy ======================================="
    python benchmark_performance_sep.py -s accuracy -al random
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
    python benchmark_performance_sep.py -s accuracy -al random
    cd eval/default
    echo "Plotting performance comparison"
    python plot_comparison.py
    mv fig-comparison.pdf ../eval_plots/
    cd ../..
    echo "==========================================================================================================================="
}
function benchmarkEvolvingDs() {
    echo "======================================= Running benchmark for evolving dataset ======================================="
    python benchmark_evolving_ds_sep.py -s medium
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
    mv fig_cost_restriction.pdf ../eval_plots/
    cd ../..
    echo "====================================================================================================================="
}
function benchmarkNumJobs() {
    echo "======================================= Running benchmark for number of training workloads ======================================="
    python benchmark_performance_sep.py -s n_jobs -al $1
    cd eval/n_jobs
    echo "Plotting performance with number of training workloads"
    python plot_job_n.py -n 100 -al $1
    mv fig_n_jobs_$1.pdf ../eval_plots/
    cd ../..
    echo "=================================================================================================================================="
}
function benchmarkSteadyState() {
    echo "======================================= Running benchmark for steady-state ======================================="
    python benchmark_performance_sep.py -s steady_state -al random
    cd eval/row_density
    echo "Plotting performance for steady-state"
    python plot_row_den.py -s steady_state -al random
    mv fig_row_den_steady_random.pdf ../eval_plots/fig_row_den_steady_random.pdf
    cd ../..
    echo "=================================================================================================================="
}
function benchmarkColdStart() {
    echo "======================================= Running benchmark for cold-start with al_strategy $1 ======================================="
    python benchmark_performance_sep.py -s cold_start -al $1
    cd eval/row_density
    echo "Plotting performance for cold-start"
    python plot_row_den.py -s cold_start -al $1
    mv fig_row_den_cold_$1.pdf ../eval_plots/fig_row_den_cold_$1.pdf
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
    benchmarkColdStart similarity > ColdStartBenchmark.log
    benchmarkColdStart variance > ColdStartBenchmark.log
    benchmarkColdStart random > ColdStartBenchmark.log
    benchmarkSteadyState > SteadyStateBenchmark.log
    benchmarkNumJobs similarity > NumJobsBenchmark.log
    benchmarkNumJobs variance > NumJobsBenchmark.log
    benchmarkNumJobs random > NumJobsBenchmark.log
    benchmarkPredictionAccuracy > PredictionAccuracyBenchmark.log
    benchmarkPerformanceComparison > PerformanceComparisonBenchmark.log
    benchmarkCostDecreaseSpeedUp > CostDecreaseSpeedUpBenchmark.log
    benchmarkEvolvingDs > EvolvingDsBenchmark.log
    benchmarkCostRestriction > CostRestrictionBenchmark.log
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
    benchmarkNumJobs similarity > NumJobsBenchmark.log
    benchmarkNumJobs variance > NumJobsBenchmark.log
    benchmarkNumJobs random > NumJobsBenchmark.log
    ;;
    "--steady-state" )
    benchmarkSteadyState
    ;;
    "--cold-start" )
#    benchmarkColdStart similarity > ColdStartBenchmark.log
    benchmarkColdStart variance > ColdStartBenchmark.log
#    benchmarkColdStart random > ColdStartBenchmark.log
    ;;
    * )
    usage
    ;;
    esac
fi
# TODO: breaking down the resource allocation problem to different dimensions i.e. cpu and memory: improvement in terms of time and accuracy
# TODO: Use AL to selectively reduce number of rows: improvement in terms of time and accuracy
# TODO: Comparison with COSE/ bayesian optimization
# TODO: add perf comparison in terms of CDF
# TODO: add perf comparison in terms cpu and memory matches/accuracy
# TODO: update figures to pdf
# TODO: update the  back-of-envelop estimates of resource availabilities to avoid scheduling nightmare at the controller.
# TODO: evaluate the operational overhead of the system
# TODO: Test on AWS lambda data by collecting execution data using aws lambda power tuning
# TODO: Add separate results for chain functions

# echo "serverless applications result: "
# cat $result


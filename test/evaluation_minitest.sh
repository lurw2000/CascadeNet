#!/bin/bash
# Ensure you're in the project root directory

# Store results
results=()

# Function to run test and track result  
run_test() {
    echo "Running: $1"
    cd "$2"
    
    # Capture output to a temp file
    temp_log=$(mktemp)
    eval "$3" > "$temp_log" 2>&1
    exit_code=$?
    
    # Check for error patterns even if exit code is 0
    if grep -q "Error\|Exception\|Traceback\|ModuleNotFoundError\|ImportError\|Failed" "$temp_log"; then
        results+=("✗ $1 - FAILED (Error detected)")
        echo "✗ $1 failed - Error detected"
        echo "Error output:"
        grep -A 2 -B 2 "Error\|Exception\|Traceback\|ModuleNotFoundError\|ImportError\|Failed" "$temp_log" | head -10 | sed 's/^/  /'
        echo ""
    elif [ $exit_code -ne 0 ]; then
        results+=("✗ $1 - FAILED (Exit code: $exit_code)")
        echo "✗ $1 failed - Exit code: $exit_code"
        echo "Last few lines:"
        tail -5 "$temp_log" | sed 's/^/  /'
        echo ""
    else
        results+=("✓ $1 - PASSED")
        echo "✓ $1 completed"
    fi
    
    rm "$temp_log"
    cd - > /dev/null
}

# Run all tests
run_test "Burst Analysis" "evaluation/burst_analysis" "python burst.py --test"
run_test "Syn Real Div" "evaluation/syn_real_div" "python run.py --test"  
run_test "Scalability" "evaluation/scalability" "python run.py --test"
run_test "Throughput Prediction" "evaluation/throughput_prediction_steps" "bash test.sh"
run_test "Stats" "evaluation/stats" "bash plot_test.sh"
run_test "Anomaly Detection" "evaluation/anomly_detection_cross" "bash test.sh"

wait

# Show summary
echo ""
echo "========== TEST SUMMARY =========="
failed_count=0
for result in "${results[@]}"; do
    echo "$result"
    if [[ "$result" == *"FAILED"* ]]; then
        ((failed_count++))
    fi
done
echo "=================================="

if [ $failed_count -gt 0 ]; then
    echo "WARNING: $failed_count test(s) failed"
    exit 1
else
    echo "All tests passed!"
    exit 0
fi
#!/bin/bash

# EAGLE Overhead vs Performance Trade-off Analysis
# Tests different action cache step sizes to analyze the trade-off between
# RL policy call frequency and inference performance

echo "=== EAGLE Overhead vs Performance Trade-off Analysis ==="
echo "This script tests different action cache step sizes to analyze trade-offs"
echo ""

# Configuration
DATE=$(date '+%Y%m%d_%H%M%S')
TEST_DIR="log/${DATE}_tradeoff_analysis"
mkdir -p "$TEST_DIR"

MODEL_PATH="yuhuili/EAGLE3-LLaMA3.1-Instruct-8B"
BASE_MODEL_PATH="meta-llama/Llama-3.1-8B-Instruct"
QUESTION_END=200  # Small number for quick analysis
NET_ARCH="128,128"

# Test different cache step sizes
declare -a CACHE_STEPS=(1 5 10 20 50)
declare -a CACHE_DESCRIPTIONS=(
    "No Caching (Every Step)"
    "Light Caching (5 Steps)"
    "Medium Caching (10 Steps)"
    "Heavy Caching (20 Steps)"
    "Max Caching (50 Steps)"
)

echo "Test Directory: $TEST_DIR"
echo "Questions: 0-$QUESTION_END"
echo "Network Architecture: $NET_ARCH"
echo "Cache Step Configurations: ${CACHE_STEPS[*]}"
echo ""

# Create results summary file
RESULTS_FILE="$TEST_DIR/tradeoff_results.csv"
echo "Cache_Steps,Description,LLM_Inference_Time_s,Tokens_Per_Second,RL_Policy_Calls,RL_Policy_Time_ms,RL_Policy_Avg_ms,SentenceBERT_Calls,SentenceBERT_Time_ms,SentenceBERT_Avg_ms,Total_Questions,Total_Overhead_ms" > "$RESULTS_FILE"

# Function to extract metrics from log file
extract_metrics() {
    local log_file=$1
    local cache_steps=$2
    local description="$3"
    
    echo "Extracting metrics from: $log_file"
    
    # Extract overhead analysis section
    if grep -q "Overhead Analysis" "$log_file"; then
        # Get the overhead analysis table
        overhead_section=$(sed -n '/Overhead Analysis/,/TOTAL MEASURED/p' "$log_file")
        
        # Extract LLM inference time (Model_Generation components)
        llm_time=$(echo "$overhead_section" | grep "Model_Generation" | awk '{sum+=$3} END {print sum+0}')
        
        # Extract RL Policy prediction metrics
        rl_calls=$(echo "$overhead_section" | grep "RL_Policy_Prediction" | awk '{sum+=$2} END {print sum+0}')
        rl_time_ms=$(echo "$overhead_section" | grep "RL_Policy_Prediction" | awk '{sum+=$3*1000} END {print sum+0}')
        rl_avg_ms=$(echo "$overhead_section" | grep "RL_Policy_Prediction" | awk '{sum+=$4} END {if(NR>0) print sum/NR; else print 0}')
        
        # Extract SentenceBERT metrics  
        sbert_calls=$(echo "$overhead_section" | grep "SentenceBERT" | awk '{sum+=$2} END {print sum+0}')
        sbert_time_ms=$(echo "$overhead_section" | grep "SentenceBERT" | awk '{sum+=$3*1000} END {print sum+0}')
        sbert_avg_ms=$(echo "$overhead_section" | grep "SentenceBERT" | awk '{sum+=$4} END {if(NR>0) print sum/NR; else print 0}')
        
        # Calculate total overhead
        total_overhead_ms=$(echo "$rl_time_ms + $sbert_time_ms" | bc -l)
        
        # Extract performance metrics
        questions_processed=$(grep -o "Processing questions:.*100%" "$log_file" | tail -1 | grep -o "/[0-9]*" | cut -d'/' -f2)
        if [ -z "$questions_processed" ]; then
            questions_processed=$QUESTION_END
        fi
        
        # Calculate tokens per second (approximate from generation time)
        if [ $(echo "$llm_time > 0" | bc) -eq 1 ]; then
            # Estimate total tokens (approximate)
            total_tokens_estimate=$(echo "$questions_processed * 300" | bc)  # ~300 tokens per question average
            tokens_per_second=$(echo "scale=2; $total_tokens_estimate / $llm_time" | bc)
        else
            tokens_per_second=0
        fi
        
        # Write to CSV
        echo "$cache_steps,$description,$llm_time,$tokens_per_second,$rl_calls,$rl_time_ms,$rl_avg_ms,$sbert_calls,$sbert_time_ms,$sbert_avg_ms,$questions_processed,$total_overhead_ms" >> "$RESULTS_FILE"
        
        echo "  Cache Steps: $cache_steps"
        echo "  LLM Time: ${llm_time}s"
        echo "  Tokens/sec: $tokens_per_second"
        echo "  RL Calls: $rl_calls (${rl_time_ms}ms total, ${rl_avg_ms}ms avg)"
        echo "  SBERT Calls: $sbert_calls (${sbert_time_ms}ms total, ${sbert_avg_ms}ms avg)"
        echo "  Total Overhead: ${total_overhead_ms}ms"
        
    else
        echo "  ❌ No overhead analysis found in log file"
        # Write empty row
        echo "$cache_steps,$description,0,0,0,0,0,0,0,0,$QUESTION_END,0" >> "$RESULTS_FILE"
    fi
    echo ""
}

# Run tests for each cache step configuration
for i in "${!CACHE_STEPS[@]}"; do
    cache_steps="${CACHE_STEPS[$i]}"
    description="${CACHE_DESCRIPTIONS[$i]}"
    test_name="cache_${cache_steps}_steps"
    
    echo "=== Test $((i+1))/${#CACHE_STEPS[@]}: $description (Cache Steps: $cache_steps) ==="
    
    # Create test-specific directory
    test_dir="$TEST_DIR/$test_name"
    mkdir -p "$test_dir/checkpoints"
    
    # Run the test
    PYTHONUNBUFFERED=1 python -m eagle.evaluation.gen_ea_answer_llama3chat_rl_measure \
        --ea-model-path "$MODEL_PATH" \
        --base-model-path "$BASE_MODEL_PATH" \
        --model-id "tradeoff_cache_${cache_steps}" \
        --question-file eagle/data/rl_training/question.jsonl \
        --question-begin 0 \
        --question-end "$QUESTION_END" \
        --answer-file "$test_dir/answers.jsonl" \
        --num-choices 1 \
        --num-gpus-per-model 1 \
        --num-gpus-total 1 \
        --max-gpu-memory "80GiB" \
        --dtype float16 \
        --temperature 0.0 \
        --use-online-rl \
        --use-optimized-sb3-discrete-ppo \
        --optimized-policy-version ofl \
        --online-lr 3e-4 \
        --ppo-net-arch "$NET_ARCH" \
        --enable-max-entropy \
        --max-entropy-ent-coef 0.1 \
        --inference-temperature 1.5 \
        --max-entropy-inference \
        --use-eagle3-features \
        --enable-overhead-measurement \
        --action-cache-steps "$cache_steps" \
        --action-cache-enabled \
        --checkpoint-dir "$test_dir/checkpoints" \
        --no-resume \
        --total-token 60 \
        --depth 7 \
        --top-k 10 \
        --use-stepwise-rl \
        --use-eagle3 2>&1 | tee "$test_dir/overhead_log.txt"
    
    # Extract metrics from this test
    extract_metrics "$test_dir/overhead_log.txt" "$cache_steps" "$description"
    
    echo "Test $((i+1)) completed: $description"
    echo "Waiting 5 seconds before next test..."
    sleep 5
done

echo ""
echo "=== All Trade-off Tests Complete ==="
echo "Results saved to: $RESULTS_FILE"
echo ""

# Generate Python script for visualization
PLOT_SCRIPT="$TEST_DIR/generate_tradeoff_chart.py"
cat > "$PLOT_SCRIPT" << 'EOF'
#!/usr/bin/env python3

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import sys
import os

def generate_tradeoff_chart(csv_file):
    """Generate trade-off bar chart with dual y-axes"""
    
    # Read the data
    try:
        df = pd.read_csv(csv_file)
        print(f"📊 Loaded data from {csv_file}")
        print(f"   Rows: {len(df)}, Columns: {len(df.columns)}")
        print("\nData preview:")
        print(df)
        print()
    except Exception as e:
        print(f"❌ Error reading CSV file: {e}")
        return
    
    if len(df) == 0:
        print("❌ No data found in CSV file")
        return
    
    # Set up the plot with dual y-axes
    fig, ax1 = plt.subplots(figsize=(14, 8))
    ax2 = ax1.twinx()
    
    # X-axis: Cache Steps
    x = np.arange(len(df))
    width = 0.35
    
    # Left Y-axis: Performance (Tokens per Second)
    bars1 = ax1.bar(x - width/2, df['Tokens_Per_Second'], width, 
                    label='Tokens per Second', color='steelblue', alpha=0.8)
    
    # Right Y-axis: Overhead (Total Overhead in ms)
    bars2 = ax2.bar(x + width/2, df['Total_Overhead_ms'], width,
                    label='Total Overhead (ms)', color='orangered', alpha=0.8)
    
    # Customize axes
    ax1.set_xlabel('RL Policy Call Frequency (Action Cache Steps)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Performance (Tokens/Second)', color='steelblue', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Overhead Time (milliseconds)', color='orangered', fontsize=12, fontweight='bold')
    
    # Set x-axis labels
    ax1.set_xticks(x)
    ax1.set_xticklabels([f"{int(steps)}\n({desc})" for steps, desc in 
                        zip(df['Cache_Steps'], df['Description'])], 
                       rotation=0, ha='center')
    
    # Color the y-axis labels
    ax1.tick_params(axis='y', labelcolor='steelblue')
    ax2.tick_params(axis='y', labelcolor='orangered')
    
    # Add value labels on bars
    for i, bar in enumerate(bars1):
        height = bar.get_height()
        ax1.annotate(f'{height:.1f}', xy=(bar.get_x() + bar.get_width()/2, height),
                    xytext=(0, 3), textcoords="offset points", ha='center', va='bottom',
                    color='steelblue', fontweight='bold', fontsize=9)
    
    for i, bar in enumerate(bars2):
        height = bar.get_height()
        ax2.annotate(f'{height:.1f}', xy=(bar.get_x() + bar.get_width()/2, height),
                    xytext=(0, 3), textcoords="offset points", ha='center', va='bottom',
                    color='orangered', fontweight='bold', fontsize=9)
    
    # Add legends
    ax1.legend(loc='upper left')
    ax2.legend(loc='upper right')
    
    # Title
    plt.title('EAGLE RL Policy Call Frequency vs Performance Trade-off\n'
              'Cache Steps: Lower = More RL Calls, Higher = Fewer RL Calls', 
              fontsize=14, fontweight='bold', pad=20)
    
    # Grid for better readability
    ax1.grid(True, alpha=0.3)
    
    # Add subplot for detailed breakdown
    fig.subplots_adjust(bottom=0.2)
    
    # Add detailed breakdown table
    table_data = []
    for _, row in df.iterrows():
        table_data.append([
            f"{int(row['Cache_Steps'])}",
            f"{row['RL_Policy_Calls']:.0f}",
            f"{row['RL_Policy_Avg_ms']:.2f}",
            f"{row['SentenceBERT_Calls']:.0f}",
            f"{row['SentenceBERT_Avg_ms']:.2f}",
            f"{row['Tokens_Per_Second']:.1f}"
        ])
    
    # Create table
    table = plt.table(cellText=table_data,
                     colLabels=['Cache\nSteps', 'RL Policy\nCalls', 'RL Avg\n(ms)', 
                               'SBERT\nCalls', 'SBERT Avg\n(ms)', 'Performance\n(tok/s)'],
                     cellLoc='center',
                     loc='bottom',
                     bbox=[0.1, -0.25, 0.8, 0.15])
    
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.5)
    
    # Style the table
    for i in range(len(df) + 1):  # +1 for header
        for j in range(6):
            cell = table[(i, j)]
            if i == 0:  # Header
                cell.set_facecolor('#40466e')
                cell.set_text_props(weight='bold', color='white')
            else:  # Data rows
                if i % 2 == 0:
                    cell.set_facecolor('#f1f1f2')
                else:
                    cell.set_facecolor('white')
    
    plt.tight_layout()
    
    # Save the chart
    output_dir = os.path.dirname(csv_file)
    chart_path = os.path.join(output_dir, 'tradeoff_chart.png')
    plt.savefig(chart_path, dpi=300, bbox_inches='tight')
    print(f"📈 Trade-off chart saved to: {chart_path}")
    
    # Show the plot
    plt.show()
    
    # Generate summary analysis
    print("\n" + "="*60)
    print("📊 TRADE-OFF ANALYSIS SUMMARY")
    print("="*60)
    
    if len(df) > 1:
        # Find optimal configuration
        # Normalize performance and overhead for comparison
        perf_normalized = (df['Tokens_Per_Second'] - df['Tokens_Per_Second'].min()) / (df['Tokens_Per_Second'].max() - df['Tokens_Per_Second'].min() + 1e-8)
        overhead_normalized = (df['Total_Overhead_ms'] - df['Total_Overhead_ms'].min()) / (df['Total_Overhead_ms'].max() - df['Total_Overhead_ms'].min() + 1e-8)
        
        # Trade-off score: high performance, low overhead
        df['tradeoff_score'] = perf_normalized - overhead_normalized
        optimal_idx = df['tradeoff_score'].idxmax()
        
        print(f"🎯 OPTIMAL CONFIGURATION:")
        print(f"   Cache Steps: {int(df.loc[optimal_idx, 'Cache_Steps'])}")
        print(f"   Description: {df.loc[optimal_idx, 'Description']}")
        print(f"   Performance: {df.loc[optimal_idx, 'Tokens_Per_Second']:.1f} tokens/sec")
        print(f"   Total Overhead: {df.loc[optimal_idx, 'Total_Overhead_ms']:.1f} ms")
        print(f"   RL Policy Calls: {df.loc[optimal_idx, 'RL_Policy_Calls']:.0f}")
        print()
        
        # Performance analysis
        max_perf_idx = df['Tokens_Per_Second'].idxmax()
        min_overhead_idx = df['Total_Overhead_ms'].idxmin()
        
        print(f"🚀 HIGHEST PERFORMANCE:")
        print(f"   Cache Steps: {int(df.loc[max_perf_idx, 'Cache_Steps'])}")
        print(f"   Performance: {df.loc[max_perf_idx, 'Tokens_Per_Second']:.1f} tokens/sec")
        print(f"   Overhead: {df.loc[max_perf_idx, 'Total_Overhead_ms']:.1f} ms")
        print()
        
        print(f"⚡ LOWEST OVERHEAD:")
        print(f"   Cache Steps: {int(df.loc[min_overhead_idx, 'Cache_Steps'])}")
        print(f"   Overhead: {df.loc[min_overhead_idx, 'Total_Overhead_ms']:.1f} ms")
        print(f"   Performance: {df.loc[min_overhead_idx, 'Tokens_Per_Second']:.1f} tokens/sec")
        print()
        
        # Cache efficiency analysis
        print(f"📈 CACHE EFFICIENCY INSIGHTS:")
        for _, row in df.iterrows():
            cache_steps = int(row['Cache_Steps'])
            rl_calls = row['RL_Policy_Calls']
            total_questions = row['Total_Questions']
            
            # Estimate total inference steps (approximate)
            estimated_steps = total_questions * 50  # ~50 steps per question
            cache_efficiency = (estimated_steps - rl_calls) / estimated_steps * 100 if estimated_steps > 0 else 0
            
            print(f"   Cache {cache_steps}: {cache_efficiency:.1f}% cache hits, {rl_calls:.0f} RL calls")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python generate_tradeoff_chart.py <csv_file>")
        sys.exit(1)
    
    csv_file = sys.argv[1]
    if not os.path.exists(csv_file):
        print(f"❌ CSV file not found: {csv_file}")
        sys.exit(1)
    
    generate_tradeoff_chart(csv_file)
EOF

echo "=== Generating Trade-off Visualization ==="
echo "Running Python script to generate chart..."
python3 "$PLOT_SCRIPT" "$RESULTS_FILE"

echo ""
echo "=== Trade-off Analysis Complete ==="
echo ""
echo "📁 Results directory: $TEST_DIR"
echo "📊 Data file: $RESULTS_FILE"
echo "📈 Chart: $TEST_DIR/tradeoff_chart.png"
echo "🐍 Plot script: $PLOT_SCRIPT"
echo ""
echo "To regenerate the chart:"
echo "  python3 $PLOT_SCRIPT $RESULTS_FILE"
echo ""
echo "Key insights:"
echo "  - Lower cache steps = More RL policy calls = Higher overhead"
echo "  - Higher cache steps = Fewer RL policy calls = Lower overhead"
echo "  - Optimal balance depends on the specific use case"

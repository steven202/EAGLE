#!/bin/bash

# Multi-GPU Script Generator for EAGLE RL Training
# This script generates multiple GPU-specific training scripts based on user preferences
#
# Usage:
#   bash generate_multi_gpu_scripts.sh                    # Interactive mode
#   bash generate_multi_gpu_scripts.sh 1 2 3 4 5 6 7 8   # All combinations
#   bash generate_multi_gpu_scripts.sh 1 3 5 7            # Specific combinations

echo "=== EAGLE RL Multi-GPU Script Generator ==="
echo ""

# Set the base script name (fixed)
BASE_SCRIPT="test_optimized_ppo_modes_comparison_single"
if [ ! -f "${BASE_SCRIPT}.sh" ]; then
    echo "Error: ${BASE_SCRIPT}.sh not found!"
    exit 1
fi
echo "Using base script: ${BASE_SCRIPT}.sh"

# Create a shorter prefix for generated files (first 15 characters)
if [ ${#BASE_SCRIPT} -gt 7 ]; then
    FILE_PREFIX="${BASE_SCRIPT:0:7}"
    echo "Using shortened prefix for files: $FILE_PREFIX"
else
    FILE_PREFIX="$BASE_SCRIPT"
fi

# Get number of available GPUs
read -p "Enter number of available GPUs (default: 4): " NUM_GPUS
if [ -z "$NUM_GPUS" ]; then
    NUM_GPUS=4
    echo "Using default: $NUM_GPUS GPUs"
fi

# Define all possible combinations
declare -a COMBINATIONS=()

echo ""
echo "Available combinations:"
echo "1. Standard Version + Standard State + Max Entropy"
echo "2. Standard Version + Standard State + No Max Entropy"
echo "3. Standard Version + Context Only + Max Entropy"
echo "4. Standard Version + Context Only + No Max Entropy"
echo "5. OFL Version + Standard State + Max Entropy"
echo "6. OFL Version + Standard State + No Max Entropy"
echo "7. OFL Version + Context Only + Max Entropy"
echo "8. OFL Version + Context Only + No Max Entropy"
echo ""

# Check for command line arguments
if [ $# -gt 0 ]; then
    SELECTED_COMBINATIONS="$*"
    echo "Using command line arguments: $SELECTED_COMBINATIONS"
else
    # Get user selection
    echo ""
    echo "Enter combination numbers to run:"
    echo "  - Enter 'all' for all 8 combinations"
    echo "  - Enter specific numbers (space-separated, e.g., 1 3 5 7)"
    echo "  - Enter '1 2 3 4 5 6 7 8' for all combinations"
    echo "  - Press Enter for default (all 8 combinations)"
    read -p "Your choice: " SELECTED_COMBINATIONS
    
    # Set default if empty
    if [ -z "$SELECTED_COMBINATIONS" ]; then
        SELECTED_COMBINATIONS="1 2 3 4 5 6 7 8"
        echo "Using default: $SELECTED_COMBINATIONS"
    fi
fi

# Parse selected combinations
if [[ "$SELECTED_COMBINATIONS" == "all" ]]; then
    COMBINATIONS=(1 2 3 4 5 6 7 8)
else
    for combo in $SELECTED_COMBINATIONS; do
        if [ "$combo" -ge 1 ] && [ "$combo" -le 8 ]; then
            COMBINATIONS+=($combo)
        else
            echo "Warning: Invalid combination number $combo, skipping..."
        fi
    done
fi

if [ ${#COMBINATIONS[@]} -eq 0 ]; then
    echo "Error: No valid combinations selected!"
    exit 1
fi

echo ""
echo "Selected combinations: ${COMBINATIONS[*]}"
echo "Number of scripts to generate: ${#COMBINATIONS[@]}"
echo "Available GPUs: $NUM_GPUS"

# Ask about output directory
echo ""
read -p "Use today's date for generated folder? (y/n, default: y): " USE_TODAY_DATE
if [[ "$USE_TODAY_DATE" =~ ^[Nn]$ ]]; then
    OUTPUT_DIR="${BASE_SCRIPT:0:7}__multi_gpu"
    echo "Using simple folder name: $OUTPUT_DIR"
else
    OUTPUT_DIR="${BASE_SCRIPT:0:7}__multi_gpu_$(date +%H%M%S)"
    echo "Using dated folder name: $OUTPUT_DIR"
fi

# Check if any existing generated scripts exist in current directory
EXISTING_SCRIPTS=$(ls ${FILE_PREFIX}__gpu*_*.sh 2>/dev/null | wc -l)
if [ $EXISTING_SCRIPTS -gt 0 ]; then
    echo "Found $EXISTING_SCRIPTS existing generated scripts in current directory."
    read -p "Do you want to delete existing scripts and regenerate? (y/n, default: n): " DELETE_EXISTING
    if [[ "$DELETE_EXISTING" =~ ^[Yy]$ ]]; then
        echo "Deleting existing generated scripts..."
        rm -f ${FILE_PREFIX}__gpu*_*.sh
        rm -f run_all_scripts.sh launch.sh script_summary.txt
        echo "Existing scripts deleted."
    else
        echo "Keeping existing scripts. New scripts will be generated alongside them."
    fi
fi

# Check if output directory already exists (shouldn't happen with HMS suffix, but just in case)
if [ -d "$OUTPUT_DIR" ]; then
    echo "Output directory $OUTPUT_DIR already exists."
    read -p "Do you want to delete it and regenerate? (y/n, default: n): " DELETE_DIR
    if [[ "$DELETE_DIR" =~ ^[Yy]$ ]]; then
        echo "Deleting existing directory: $OUTPUT_DIR"
        rm -rf "$OUTPUT_DIR"
    else
        echo "Keeping existing directory. Will create a new one with different timestamp."
        # Generate a new timestamp to avoid conflict
        OUTPUT_DIR="${BASE_SCRIPT:0:7}__multi_gpu_$(date +%H%M%S)"
        echo "Using new directory name: $OUTPUT_DIR"
    fi
fi

mkdir -p "$OUTPUT_DIR"
echo ""
echo "Generating scripts in directory: $OUTPUT_DIR"

# Function to get combination parameters
get_combination_params() {
    local combo=$1
    case $combo in
        1) echo "1 0 1 0 1 0" ;;  # Standard Version + Standard State + Max Entropy
        2) echo "1 0 1 0 0 1" ;;  # Standard Version + Standard State + No Max Entropy
        3) echo "1 0 0 1 1 0" ;;  # Standard Version + Context Only + Max Entropy
        4) echo "1 0 0 1 0 1" ;;  # Standard Version + Context Only + No Max Entropy
        5) echo "0 1 1 0 1 0" ;;  # OFL Version + Standard State + Max Entropy
        6) echo "0 1 1 0 0 1" ;;  # OFL Version + Standard State + No Max Entropy
        7) echo "0 1 0 1 1 0" ;;  # OFL Version + Context Only + Max Entropy
        8) echo "0 1 0 1 0 1" ;;  # OFL Version + Context Only + No Max Entropy
    esac
}

# Function to get combination description
get_combination_desc() {
    local combo=$1
    case $combo in
        1) echo "standard_hiddstat_maxent" ;;
        2) echo "standard_hiddstat_noent" ;;
        3) echo "standard_context_maxent" ;;
        4) echo "standard_context_noent" ;;
        5) echo "ofl_hiddstat_maxent" ;;
        6) echo "ofl_hiddstat_noent" ;;
        7) echo "ofl_context_maxent" ;;
        8) echo "ofl_context_noent" ;;
    esac
}

# Generate scripts
for i in "${!COMBINATIONS[@]}"; do
    combo=${COMBINATIONS[$i]}
    gpu_id=$((i % NUM_GPUS))
    desc=$(get_combination_desc $combo)
    script_name="${FILE_PREFIX}__gpu${gpu_id}_${desc}.sh"
    
    echo "Generating script $((i+1))/${#COMBINATIONS[@]}: $script_name (GPU $gpu_id, $desc)"
    
    # Read the base script
    cp "${BASE_SCRIPT}.sh" "$OUTPUT_DIR/$script_name"
    
    # Get parameters for this combination
    params=$(get_combination_params $combo)
    read -r RUN_STANDARD_VERSION RUN_OFL_VERSION RUN_STANDARD RUN_CONTEXT_ONLY RUN_MAX_ENTROPY RUN_NO_MAX_ENTROPY <<< "$params"
    
    # Replace the configuration section
    sed -i "s/RUN_STANDARD_VERSION=.*/RUN_STANDARD_VERSION=$RUN_STANDARD_VERSION   # Run standard policy version/" "$OUTPUT_DIR/$script_name"
    sed -i "s/RUN_OFL_VERSION=.*/RUN_OFL_VERSION=$RUN_OFL_VERSION       # Run OFL policy version with enhanced features/" "$OUTPUT_DIR/$script_name"
    sed -i "s/RUN_STANDARD=.*/RUN_STANDARD=$RUN_STANDARD          # Run without --use-context-only-state/" "$OUTPUT_DIR/$script_name"
    sed -i "s/RUN_CONTEXT_ONLY=.*/RUN_CONTEXT_ONLY=$RUN_CONTEXT_ONLY      # Run with --use-context-only-state/" "$OUTPUT_DIR/$script_name"
    sed -i "s/RUN_MAX_ENTROPY=.*/RUN_MAX_ENTROPY=$RUN_MAX_ENTROPY       # Run with max-entropy PPO/" "$OUTPUT_DIR/$script_name"
    sed -i "s/RUN_NO_MAX_ENTROPY=.*/RUN_NO_MAX_ENTROPY=$RUN_NO_MAX_ENTROPY    # Run without max-entropy (standard PPO)/" "$OUTPUT_DIR/$script_name"
    
    # Replace all "python -m" with "CUDA_VISIBLE_DEVICES=X python -m"
    sed -i "s/python -m/CUDA_VISIBLE_DEVICES=$gpu_id python -m/g" "$OUTPUT_DIR/$script_name"
    
    # Update DATE variable to today's date and time
    TODAY_DATE=$(date '+%Y%m%d_%H%M%S')
    # Replace the DATE line more carefully to avoid multiple replacements
    sed -i "/^DATE=.*_optimized_ppo/c\DATE=\"${TODAY_DATE}_optimized_ppo\"" "$OUTPUT_DIR/$script_name"
    
    # Add header comment to identify the script
    sed -i "1i# Generated script for GPU $gpu_id, Combination $combo ($desc)" "$OUTPUT_DIR/$script_name"
    sed -i "2i# Original script: ${BASE_SCRIPT}.sh" "$OUTPUT_DIR/$script_name"
    sed -i "3i# Generated on: $(date)" "$OUTPUT_DIR/$script_name"
    sed -i "4i#" "$OUTPUT_DIR/$script_name"
done

# Create a master script to run all generated scripts
MASTER_SCRIPT="$OUTPUT_DIR/run_all_scripts.sh"
echo "#!/bin/bash" > "$MASTER_SCRIPT"
echo "" >> "$MASTER_SCRIPT"
echo "# Master script to run all generated GPU scripts" >> "$MASTER_SCRIPT"
echo "# Generated on: $(date)" >> "$MASTER_SCRIPT"
echo "# Number of scripts: ${#COMBINATIONS[@]}" >> "$MASTER_SCRIPT"
echo "# Available GPUs: $NUM_GPUS" >> "$MASTER_SCRIPT"
echo "" >> "$MASTER_SCRIPT"

# Group scripts by GPU
declare -A gpu_scripts
for i in "${!COMBINATIONS[@]}"; do
    combo=${COMBINATIONS[$i]}
    gpu_id=$((i % NUM_GPUS))
    desc=$(get_combination_desc $combo)
    script_name="${FILE_PREFIX}_gpu${gpu_id}_${desc}.sh"
    
    if [ -z "${gpu_scripts[$gpu_id]}" ]; then
        gpu_scripts[$gpu_id]=""
    fi
    gpu_scripts[$gpu_id]="${gpu_scripts[$gpu_id]} $script_name"
done

# Add GPU-specific sections to master script
echo "echo \"=== EAGLE RL Multi-GPU Training Started ===\"" >> "$MASTER_SCRIPT"
echo "echo \"Start time: \$(date)\"" >> "$MASTER_SCRIPT"
echo "echo \"Number of scripts: ${#COMBINATIONS[@]}\"" >> "$MASTER_SCRIPT"
echo "echo \"Available GPUs: $NUM_GPUS\"" >> "$MASTER_SCRIPT"
echo "echo \"\"" >> "$MASTER_SCRIPT"

# Create a function to run scripts on each GPU
echo "run_gpu_scripts() {" >> "$MASTER_SCRIPT"
echo "    local gpu_id=\$1" >> "$MASTER_SCRIPT"
echo "    shift" >> "$MASTER_SCRIPT"
echo "    local scripts=(\"\$@\")" >> "$MASTER_SCRIPT"
echo "    echo \"=== Starting scripts for GPU \$gpu_id ===\"" >> "$MASTER_SCRIPT"
echo "    for script in \"\${scripts[@]}\"; do" >> "$MASTER_SCRIPT"
echo "        echo \"Running \$script on GPU \$gpu_id...\"" >> "$MASTER_SCRIPT"
echo "        bash \"\$script\" &" >> "$MASTER_SCRIPT"
echo "    done" >> "$MASTER_SCRIPT"
echo "    wait" >> "$MASTER_SCRIPT"
echo "    echo \"=== Completed all scripts for GPU \$gpu_id ===\"" >> "$MASTER_SCRIPT"
echo "}" >> "$MASTER_SCRIPT"
echo "" >> "$MASTER_SCRIPT"

# Run scripts for each GPU
for gpu_id in "${!gpu_scripts[@]}"; do
    echo "run_gpu_scripts $gpu_id ${gpu_scripts[$gpu_id]} &" >> "$MASTER_SCRIPT"
done

echo "" >> "$MASTER_SCRIPT"
echo "# Wait for all GPUs to complete" >> "$MASTER_SCRIPT"
echo "wait" >> "$MASTER_SCRIPT"
echo "" >> "$MASTER_SCRIPT"
echo "echo \"\"" >> "$MASTER_SCRIPT"
echo "echo \"=== All GPU scripts completed ===\"" >> "$MASTER_SCRIPT"
echo "echo \"End time: \$(date)\"" >> "$MASTER_SCRIPT"

chmod +x "$MASTER_SCRIPT"

# Create a simple launcher script
LAUNCHER_SCRIPT="$OUTPUT_DIR/launch.sh"
echo "#!/bin/bash" > "$LAUNCHER_SCRIPT"
echo "" >> "$LAUNCHER_SCRIPT"
echo "# Simple launcher for EAGLE RL Multi-GPU Training" >> "$LAUNCHER_SCRIPT"
echo "# Usage: bash launch.sh" >> "$LAUNCHER_SCRIPT"
echo "" >> "$LAUNCHER_SCRIPT"
echo "echo \"=== EAGLE RL Multi-GPU Training Launcher ===\"" >> "$LAUNCHER_SCRIPT"
echo "echo \"Start time: \$(date)\"" >> "$LAUNCHER_SCRIPT"
echo "echo \"\"" >> "$LAUNCHER_SCRIPT"
echo "" >> "$LAUNCHER_SCRIPT"
echo "# Run the master script" >> "$LAUNCHER_SCRIPT"
echo "bash run_all_scripts.sh" >> "$LAUNCHER_SCRIPT"
echo "" >> "$LAUNCHER_SCRIPT"
echo "echo \"\"" >> "$LAUNCHER_SCRIPT"
echo "echo \"=== Training completed ===\"" >> "$LAUNCHER_SCRIPT"
echo "echo \"End time: \$(date)\"" >> "$LAUNCHER_SCRIPT"
chmod +x "$LAUNCHER_SCRIPT"

# Create a summary file
SUMMARY_FILE="$OUTPUT_DIR/script_summary.txt"
echo "EAGLE RL Multi-GPU Script Generation Summary" > "$SUMMARY_FILE"
echo "=============================================" >> "$SUMMARY_FILE"
echo "Generated on: $(date)" >> "$SUMMARY_FILE"
echo "Base script: ${BASE_SCRIPT}.sh" >> "$SUMMARY_FILE"
echo "Number of GPUs: $NUM_GPUS" >> "$SUMMARY_FILE"
echo "Number of scripts: ${#COMBINATIONS[@]}" >> "$SUMMARY_FILE"
echo "" >> "$SUMMARY_FILE"
echo "Selected combinations:" >> "$SUMMARY_FILE"
for i in "${!COMBINATIONS[@]}"; do
    combo=${COMBINATIONS[$i]}
    gpu_id=$((i % NUM_GPUS))
    desc=$(get_combination_desc $combo)
    script_name="${FILE_PREFIX}_gpu${gpu_id}_${desc}.sh"
    echo "  Script $((i+1)): $script_name (GPU $gpu_id, $desc)" >> "$SUMMARY_FILE"
done
echo "" >> "$SUMMARY_FILE"
echo "GPU distribution:" >> "$SUMMARY_FILE"
for gpu_id in "${!gpu_scripts[@]}"; do
    echo "  GPU $gpu_id: ${gpu_scripts[$gpu_id]}" >> "$SUMMARY_FILE"
done
echo "" >> "$SUMMARY_FILE"
echo "To run all scripts: bash $MASTER_SCRIPT" >> "$SUMMARY_FILE"
echo "To run individual scripts: bash <script_name>" >> "$SUMMARY_FILE"
echo "To run with launcher: bash launch.sh" >> "$SUMMARY_FILE"

echo ""
echo "=== Generation Complete ==="
echo "Generated ${#COMBINATIONS[@]} scripts in directory: $OUTPUT_DIR"
echo ""
echo "Scripts created:"
for i in "${!COMBINATIONS[@]}"; do
    combo=${COMBINATIONS[$i]}
    gpu_id=$((i % NUM_GPUS))
    desc=$(get_combination_desc $combo)
    script_name="${FILE_PREFIX}_gpu${gpu_id}_${desc}.sh"
    echo "  $script_name (GPU $gpu_id, $desc)"
done
echo ""
echo "Master script: $MASTER_SCRIPT"
echo "Launcher script: $LAUNCHER_SCRIPT"
echo "Summary file: $SUMMARY_FILE"
echo ""
echo "To run all scripts: bash $MASTER_SCRIPT"
echo "To run with launcher: bash launch.sh"
echo "To run individual scripts: cd $OUTPUT_DIR && bash <script_name>"

# Ask user if they want to run the scripts now
echo ""
read -p "Do you want to run all generated scripts now? (y/n, default: n): " RUN_NOW

if [[ "$RUN_NOW" =~ ^[Yy]$ ]]; then
    echo ""
    echo "=== Starting execution of all scripts ==="
    echo "Changing to output directory: $OUTPUT_DIR"
    cd "$OUTPUT_DIR"
    
    echo ""
    echo "Running master script: $MASTER_SCRIPT"
    echo "This will execute all ${#COMBINATIONS[@]} scripts across $NUM_GPUS GPUs"
    echo "Press Ctrl+C to stop execution"
    echo ""
    
    # Run the master script
    bash "$MASTER_SCRIPT"
    
    echo ""
    echo "=== Execution completed ==="
    echo "Check the log files in $OUTPUT_DIR for results"
else
    echo ""
    echo "Scripts generated but not executed."
    echo "To run later: cd $OUTPUT_DIR && bash launch.sh"
    echo "Or run individual scripts: cd $OUTPUT_DIR && bash <script_name>"
fi 
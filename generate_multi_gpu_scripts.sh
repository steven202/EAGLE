#!/bin/bash

# Multi-GPU Script Generator for EAGLE RL Training
# This script generates multiple GPU-specific training scripts based on user preferences
#
# Usage:
#   bash generate_multi_gpu_scripts.sh                    # Interactive mode
#   bash generate_multi_gpu_scripts.sh 1 2 3 4 5 6 7 8   # All combinations
#   bash generate_multi_gpu_scripts.sh 1 3 5 7            # Specific combinations
#   bash generate_multi_gpu_scripts.sh --datetime YYYYMMDD_HHMMSS 1 2 3 4  # With custom date/time
#   bash generate_multi_gpu_scripts.sh --overwrite 1 2 3 4  # Overwrite existing scripts

echo "=== EAGLE RL Multi-GPU Script Generator ==="
echo ""

# Show usage if no arguments provided
if [ $# -eq 0 ]; then
    echo "Usage examples:"
    echo "  bash $0                    # Interactive mode (with file handling options)"
    echo "  bash $0 1 2 3 4 5 6 7 8   # All combinations"
    echo "  bash $0 1 3 5 7            # Specific combinations"
    echo "  bash $0 --datetime 20241201_143052 1 2 3 4  # With custom date/time"
    echo "  bash $0 --overwrite 1 2 3 4  # Overwrite existing scripts (keep same date/time)"
    echo ""
    echo "Date/Time options (interactive mode):"
    echo "  1. Use current date/time (default)"
    echo "  2. Specify custom date/time"
    echo "  3. Use simple folder name (no date/time)"
    echo "  4. Overwrite existing scripts (keep same date/time)"
    echo "  5. Delete old files and start fresh"
    echo ""
    echo "File handling options (when existing scripts found):"
    echo "  1. Keep old files (default) - new scripts alongside existing ones"
    echo "  2. Delete old files - remove existing scripts and generate new ones"
    echo "  3. Overwrite old files - replace existing scripts with new ones"
    echo ""
fi

# Parse command line arguments for custom date/time and overwrite
CUSTOM_DATETIME_ARG=""
OVERWRITE_MODE=false
if [ $# -gt 0 ] && [ "$1" == "--datetime" ]; then
    if [ $# -lt 2 ]; then
        echo "Error: --datetime requires a date/time argument in format YYYYMMDD_HHMMSS"
        exit 1
    fi
    CUSTOM_DATETIME_ARG="$2"
    # Validate the format
    if [[ ! "$CUSTOM_DATETIME_ARG" =~ ^[0-9]{8}_[0-9]{6}$ ]]; then
        echo "Error: Invalid date/time format. Please use YYYYMMDD_HHMMSS format."
        exit 1
    fi
    
    # Extract components for validation
    YEAR=${CUSTOM_DATETIME_ARG:0:4}
    MONTH=${CUSTOM_DATETIME_ARG:4:2}
    DAY=${CUSTOM_DATETIME_ARG:6:2}
    HOUR=${CUSTOM_DATETIME_ARG:9:2}
    MINUTE=${CUSTOM_DATETIME_ARG:11:2}
    SECOND=${CUSTOM_DATETIME_ARG:13:2}
    
    # Basic validation
    if [ "$YEAR" -lt 2000 ] || [ "$YEAR" -gt 2100 ]; then
        echo "Error: Year must be between 2000 and 2100"
        exit 1
    fi
    if [ "$MONTH" -lt 1 ] || [ "$MONTH" -gt 12 ]; then
        echo "Error: Month must be between 01 and 12"
        exit 1
    fi
    if [ "$DAY" -lt 1 ] || [ "$DAY" -gt 31 ]; then
        echo "Error: Day must be between 01 and 31"
        exit 1
    fi
    if [ "$HOUR" -gt 23 ]; then
        echo "Error: Hour must be between 00 and 23"
        exit 1
    fi
    if [ "$MINUTE" -gt 59 ]; then
        echo "Error: Minute must be between 00 and 59"
        exit 1
    fi
    if [ "$SECOND" -gt 59 ]; then
        echo "Error: Second must be between 00 and 59"
        exit 1
    fi
    
    echo "Using command-line specified date/time: $CUSTOM_DATETIME_ARG"
    # Remove the --datetime and its argument from the command line
    shift 2
elif [ $# -gt 0 ] && [ "$1" == "--overwrite" ]; then
    OVERWRITE_MODE=true
    echo "Overwrite mode enabled - will use existing folder with same date/time"
    # Remove the --overwrite argument from the command line
    shift 1
fi

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

# Handle date/time selection
if [ "$OVERWRITE_MODE" == true ]; then
    # Overwrite mode - find the most recent generated folder
    echo ""
    echo "Looking for existing generated folders..."
    
    # Find all existing multi-gpu folders
    EXISTING_FOLDERS=($(ls -d ${FILE_PREFIX}__multi_gpu* 2>/dev/null | sort -r))
    
    if [ ${#EXISTING_FOLDERS[@]} -eq 0 ]; then
        echo "No existing generated folders found. Using current date/time instead."
        CUSTOM_DATE=$(date '+%Y%m%d_%H%M%S')
        OUTPUT_DIR="${BASE_SCRIPT:0:7}__multi_gpu_${CUSTOM_DATE}"
        echo "Using current date/time: $CUSTOM_DATE"
        echo "Folder name: $OUTPUT_DIR"
    else
        # Use the most recent folder
        MOST_RECENT_FOLDER="${EXISTING_FOLDERS[0]}"
        echo "Found existing folder: $MOST_RECENT_FOLDER"
        
        # Extract date/time from folder name
        if [[ "$MOST_RECENT_FOLDER" =~ ${FILE_PREFIX}__multi_gpu_(.+)$ ]]; then
            EXTRACTED_DATE="${BASH_REMATCH[1]}"
            # Validate if it looks like a date/time format
            if [[ "$EXTRACTED_DATE" =~ ^[0-9]{8}_[0-9]{6}$ ]]; then
                CUSTOM_DATE="$EXTRACTED_DATE"
                OUTPUT_DIR="$MOST_RECENT_FOLDER"
                echo "Using existing date/time: $CUSTOM_DATE"
                echo "Formatted: ${CUSTOM_DATE:0:4}-${CUSTOM_DATE:4:2}-${CUSTOM_DATE:6:2} ${CUSTOM_DATE:9:2}:${CUSTOM_DATE:11:2}:${CUSTOM_DATE:13:2}"
                echo "Folder name: $OUTPUT_DIR"
                echo "This will overwrite existing scripts in this folder."
            else
                # Simple folder without date/time
                CUSTOM_DATE=""
                OUTPUT_DIR="$MOST_RECENT_FOLDER"
                echo "Using existing simple folder: $OUTPUT_DIR"
                echo "This will overwrite existing scripts in this folder."
            fi
        else
            # Simple folder without date/time
            CUSTOM_DATE=""
            OUTPUT_DIR="$MOST_RECENT_FOLDER"
            echo "Using existing simple folder: $OUTPUT_DIR"
            echo "This will overwrite existing scripts in this folder."
        fi
    fi
elif [ -n "$CUSTOM_DATETIME_ARG" ]; then
    # Use command-line specified date/time
    CUSTOM_DATE="$CUSTOM_DATETIME_ARG"
    OUTPUT_DIR="${BASE_SCRIPT:0:7}__multi_gpu_${CUSTOM_DATETIME_ARG}"
    echo "Using command-line specified date/time: $CUSTOM_DATETIME_ARG"
    echo "Formatted: ${CUSTOM_DATETIME_ARG:0:4}-${CUSTOM_DATETIME_ARG:4:2}-${CUSTOM_DATETIME_ARG:6:2} ${CUSTOM_DATETIME_ARG:9:2}:${CUSTOM_DATETIME_ARG:11:2}:${CUSTOM_DATETIME_ARG:13:2}"
    echo "Folder name: $OUTPUT_DIR"
else
    # Ask about custom date/time
echo ""
echo "Date/Time options:"
echo "1. Use current date/time (default)"
echo "2. Specify custom date/time"
echo "3. Use simple folder name (no date/time)"
echo "4. Overwrite existing scripts (keep same date/time)"
echo "5. Delete old files and start fresh"
read -p "Choose option (1/2/3/4/5, default: 1): " DATE_OPTION

    if [[ "$DATE_OPTION" == "2" ]]; then
        echo ""
        echo "Enter custom date/time in format YYYYMMDD_HHMMSS"
        echo "Example: 20241201_143052 for December 1, 2024 at 14:30:52"
        read -p "Custom date/time: " CUSTOM_DATETIME
        
        # Validate the format
        if [[ ! "$CUSTOM_DATETIME" =~ ^[0-9]{8}_[0-9]{6}$ ]]; then
            echo "Error: Invalid format. Please use YYYYMMDD_HHMMSS format."
            exit 1
        fi
        
        # Extract components for validation
        YEAR=${CUSTOM_DATETIME:0:4}
        MONTH=${CUSTOM_DATETIME:4:2}
        DAY=${CUSTOM_DATETIME:6:2}
        HOUR=${CUSTOM_DATETIME:9:2}
        MINUTE=${CUSTOM_DATETIME:11:2}
        SECOND=${CUSTOM_DATETIME:13:2}
        
        # Basic validation
        if [ "$YEAR" -lt 2000 ] || [ "$YEAR" -gt 2100 ]; then
            echo "Error: Year must be between 2000 and 2100"
            exit 1
        fi
        if [ "$MONTH" -lt 1 ] || [ "$MONTH" -gt 12 ]; then
            echo "Error: Month must be between 01 and 12"
            exit 1
        fi
        if [ "$DAY" -lt 1 ] || [ "$DAY" -gt 31 ]; then
            echo "Error: Day must be between 01 and 31"
            exit 1
        fi
        if [ "$HOUR" -gt 23 ]; then
            echo "Error: Hour must be between 00 and 23"
            exit 1
        fi
        if [ "$MINUTE" -gt 59 ]; then
            echo "Error: Minute must be between 00 and 59"
            exit 1
        fi
        if [ "$SECOND" -gt 59 ]; then
            echo "Error: Second must be between 00 and 59"
            exit 1
        fi
        
        CUSTOM_DATE="$CUSTOM_DATETIME"
        OUTPUT_DIR="${BASE_SCRIPT:0:7}__multi_gpu_${CUSTOM_DATETIME}"
        echo "Using custom date/time: $CUSTOM_DATETIME"
        echo "Formatted: ${CUSTOM_DATETIME:0:4}-${CUSTOM_DATETIME:4:2}-${CUSTOM_DATETIME:6:2} ${CUSTOM_DATETIME:9:2}:${CUSTOM_DATETIME:11:2}:${CUSTOM_DATETIME:13:2}"
        echo "Folder name: $OUTPUT_DIR"
    elif [[ "$DATE_OPTION" == "3" ]]; then
        OUTPUT_DIR="${BASE_SCRIPT:0:7}__multi_gpu"
        CUSTOM_DATE=""
        echo "Using simple folder name: $OUTPUT_DIR"
    elif [[ "$DATE_OPTION" == "4" ]]; then
        # Overwrite existing scripts - find the most recent generated folder
        echo ""
        echo "Looking for existing generated folders..."
        
        # Find all existing multi-gpu folders
        EXISTING_FOLDERS=($(ls -d ${FILE_PREFIX}__multi_gpu* 2>/dev/null | sort -r))
        
        if [ ${#EXISTING_FOLDERS[@]} -eq 0 ]; then
            echo "No existing generated folders found. Using current date/time instead."
            CUSTOM_DATE=$(date '+%Y%m%d_%H%M%S')
            OUTPUT_DIR="${BASE_SCRIPT:0:7}__multi_gpu_${CUSTOM_DATE}"
            echo "Using current date/time: $CUSTOM_DATE"
            echo "Folder name: $OUTPUT_DIR"
        else
            # Use the most recent folder
            MOST_RECENT_FOLDER="${EXISTING_FOLDERS[0]}"
            echo "Found existing folder: $MOST_RECENT_FOLDER"
            
            # Extract date/time from folder name
            if [[ "$MOST_RECENT_FOLDER" =~ ${FILE_PREFIX}__multi_gpu_(.+)$ ]]; then
                EXTRACTED_DATE="${BASH_REMATCH[1]}"
                # Validate if it looks like a date/time format
                if [[ "$EXTRACTED_DATE" =~ ^[0-9]{8}_[0-9]{6}$ ]]; then
                    CUSTOM_DATE="$EXTRACTED_DATE"
                    OUTPUT_DIR="$MOST_RECENT_FOLDER"
                    echo "Using existing date/time: $CUSTOM_DATE"
                    echo "Formatted: ${CUSTOM_DATE:0:4}-${CUSTOM_DATE:4:2}-${CUSTOM_DATE:6:2} ${CUSTOM_DATE:9:2}:${CUSTOM_DATE:11:2}:${CUSTOM_DATE:13:2}"
                    echo "Folder name: $OUTPUT_DIR"
                    echo "This will overwrite existing scripts in this folder."
                else
                    # Simple folder without date/time
                    CUSTOM_DATE=""
                    OUTPUT_DIR="$MOST_RECENT_FOLDER"
                    echo "Using existing simple folder: $OUTPUT_DIR"
                    echo "This will overwrite existing scripts in this folder."
                fi
            else
                # Simple folder without date/time
                CUSTOM_DATE=""
                OUTPUT_DIR="$MOST_RECENT_FOLDER"
                echo "Using existing simple folder: $OUTPUT_DIR"
                echo "This will overwrite existing scripts in this folder."
            fi
            
            # Ask for confirmation
            read -p "Do you want to overwrite scripts in $OUTPUT_DIR? (y/n, default: y): " CONFIRM_OVERWRITE
            if [[ "$CONFIRM_OVERWRITE" =~ ^[Nn]$ ]]; then
                echo "Operation cancelled."
                exit 0
            fi
        fi
    elif [[ "$DATE_OPTION" == "5" ]]; then
        # Delete old files and start fresh
        echo ""
        echo "Delete mode: Looking for existing files to clean up..."
        
        # Delete existing scripts in current directory
        EXISTING_SCRIPTS_CURRENT=$(ls ${FILE_PREFIX}__gpu*_*.sh 2>/dev/null | wc -l)
        if [ $EXISTING_SCRIPTS_CURRENT -gt 0 ]; then
            echo "Found $EXISTING_SCRIPTS_CURRENT existing scripts in current directory."
            rm -f ${FILE_PREFIX}__gpu*_*.sh
            rm -f run_all_scripts.sh launch.sh script_summary.txt
            echo "Deleted existing scripts in current directory."
        fi
        
        # Delete existing multi-gpu folders
        EXISTING_FOLDERS=($(ls -d ${FILE_PREFIX}__multi_gpu* 2>/dev/null))
        if [ ${#EXISTING_FOLDERS[@]} -gt 0 ]; then
            echo "Found ${#EXISTING_FOLDERS[@]} existing multi-gpu folders:"
            for folder in "${EXISTING_FOLDERS[@]}"; do
                echo "  - $folder"
            done
            read -p "Do you want to delete all existing multi-gpu folders? (y/n, default: n): " DELETE_FOLDERS
            if [[ "$DELETE_FOLDERS" =~ ^[Yy]$ ]]; then
                for folder in "${EXISTING_FOLDERS[@]}"; do
                    echo "Deleting folder: $folder"
                    rm -rf "$folder"
                done
                echo "All existing multi-gpu folders deleted."
            else
                echo "Keeping existing folders."
            fi
        fi
        
        # Use current date/time for new generation
        CUSTOM_DATE=$(date '+%Y%m%d_%H%M%S')
        OUTPUT_DIR="${BASE_SCRIPT:0:7}__multi_gpu_${CUSTOM_DATE}"
        echo "Using current date/time: $CUSTOM_DATE"
        echo "Folder name: $OUTPUT_DIR"
        echo "Starting fresh with clean environment."
    else
        # Default: use current date/time
        CUSTOM_DATE=$(date '+%Y%m%d_%H%M%S')
        OUTPUT_DIR="${BASE_SCRIPT:0:7}__multi_gpu_${CUSTOM_DATE}"
        echo "Using current date/time: $CUSTOM_DATE"
        echo "Folder name: $OUTPUT_DIR"
    fi
fi

# Check if any existing generated scripts exist in current directory
EXISTING_SCRIPTS=$(ls ${FILE_PREFIX}__gpu*_*.sh 2>/dev/null | wc -l)
if [ $EXISTING_SCRIPTS -gt 0 ] && [ "$OVERWRITE_MODE" != true ]; then
    echo "Found $EXISTING_SCRIPTS existing generated scripts in current directory."
    echo ""
    echo "File handling options:"
    echo "1. Keep old files (default) - new scripts will be generated alongside existing ones"
    echo "2. Delete old files - remove existing scripts and generate new ones"
    echo "3. Overwrite old files - replace existing scripts with new ones"
    read -p "Choose option (1/2/3, default: 1): " FILE_HANDLING_OPTION
    
    if [[ "$FILE_HANDLING_OPTION" == "2" ]]; then
        echo "Deleting existing generated scripts..."
        rm -f ${FILE_PREFIX}__gpu*_*.sh
        rm -f run_all_scripts.sh launch.sh script_summary.txt
        echo "Existing scripts deleted."
    elif [[ "$FILE_HANDLING_OPTION" == "3" ]]; then
        echo "Overwriting existing scripts..."
        # We'll handle the actual overwriting later in the script generation
        OVERWRITE_EXISTING=true
    else
        echo "Keeping existing scripts. New scripts will be generated alongside them."
        OVERWRITE_EXISTING=false
    fi
elif [ $EXISTING_SCRIPTS -gt 0 ] && [ "$OVERWRITE_MODE" == true ]; then
    echo "Overwrite mode: Found $EXISTING_SCRIPTS existing generated scripts in current directory."
    echo "These will be overwritten with new scripts using the same date/time."
    OVERWRITE_EXISTING=true
else
    OVERWRITE_EXISTING=false
fi

# Check if output directory already exists
if [ -d "$OUTPUT_DIR" ]; then
    if [ "$OVERWRITE_MODE" == true ]; then
        echo "Overwrite mode: Output directory $OUTPUT_DIR already exists."
        echo "This directory will be used for regenerating scripts with the same date/time."
    else
        echo "Output directory $OUTPUT_DIR already exists."
        read -p "Do you want to delete it and regenerate? (y/n, default: n): " DELETE_DIR
        if [[ "$DELETE_DIR" =~ ^[Yy]$ ]]; then
            echo "Deleting existing directory: $OUTPUT_DIR"
            rm -rf "$OUTPUT_DIR"
        else
            echo "Keeping existing directory. Will create a new one with different timestamp."
            # Generate a new timestamp to avoid conflict
            if [ -n "$CUSTOM_DATE" ]; then
                # Add a suffix to the custom date to avoid conflict
                OUTPUT_DIR="${BASE_SCRIPT:0:7}__multi_gpu_${CUSTOM_DATE}_$(date +%H%M%S)"
            else
                OUTPUT_DIR="${BASE_SCRIPT:0:7}__multi_gpu_$(date +%H%M%S)"
            fi
            echo "Using new directory name: $OUTPUT_DIR"
        fi
    fi
fi

mkdir -p "$OUTPUT_DIR"
echo ""
echo "Generating scripts in directory: $OUTPUT_DIR"

# Clean up existing scripts in output directory if in overwrite mode
if [ "$OVERWRITE_MODE" == true ] || [ "$OVERWRITE_EXISTING" == true ]; then
    if [ "$OVERWRITE_MODE" == true ]; then
        echo "Overwrite mode: Cleaning up existing scripts in $OUTPUT_DIR..."
    else
        echo "Cleaning up existing scripts in $OUTPUT_DIR for overwrite..."
    fi
    rm -f "$OUTPUT_DIR"/${FILE_PREFIX}__gpu*_*.sh
    rm -f "$OUTPUT_DIR"/run_all_scripts.sh
    rm -f "$OUTPUT_DIR"/launch.sh
    rm -f "$OUTPUT_DIR"/script_summary.txt
    echo "Existing scripts cleaned up."
fi

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
    script_name="${FILE_PREFIX}__gpu${gpu_id}_${desc}_${combo}.sh"
    
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
    
    # Update DATE variable to use the selected date/time
    if [ -n "$CUSTOM_DATE" ]; then
        # Use custom date/time
        sed -i "/^DATE=.*_optimized_ppo/c\DATE=\"${CUSTOM_DATE}_optimized_ppo\"" "$OUTPUT_DIR/$script_name"
    else
        # Use current date/time for simple folder option
        TODAY_DATE=$(date '+%Y%m%d_%H%M%S')
        sed -i "/^DATE=.*_optimized_ppo/c\DATE=\"${TODAY_DATE}_optimized_ppo\"" "$OUTPUT_DIR/$script_name"
    fi
    
    # Add header comment to identify the script
    sed -i "1i# Generated script for GPU $gpu_id, Combination $combo ($desc)" "$OUTPUT_DIR/$script_name"
    sed -i "2i# Original script: ${BASE_SCRIPT}.sh" "$OUTPUT_DIR/$script_name"
    if [ -n "$CUSTOM_DATE" ]; then
        sed -i "3i# Generated on: ${CUSTOM_DATE:0:4}-${CUSTOM_DATE:4:2}-${CUSTOM_DATE:6:2} ${CUSTOM_DATE:9:2}:${CUSTOM_DATE:11:2}:${CUSTOM_DATE:13:2}" "$OUTPUT_DIR/$script_name"
    else
        sed -i "3i# Generated on: $(date)" "$OUTPUT_DIR/$script_name"
    fi
    sed -i "4i#" "$OUTPUT_DIR/$script_name"
done

# Create a master script to run all generated scripts
MASTER_SCRIPT="$OUTPUT_DIR/run_all_scripts.sh"
echo "#!/bin/bash" > "$MASTER_SCRIPT"
echo "" >> "$MASTER_SCRIPT"
echo "# Master script to run all generated GPU scripts" >> "$MASTER_SCRIPT"
if [ -n "$CUSTOM_DATE" ]; then
    echo "# Generated on: ${CUSTOM_DATE:0:4}-${CUSTOM_DATE:4:2}-${CUSTOM_DATE:6:2} ${CUSTOM_DATE:9:2}:${CUSTOM_DATE:11:2}:${CUSTOM_DATE:13:2}" >> "$MASTER_SCRIPT"
else
    echo "# Generated on: $(date)" >> "$MASTER_SCRIPT"
fi
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
if [ -n "$CUSTOM_DATE" ]; then
    echo "# Generated on: ${CUSTOM_DATE:0:4}-${CUSTOM_DATE:4:2}-${CUSTOM_DATE:6:2} ${CUSTOM_DATE:9:2}:${CUSTOM_DATE:11:2}:${CUSTOM_DATE:13:2}" >> "$LAUNCHER_SCRIPT"
else
    echo "# Generated on: $(date)" >> "$LAUNCHER_SCRIPT"
fi
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
if [ -n "$CUSTOM_DATE" ]; then
    echo "Generated on: ${CUSTOM_DATE:0:4}-${CUSTOM_DATE:4:2}-${CUSTOM_DATE:6:2} ${CUSTOM_DATE:9:2}:${CUSTOM_DATE:11:2}:${CUSTOM_DATE:13:2}" >> "$SUMMARY_FILE"
else
    echo "Generated on: $(date)" >> "$SUMMARY_FILE"
fi
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
if [ "$OVERWRITE_MODE" == true ]; then
    echo "Regenerated ${#COMBINATIONS[@]} scripts in directory: $OUTPUT_DIR (overwrite mode)"
    echo "Date/time preserved for training continuity: $CUSTOM_DATE"
elif [ "$OVERWRITE_EXISTING" == true ]; then
    echo "Regenerated ${#COMBINATIONS[@]} scripts in directory: $OUTPUT_DIR (overwrite existing)"
    if [ -n "$CUSTOM_DATE" ]; then
        echo "Date/time preserved for training continuity: $CUSTOM_DATE"
    fi
elif [[ "$DATE_OPTION" == "5" ]]; then
    echo "Generated ${#COMBINATIONS[@]} scripts in directory: $OUTPUT_DIR (fresh start)"
    echo "Old files cleaned up, starting with clean environment."
else
    echo "Generated ${#COMBINATIONS[@]} scripts in directory: $OUTPUT_DIR"
fi
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
    
    # Add note about file handling if scripts were kept
    if [ "$OVERWRITE_EXISTING" == false ] && [ $EXISTING_SCRIPTS -gt 0 ]; then
        echo ""
        echo "Note: Existing scripts were kept. New scripts are generated alongside them."
        echo "You may want to organize or clean up the directory manually."
    fi
fi

# Add helpful note about overwrite functionality
if [ "$OVERWRITE_MODE" == true ] || [ "$OVERWRITE_EXISTING" == true ]; then
    echo ""
    echo "=== Overwrite Information ==="
    if [ -n "$CUSTOM_DATE" ]; then
        echo "Scripts were regenerated with the same date/time: $CUSTOM_DATE"
        echo "This allows you to resume training from where you left off."
        echo "The DATE variable in all scripts remains consistent."
        echo "Log directories will use the same timestamp for continuity."
    else
        echo "Scripts were regenerated with overwrite mode."
        echo "Existing scripts have been replaced with new ones."
    fi
fi 
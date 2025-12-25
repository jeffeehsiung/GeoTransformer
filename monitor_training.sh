#!/usr/bin/env bash
# monitor_training.sh - Monitor active training process
# Usage: ./monitor_training.sh [interval_seconds]

set -Eeuo pipefail

INTERVAL=${1:-10}  # Default 10 seconds

# ANSI colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo "=========================================="
echo "GeoTransformer Training Monitor"
echo "=========================================="

# Check if training is running
if [ ! -f process_debugger.pid ]; then
    echo -e "${RED}✗ No training process found (process_debugger.pid missing)${NC}"
    echo "Start training with: ./launch_training_monitored.sh"
    exit 1
fi


TRAIN_PID=$(cat process_debugger.pid)
if ! kill -0 "$TRAIN_PID" 2>/dev/null; then
    echo -e "${RED}✗ Training process $TRAIN_PID is not running${NC}"
    echo "Check logs for errors:"
    ls -lht train_run_*.log process_debugger_*.out | head -5
    exit 1
fi

echo -e "${GREEN}✓ Training is running (PID: $TRAIN_PID)${NC}"
echo ""

if [ -f "/proc/$TRAIN_PID/environ" ]; then
    CUDA_DEVICES=$(tr '\0' '\n' < "/proc/$TRAIN_PID/environ" | grep "^CUDA_VISIBLE_DEVICES=" | cut -d= -f2)
    if [ -n "$CUDA_DEVICES" ]; then
        echo -e "GPU Assignment: ${BLUE}$CUDA_DEVICES${NC}"
    fi
fi
echo ""

# Function to get GPU memory
get_gpu_memory() {
    if command -v nvidia-smi >/dev/null 2>&1; then
        nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits
    fi
}

# Function to get process info
get_process_info() {
    local pid=$1
    if [ -f "/proc/$pid/status" ]; then
        local rss_kb=$(awk '/^VmRSS:/ {print $2}' "/proc/$pid/status")
        local rss_gb=$(awk "BEGIN {printf \"%.2f\", $rss_kb/1024/1024}")
        local peak_kb=$(awk '/^VmHWM:/ {print $2}' "/proc/$pid/status")
        local peak_gb=$(awk "BEGIN {printf \"%.2f\", $peak_kb/1024/1024}")
        echo "$rss_gb,$peak_gb"
    else
        echo "N/A,N/A"
    fi
}

# Function to parse latest log for epoch info
get_training_progress() {
    local latest_log=$(ls -t train_run_*.log 2>/dev/null | head -1)
    if [ -z "$latest_log" ]; then
        echo "No log found"
        return
    fi

    # Try to find epoch and loss information
    local epoch_info=$(grep -E "Epoch|Loss|loss|Iteration" "$latest_log" | tail -10 | sed 's/^/  /')
    if [ -n "$epoch_info" ]; then
        echo "$epoch_info"
    else
        echo "  (No progress info yet)"
    fi
}

# Main monitoring loop
echo "Monitoring interval: ${INTERVAL}s (Ctrl+C to stop)"
echo "=========================================="
echo ""

while true; do
    clear
    echo "=========================================="
    echo -e "${BLUE}Training Monitor - $(date '+%Y-%m-%d %H:%M:%S')${NC}"
    echo "=========================================="
    echo ""

    # Check if process is still alive
    if ! kill -0 "$TRAIN_PID" 2>/dev/null; then
        echo -e "${RED}✗ Training process $TRAIN_PID has stopped${NC}"
        echo ""
        echo "Recent log files:"
	sed -i '/Loading directly from disk/d' process_debugger_*.out
	sed -i '/Loading directly from disk/d' *.log
	sed -i '/Loading directly from disk/d' output/transfer_learning_phase_0/logs/*.log
        ls -lht train_run_*.log process_debugger_*.out | head -2
        break
    fi

    echo -e "${GREEN}✓ Training PID: $TRAIN_PID${NC}"
    echo ""

    # Process memory
    echo -e "${YELLOW}[Process Memory]${NC}"
    IFS=',' read -r rss_gb peak_gb <<< $(get_process_info "$TRAIN_PID")
    echo "  Current RSS: ${rss_gb} GB"
    echo "  Peak RSS:    ${peak_gb} GB"
    echo ""

    # GPU memory
    echo -e "${YELLOW}[GPU Memory]${NC}"
    if command -v nvidia-smi >/dev/null 2>&1; then
        gpu_info=$(get_gpu_memory)
        echo "$gpu_info" | while IFS=',' read -r idx name used total util; do
            used_gb=$(awk "BEGIN {printf \"%.2f\", $used/1024}")
            total_gb=$(awk "BEGIN {printf \"%.2f\", $total/1024}")
            percent=$(awk "BEGIN {printf \"%.1f\", ($used/$total)*100}")
            echo "  GPU $idx: ${used_gb}/${total_gb} GB (${percent}%, Util: ${util}%)"
        done

        # Show process GPU usage
        nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader,nounits 2>/dev/null | \
        grep "^$TRAIN_PID," | while IFS=',' read -r pid pname mem; do
            mem_gb=$(awk "BEGIN {printf \"%.2f\", $mem/1024}")
            echo "  Training Process: ${mem_gb} GB"
        done
    else
        echo "  nvidia-smi not available"
    fi
    echo ""

    # Training progress
    echo -e "${YELLOW}[Training Progress]${NC}"
    sed -i '/Loading directly from disk/d' output/transfer_learning_phase_0/logs/*.log
    sed -i '/Loading directly from disk/d' *.log
    sed -i '/Loading directly from disk/d' *.out
    $(get_training_progress)
    echo ""

    # System load
    echo -e "${YELLOW}[System Load]${NC}"
    uptime | sed 's/^/  /'
    echo ""

    # Cache info
    echo -e "${YELLOW}[Cache Status]${NC}"
    cache_dir="output/*/cache"
    if compgen -G "$cache_dir" > /dev/null; then
        for dir in $cache_dir; do
            if [ -d "$dir" ]; then
                train_cache=$(find "$dir/train_per_sample" -name "*.pkl" 2>/dev/null | wc -l)
                val_cache=$(find "$dir/val_per_sample" -name "*.pkl" 2>/dev/null | wc -l)
                cache_size=$(du -sh "$dir" 2>/dev/null | cut -f1)
                echo "  Train cache: $train_cache files"
                echo "  Val cache:   $val_cache files"
                echo "  Cache size:  $cache_size"
                break
            fi
        done
    else
        echo "  No cache directory found"
    fi
    echo ""

    echo "=========================================="
    echo "Refreshing in ${INTERVAL}s... (Ctrl+C to stop)"
    sleep "$INTERVAL"
done

echo ""
echo "Monitoring memory usage."

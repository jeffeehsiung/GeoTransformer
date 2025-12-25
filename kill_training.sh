#!/usr/bin/env bash
# kill_training.sh - Safely stop training process and debugger
# Usage: ./kill_training.sh [--force]

set -Eeo pipefail

FORCE=false
if [[ "${1:-}" == "--force" ]]; then
    FORCE=true
fi

# ANSI colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo "=========================================="
echo "Stop GeoTransformer Training"
echo "=========================================="

# Function to kill process by PID file
kill_by_pidfile() {
    local pidfile=$1
    local name=$2
    local signal=${3:-TERM}

    if [ ! -f "$pidfile" ]; then
        echo -e "${YELLOW}⚠ $name PID file not found: $pidfile${NC}"
        return 1
    fi

    local pid=$(cat "$pidfile")

    if ! kill -0 "$pid" 2>/dev/null; then
        echo -e "${YELLOW}⚠ $name process (PID: $pid) is not running${NC}"
        rm -f "$pidfile"
        return 1
    fi

    echo -e "${GREEN}Stopping $name (PID: $pid) with signal $signal...${NC}"

    if [ "$signal" == "KILL" ]; then
        kill -9 "$pid" 2>/dev/null || true
    else
        kill -"$signal" "$pid" 2>/dev/null || true
    fi

    # Wait for process to die
    local timeout=10
    local count=0
    while kill -0 "$pid" 2>/dev/null && [ $count -lt $timeout ]; do
        sleep 1
        count=$((count + 1))
        echo -n "."
    done
    echo ""

    if kill -0 "$pid" 2>/dev/null; then
        echo -e "${RED}✗ Failed to stop $name process${NC}"
        if [ "$FORCE" = false ]; then
            echo -e "${YELLOW}  Use --force to send SIGKILL${NC}"
        fi
        return 1
    else
        echo -e "${GREEN}✓ $name stopped${NC}"
        rm -f "$pidfile"
        return 0
    fi
}

# Find and kill training process
if [ -f process_debugger.pid ]; then
    if [ "$FORCE" = true ]; then
        kill_by_pidfile process_debugger.pid "Training process" KILL
    else
        kill_by_pidfile process_debugger.pid "Training process" TERM
    fi
else
    echo -e "${YELLOW}⚠ Training process PID file not found (process_debugger.pid)${NC}"
fi

# Find and kill launcher/debugger
if [ -f process_debugger.launcher.pid ]; then
    if [ "$FORCE" = true ]; then
        kill_by_pidfile process_debugger.launcher.pid "Debugger launcher" KILL
    else
        kill_by_pidfile process_debugger.launcher.pid "Debugger launcher" TERM
    fi
else
    echo -e "${YELLOW}⚠ Debugger launcher PID file not found (process_debugger.launcher.pid)${NC}"
fi

# Clean up any stray Python training processes
echo ""
echo "Checking for stray training processes..."
STRAY_PIDS=$(pgrep -f "train_transfer_learning.py" || true)

if [ -n "$STRAY_PIDS" ]; then
    echo -e "${YELLOW}Found stray training process(es): $STRAY_PIDS${NC}"

    if [ "$FORCE" = true ]; then
        echo "Force killing stray processes..."
        echo "$STRAY_PIDS" | xargs kill -9 2>/dev/null || true
        echo -e "${GREEN}✓ Stray processes killed${NC}"
    else
        echo -e "${YELLOW}Run with --force to kill these processes${NC}"
    fi
else
    echo -e "${GREEN}✓ No stray training processes found${NC}"
fi

# Summary
echo ""
echo "=========================================="
echo "Summary"
echo "=========================================="

if [ -f process_debugger.pid ] || [ -f process_debugger.launcher.pid ]; then
    echo -e "${RED}✗ Some processes may still be running${NC}"
    echo "Run: ./kill_training.sh --force"
    exit 1
else
    echo -e "${GREEN}✓ All training processes stopped${NC}"
    echo ""
    echo "Training logs preserved:"
    echo "  train_run_*.log"
    echo "  process_debugger_*.out"
    echo ""
    echo "To restart training:"
    echo "  ./launch_training_monitored.sh"
fi

echo "=========================================="

#!/usr/bin/env bash
# launch_training_monitored.sh
#
# Usage (interactive, background with nohup):
#   ./launch_training_monitored.sh           # auto-pick GPU
#   ./launch_training_monitored.sh 1         # force GPU 1
#
# Usage (systemd or foreground):
#   ./launch_training_monitored.sh --foreground        # auto-pick GPU
#   ./launch_training_monitored.sh --foreground 1      # force GPU 1

set -Eeuo pipefail

# -------- GPU auto-selection --------
choose_gpu() {
  # Fallback: if nvidia-smi is not available, just use GPU 0
  if ! command -v nvidia-smi >/dev/null 2>&1; then
    echo "0"
    return
  fi

  # Query GPU index, utilization, and memory used (MiB, no units)
  # Format per line: "0, 0, 12"
  local best_idx="-1"
  local best_mem=999999

  while IFS=',' read -r idx util mem; do
    idx="${idx//[[:space:]]/}"
    util="${util//[[:space:]]/}"
    mem="${mem//[[:space:]]/}"

    # Treat idle+tiny-mem GPUs as free: pick immediately
    if [ "${util:-0}" = "0" ] && [ "${mem:-0}" -lt 500 ]; then
      echo "$idx"
      return
    fi

    # Track GPU with lowest memory usage
    if [ "${mem:-0}" -lt "$best_mem" ]; then
      best_mem="$mem"
      best_idx="$idx"
    fi
  done < <(nvidia-smi --query-gpu=index,utilization.gpu,memory.used \
                      --format=csv,noheader,nounits 2>/dev/null)

  # If nothing parsed, default to 0
  if [ "$best_idx" = "-1" ]; then
    best_idx="0"
  fi
  echo "$best_idx"
}

# Timestamp helper
ts(){ date "+%Y-%m-%d %H:%M:%S"; }

# -------- Argument parsing --------
mode="background"   # default mode when you run it by hand
GPU_ARG=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --foreground)
      mode="foreground"
      shift
      ;;
    --background)
      mode="background"
      shift
      ;;
    [0-9])
      GPU_ARG="$1"
      shift
      ;;
    *)
      echo "Usage: $0 [--foreground|--background] [GPU_ID]" >&2
      exit 2
      ;;
  esac
done

GPU_ID="${GPU_ARG:-$(choose_gpu)}"
export CUDA_VISIBLE_DEVICES="$GPU_ID"

LOG_TAG=$(date +'%Y%m%d_%H%M%S')

echo "=========================================="
echo "GeoTransformer Training Launch"
echo "Timestamp: $LOG_TAG"
echo "Mode:       $mode"
echo "Selected GPU: $GPU_ID (CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES)"
echo "=========================================="

# -------- Env + training params --------
export PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:64,expandable_segments:True"
export CUDA_LAUNCH_BLOCKING=0

EPOCHS=80
FREEZE_EPOCHS="1 24 48 64"
DATASET_ID="HST-YZ/CAD6608272632/2025-10-14-rvbust_RE_30_synthetic"

echo "Memory-optimized parameters:"
echo "  Epochs:     $EPOCHS"
echo "  Freeze:     $FREEZE_EPOCHS"
echo "  Dataset ID: $DATASET_ID"
echo "  GPU:        $GPU_ID"
echo "  Note: num_points and voxel_size set in config.py"
echo ""

BAZEL_TARGET="//roboeye/iris/src/geoTransformer/GeoTransformer:train_transfer_learning"

# Build command as array for clean argument passing
TRAIN_ARGS=(
  "bazel" "run" "$BAZEL_TARGET" "--"
  "--dataset_id" "$DATASET_ID"
  "--enable_progressive_unfreezing"
  "--enable_so3_curriculum"
  "--optim_max_epoch" "$EPOCHS"
  "--freeze_epochs" "$FREEZE_EPOCHS"
  "--output_dir" "${DATASET_ID}"
  "--resume"
  "--cache_bool"
)

echo "Training command:"
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES ${TRAIN_ARGS[*]}"
echo ""

RUN_LOG="train_run_${LOG_TAG}.log"
DBG_LOG="process_debugger_${LOG_TAG}.out"

# -------- Launch modes --------
if [[ "$mode" == "foreground" ]]; then
  echo "[$(ts)] [INFO] Running in foreground (systemd-friendly)."
  rm -f process_debugger.pid process_debugger.launcher.pid || true

  exec ./process_debugger.sh \
    -i 5 \
    -g \
    -j 10 \
    -o "$RUN_LOG" \
    -- "${TRAIN_ARGS[@]}" \
    > "$DBG_LOG" 2>&1

else
  echo "[$(ts)] [INFO] Launching in background with nohup..."
  nohup ./process_debugger.sh \
    -i 5 \
    -g \
    -j 10 \
    -o "$RUN_LOG" \
    -- "${TRAIN_ARGS[@]}" \
    > "$DBG_LOG" 2>&1 &

  LAUNCHER_PID=$!
  echo "$LAUNCHER_PID" > process_debugger.launcher.pid
  disown

  echo "✓ Training launched in background on GPU $GPU_ID"
  echo "  Launcher PID: $LAUNCHER_PID (saved to process_debugger.launcher.pid)"
  echo "  Training PID will be in: process_debugger.pid"
  echo ""
  echo "Log files:"
  echo "  Training log:     $RUN_LOG"
  echo "  Debugger output:  $DBG_LOG"
  echo ""
  echo "Monitor with:"
  echo "  ./monitor_training.sh"
  echo "  tail -f $RUN_LOG"
  echo "  tail -f $DBG_LOG"
  echo ""
  echo "Stop training:"
  echo "  kill \$(cat process_debugger.pid)          # Stop training process"
  echo "  kill \$(cat process_debugger.launcher.pid) # Stop debugger"
  echo ""

  # Quick health check
  sleep 3
  if [ -f process_debugger.pid ]; then
      TRAIN_PID=$(cat process_debugger.pid)
      if kill -0 "$TRAIN_PID" 2>/dev/null; then
          echo "✓ Training is running (PID: $TRAIN_PID)"
      else
          echo "✗ Training process died immediately. Check $DBG_LOG"
          exit 1
      fi
  else
      echo "⚠ process_debugger.pid not created yet, training may still be starting..."
  fi

  echo ""
  echo "=========================================="
  echo "Launch complete. Training running in background."
  echo "=========================================="
fi

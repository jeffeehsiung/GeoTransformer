#!/usr/bin/env bash
set -euo pipefail

# --- CONFIG -------------------------------------------------------------
# Adjust these if your layout changes
PROJECT_ROOT="$HOME/repos/roboeye/iris/src/geoTransformer/GeoTransformer"
PHASE_DIR="$PROJECT_ROOT/output/transfer_learning_phase_0"
LOG_DIR="$PHASE_DIR/logs"

TRAIN_SCRIPT_NAME="train_transfer_learning.py"
NOISY_LINE="Loading directly from disk as overwrite is set to False, data may be out of date"
# ------------------------------------------------------------------------

cd "$PROJECT_ROOT"

echo "=== process_logs_debugger.sh ==="
echo "Project root: $PROJECT_ROOT"
echo "Phase dir   : $PHASE_DIR"
echo "Log dir     : $LOG_DIR"
echo

# --- PID / PROCESS INFO -------------------------------------------------
# Optional PID argument, otherwise auto-detect first matching train script
PID="${1:-}"

if [[ -z "$PID" ]]; then
  PID="$(pgrep -f "$TRAIN_SCRIPT_NAME" | head -n 1 || true)"
fi

if [[ -n "${PID:-}" ]]; then
  echo "[INFO] Monitoring PID: $PID"
  echo
  echo "ps -fp $PID"
  ps -fp "$PID" || true
  echo
  echo "pstree -p $PID"
  pstree -p "$PID" || true
  echo
else
  echo "[WARN] No PID found for \"$TRAIN_SCRIPT_NAME\" (maybe it finished or hasn't started yet)."
  echo
fi

# --- CLEAN NOISY LINES FROM LOGS ---------------------------------------
echo "[INFO] Cleaning noisy lines from logs (if any)..."

# process_debugger_*.out in project root
sed -i "/$NOISY_LINE/d" process_debugger_*.out 2>/dev/null || true

# train_run_*.log in project root (if you have these)
sed -i "/$NOISY_LINE/d" train_run_*.log 2>/dev/null || true

# train-*.log under phase logs
sed -i "/$NOISY_LINE/d" "$LOG_DIR"/train-*.log 2>/dev/null || true

echo "[INFO] Done cleaning."
echo


# --- SHOW PROCESS -------------------------------------------------------
echo "[INFO] Grapping Python Processes"
echo "================[PYTHON PROCESSES]==================="
pgrep -a -f "$TRAIN_SCRIPT_NAME"
echo "================[PYTHON PROCESSES]==================="

# --- FIND LATEST LOGS ---------------------------------------------------
echo "[INFO] Discovering latest log files..."

# Latest process_debugger_*.out
LATEST_DEBUG_OUT="$(ls -t process_debugger_*.out 2>/dev/null | head -n 1 || true)"

# Latest train-*.log in phase logs
LATEST_TRAIN_LOG="$(ls -t "$LOG_DIR"/train-*.log 2>/dev/null | head -n 1 || true)"

# Optional: latest train_run_*.log in project root
LATEST_RUN_LOG="$(ls -t train_run_*.log 2>/dev/null | head -n 1 || true)"

[[ -n "$LATEST_DEBUG_OUT" ]] && echo "  Latest process_debugger_*.out: $LATEST_DEBUG_OUT"
[[ -n "$LATEST_TRAIN_LOG" ]] && echo "  Latest train-*.log           : $LATEST_TRAIN_LOG"
[[ -n "$LATEST_RUN_LOG"   ]] && echo "  Latest train_run_*.log       : $LATEST_RUN_LOG"

if [[ -z "$LATEST_DEBUG_OUT" && -z "$LATEST_TRAIN_LOG" && -z "$LATEST_RUN_LOG" ]]; then
  echo
  echo "[ERROR] No matching .out or .log files found. Nothing to tail."
  exit 1
fi

echo
echo "[INFO] Tailing latest logs (Ctrl+C to exit)..."
echo

# --- TAIL THE LATEST FILES ---------------------------------------------
# Build the list of files we actually have
FILES_TO_TAIL=()

[[ -n "$LATEST_DEBUG_OUT" ]] && FILES_TO_TAIL+=("$LATEST_DEBUG_OUT")
[[ -n "$LATEST_TRAIN_LOG" ]] && FILES_TO_TAIL+=("$LATEST_TRAIN_LOG")
# Uncomment this if you also want the train_run_*.log in the same tail:
[[ -n "$LATEST_RUN_LOG"   ]] && FILES_TO_TAIL+=("$LATEST_RUN_LOG")

# Tail last 100 lines and keep following (-F handles rotation)
tail -n 100 -F "${FILES_TO_TAIL[@]}"

#!/usr/bin/env bash
# process_debugger.sh — run a command, monitor it, and on exit print why it died
# Usage:
#   ./process_debugger.sh [-i 5] [-g] [-j 10] [-o run.log] [-c] -- <your command...>
# Options:
#   -i <secs>   Sample interval (default 5)
#   -g          Sample GPU stats with nvidia-smi if available
#   -j <mins>   Tail recent kernel/journal logs for this many minutes (default 5)
#   -o <file>   Tee this script's own stdout/stderr to file
#   -c          Enable core dumps (ulimit -c unlimited)
#   -h          Show this header

set -Eeuo pipefail

interval=5
gpu=0
tail_mins=5
logfile=""
enable_core=0

while getopts ":i:j:o:gch" opt; do
  case "$opt" in
    i) interval="${OPTARG}";;
    j) tail_mins="${OPTARG}";;
    o) logfile="${OPTARG}";;
    g) gpu=1;;
    c) enable_core=1;;
    h) sed -n '1,80p' "$0"; exit 0;;
    \?) echo "Unknown option: -$OPTARG" >&2; exit 2;;
    :) echo "Option -$OPTARG requires an argument." >&2; exit 2;;
  esac
done

# Position to the first non-option
shift $((OPTIND - 1))
# If user included a literal "--", skip it; getopts may have consumed it already.
if [ "${1-}" = "--" ]; then shift; fi
# Require at least one command token
if [ "$#" -eq 0 ]; then
  echo "Usage: $0 [opts] -- <cmd...>"
  exit 2
fi

# Capture the command as an array, and build a safely-quoted string for bash -lc
cmd=("$@")
cmd_str="$(printf "%q " "${cmd[@]}")"

ts(){ date "+%Y-%m-%d %H:%M:%S"; }

# Tee this script's own output if requested
if [ -n "$logfile" ]; then
  exec > >(tee -a "$logfile") 2>&1
fi

echo "[$(ts)] [dbg] starting: ${cmd[*]}"
echo "[$(ts)] [dbg] interval=${interval}s gpu=${gpu} tail_mins=${tail_mins} core=${enable_core}"
echo "[$(ts)] [dbg] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
echo "[$(ts)] [dbg] WORLD_SIZE=${WORLD_SIZE:-unset} RANK=${RANK:-unset} LOCAL_RANK=${LOCAL_RANK:-unset}"

# Core dumps optional
if [ "$enable_core" -eq 1 ]; then
  (ulimit -c unlimited && echo "[$(ts)] [dbg] core dumps enabled") || echo "[$(ts)] [warn] couldn't enable core dumps"
fi

# Where we'll stash snapshots
snapdir="$(mktemp -d -t proc_snap_XXXX)"
echo "[$(ts)] [dbg] snapdir=$snapdir"

time_out="$snapdir/time.txt"

# Launch target command under /usr/bin/time in a subshell, detached from this monitor
 /usr/bin/time -v -o "$time_out" bash -lc "$cmd_str" &
pid=$!
echo "$pid" > process_debugger.pid
echo "[$(ts)] [dbg] pid=$pid (saved to process_debugger.pid)"
echo "[$(ts)] [dbg] child PIDs: $(pgrep -P $pid | xargs)"  # List child PIDs if any
# Trap external kills on the debugger itself
trap 'echo "[$(ts)] [WARN] process_debugger.sh received a termination signal.
If you do NOT see a \"---- process exit ----\" block above,
it means the monitor was killed externally (logout/systemd/kill).
Child pid=$pid"; exit 1' TERM INT QUIT HUP

# Snapshot function (best-effort)
snap_proc(){
  local p="$1" d="$2"
  [ -d "/proc/$p" ] || return 0
  readlink -f "/proc/$p/exe"      > "$d/exe.txt" 2>/dev/null || true
  cat "/proc/$p/cmdline"          > "$d/cmdline.txt" 2>/dev/null || true
  tr '\0' ' ' < "$d/cmdline.txt" | sed 's/ $//' > "$d/cmdline_readable.txt" 2>/dev/null || true
  cat "/proc/$p/status"           > "$d/status.txt" 2>/dev/null || true
  cat "/proc/$p/limits"           > "$d/limits.txt" 2>/dev/null || true
  cat "/proc/$p/oom_score"        > "$d/oom_score.txt" 2>/dev/null || true
  cat "/proc/$p/oom_score_adj"    > "$d/oom_score_adj.txt" 2>/dev/null || true
  cat "/proc/$p/smaps_rollup"     > "$d/smaps_rollup.txt" 2>/dev/null || true
  cat "/proc/$p/cgroup"           > "$d/cgroup.txt" 2>/dev/null || true
}

# Dump cgroup (v1/v2) paths based on saved cgroup.txt
cgroup_dump_by_path(){
  local cg_line cg_path base base2
  if [ -f "$1/cgroup.txt" ]; then
    cg_line="$(grep -E '^[0-9]+:memory:' "$1/cgroup.txt" || true)"
    if [ -n "$cg_line" ]; then
      cg_path="${cg_line##*:}"
      base="/sys/fs/cgroup/memory${cg_path}"
      if [ -d "$base" ]; then
        echo "---- cgroup (v1) $base ----"
        for f in memory.limit_in_bytes memory.usage_in_bytes memory.max_usage_in_bytes memory.oom_control memory.failcnt memory.stat; do
          [ -f "$base/$f" ] && { echo "[$f]"; sed -n '1,200p' "$base/$f"; }
        done
        return
      fi
    fi
    cg_path="$(awk -F: '$2==""{print $3}' "$1/cgroup.txt" || true)"
    base2="/sys/fs/cgroup${cg_path}"
    if [ -d "$base2" ]; then
      echo "---- cgroup (v2) $base2 ----"
      for f in memory.max memory.current memory.high memory.events memory.swap.max memory.swap.current; do
        [ -f "$base2/$f" ] && { echo "[$f]"; sed -n '1,200p' "$base2/$f"; }
      done
    fi
  fi
}

# Prime a first snapshot immediately (so we have data even if it dies fast)
# Prime a first snapshot immediately (so we have data even if it dies fast)
snap_proc "$pid" "$snapdir" || true
# Optionally snapshot child processes (advanced, uncomment if needed)
# for cpid in $(pgrep -P $pid); do snap_proc "$cpid" "$snapdir/child_$cpid" || true; done

printf "%-6s %-12s %-10s %-6s %-6s %s\n" "PID" "ETIME" "RSS(KB)" "%MEM" "%CPU" "CMD"

# Monitor loop
while kill -0 "$pid" 2>/dev/null; do
  ps -p "$pid" -o pid,etime,rss,%mem,%cpu,cmd --no-headers
  # Refresh a lightweight snapshot each tick (fast files only)
  cat "/proc/$pid/status"       > "$snapdir/status.txt" 2>/dev/null || true
  cat "/proc/$pid/smaps_rollup" > "$snapdir/smaps_rollup.txt" 2>/dev/null || true
  cat "/proc/$pid/oom_score"    > "$snapdir/oom_score.txt" 2>/dev/null || true
  # Optional GPU sample
  if (( gpu == 1 )) && command -v nvidia-smi >/dev/null 2>&1; then
    echo "[GPU] $(ts)"
    nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu,utilization.memory --format=csv,noheader 2>/dev/null || true
    nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader 2>/dev/null | head -n 10 || true
  fi
  sleep "$interval"
done

# Get true exit status
set +e
wait "$pid"; status=$?
set -e
echo "---- process exit ----"
if (( status >= 128 )); then
  sig=$((status-128)); echo "exit_status=$status (terminated_by_signal=$sig $(kill -l "$sig" 2>/dev/null || echo SIG?))"
else
  echo "exit_status=$status"
fi

# Print /usr/bin/time stats
[ -s "$time_out" ] && { echo "---- /usr/bin/time -v ----"; cat "$time_out"; }

# Show last snapshots we captured
if [ -f "$snapdir/status.txt" ]; then
  echo "---- /proc/$pid (last snapshot) ----"
  [ -f "$snapdir/exe.txt" ]              && { echo "exe=$(cat "$snapdir/exe.txt")"; }
  [ -f "$snapdir/cmdline_readable.txt" ] && { echo "cmdline=$(cat "$snapdir/cmdline_readable.txt")"; }
  [ -f "$snapdir/oom_score.txt" ]        && { echo "oom_score=$(cat "$snapdir/oom_score.txt")"; }
  [ -f "$snapdir/oom_score_adj.txt" ]    && { echo "oom_score_adj=$(cat "$snapdir/oom_score_adj.txt")"; }
  [ -f "$snapdir/limits.txt" ]           && { echo "--- limits ---"; sed -n '1,40p' "$snapdir/limits.txt"; }
  [ -f "$snapdir/smaps_rollup.txt" ]     && { echo "--- smaps_rollup (PSS/RSS summary) ---"; sed -n '1,80p' "$snapdir/smaps_rollup.txt"; }
fi

# Dump cgroup info using the saved path (works even after pid exit)
cgroup_dump_by_path "$snapdir"

# Tail logs around now (no sudo to avoid blocking under nohup)
echo "---- recent kernel messages (last ${tail_mins}m) ----"
if command -v journalctl >/dev/null 2>&1; then
  journalctl -k --since "${tail_mins} minutes ago" | tail -n 400 || true
else
  dmesg | tail -n 400 || true
fi

echo "---- recent executable messages (last ${tail_mins}m) ----"
if command -v journalctl >/dev/null 2>&1; then
  if [ -f "$snapdir/exe.txt" ]; then
    exe=$(cat "$snapdir/exe.txt")
    journalctl "$exe" --since "${tail_mins} minutes ago" | tail -n 200 || true
  fi
  journalctl _PID="$pid" --since "${tail_mins} minutes ago" -o short-precise | tail -n 200 || true
fi

# GPU/driver events (NVIDIA/Xid)
if command -v dmesg >/dev/null 2>&1; then
  echo "---- GPU/driver events (grep NVRM/Xid/timeout/reset, last ${tail_mins}m approx) ----"
  dmesg -T 2>/dev/null | grep -E "NVRM|Xid|timeout|reset|oom" | tail -n 200 || true
fi

# Coredump pointer if any
if command -v coredumpctl >/dev/null 2>&1; then
  if coredumpctl --no-pager info "$pid" >/dev/null 2>&1; then
    echo "---- coredump found for pid=$pid ----"
    coredumpctl --no-pager info "$pid" || true
    echo "Use: coredumpctl gdb $pid"
  fi
fi

echo "[$(ts)] [dbg] snapshots at: $snapdir"

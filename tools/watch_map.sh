#!/bin/bash
LOGDIR="logs"
STATE="/tmp/watch_map.state"
touch "$STATE"
echo "=== mAP watcher started $(date) ==="
while true; do
  for stream in RGB LARGE FLOW BODY; do
    case $stream in
      RGB)   newest=$(ls -t "$LOGDIR"/train_adatad_1*.log 2>/dev/null | grep -v large | grep -v body | grep -v flow | head -1) ;;
      LARGE) newest=$(ls -t "$LOGDIR"/train_adatad_large_*.log 2>/dev/null | head -1) ;;
      FLOW)  newest=$(ls -t "$LOGDIR"/train_adatad_flow_*.log 2>/dev/null | head -1) ;;
      BODY)  newest=$(ls -t "$LOGDIR"/train_adatad_body_*.log 2>/dev/null | head -1) ;;
    esac
    [ -z "$newest" ] && continue
    grep "Average-mAP" "$newest" 2>/dev/null | while read -r line; do
      val=$(echo "$line" | grep -oP 'Average-mAP: \K[0-9.]+')
      ts=$(echo "$line" | grep -oP '^[0-9-]+ [0-9:]+')
      key="${stream}_${newest##*/}_${ts}"
      grep -qF "$key" "$STATE" && continue
      echo "$key" >> "$STATE"
      if [ "$val" = "0.00" ]; then
        echo "[$ts] $stream: mAP=0.00% (still early)"
      else
        echo ""
        echo ">>> FIRST NON-ZERO mAP: $stream = ${val}% at $ts"
        echo ">>> log: $newest"
        echo ""
      fi
    done
  done
  sleep 300
done

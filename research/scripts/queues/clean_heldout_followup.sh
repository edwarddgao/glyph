#!/bin/zsh
cd /Users/edwarddgao/Documents/swipe/research
until grep -q '=== followup done' runs/clean_queue.log; do sleep 60; done
PY=.venv/bin/python; L=runs/clean_queue.log
echo "=== followup2: hws held-out users, all three models $(date)" >> $L
for m in ar_full_s1 ar_clean_s1 ar_mixed_s1; do
  $PY scripts/eval_ar_decoder.py --checkpoint runs/$m/ar_decoder.pt --splits hws_heldout/test --limit 26000 > runs/$m/beam_eval_hws_heldout.log 2>&1 || echo "eval $m heldout FAILED" >> $L
  echo "$m held-out: $(grep -E 'beam top-1|truth among' runs/$m/beam_eval_hws_heldout.log | tr '\n' ' ')" >> $L
done
echo "=== followup2 done $(date)" >> $L

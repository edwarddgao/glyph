#!/bin/zsh
cd /Users/edwarddgao/Documents/swipe/research
until grep -q '=== followup2 done' runs/clean_queue.log; do sleep 60; done
PY=.venv/bin/python; L=runs/clean_queue.log
echo "=== followup3: clean seed-2 replication $(date)" >> $L
$PY scripts/train_ar_decoder.py --train-path futo_clean/train --seed 2 --out runs/ar_clean_s2 > runs/ar_clean_s2.log 2>&1 || echo "train clean s2 FAILED" >> $L
$PY scripts/eval_ar_decoder.py --checkpoint runs/ar_clean_s2/ar_decoder.pt > runs/ar_clean_s2/beam_eval.log 2>&1 || echo "eval clean s2 FAILED" >> $L
echo "clean s2: $(grep -E 'beam top-1' runs/ar_clean_s2/beam_eval.log | tr '\n' ' ')" >> $L
$PY scripts/eval_ar_decoder.py --checkpoint runs/ar_clean_s2/ar_decoder.pt --splits hws_heldout/test --limit 26000 > runs/ar_clean_s2/beam_eval_hws_heldout.log 2>&1
echo "clean s2 held-out: $(grep -E 'beam top-1' runs/ar_clean_s2/beam_eval_hws_heldout.log | tr '\n' ' ')" >> $L
echo "=== followup3 done $(date)" >> $L

#!/bin/zsh
cd /Users/edwarddgao/Documents/swipe/research
until grep -q '=== all done' runs/clean_queue.log; do sleep 60; done
PY=.venv/bin/python; L=runs/clean_queue.log
echo "=== followup: hws bundles $(date)" >> $L
for m in ar_full_s1 ar_clean_s1; do
  PYTHONPATH=src $PY scripts/export_fused_bundle.py --checkpoint runs/$m/ar_decoder.pt --split how_we_swipe/test --out fused_${m}_hws.pkl > runs/export_${m}_hws.log 2>&1 || echo "export $m hws FAILED" >> $L
done
PYTHONPATH=src $PY scripts/sweep_prior_algebra.py --split hws --arms s1=fused_ar_full_s1_hws.pkl,clean=fused_ar_clean_s1_hws.pkl --alphas 0.4,0.6 --lams 0,0.25 > runs/sweep_clean_hws.log 2>&1
sed -n '/best:/,$p' runs/sweep_clean_hws.log >> $L
echo "=== followup done $(date)" >> $L

#!/bin/zsh
cd /Users/edwarddgao/Documents/swipe/research
PY=.venv/bin/python; L=runs/clean_queue.log
echo "=== export s1 val bundle $(date)" >> $L
PYTHONPATH=src $PY scripts/export_fused_bundle.py --checkpoint runs/ar_full_s1/ar_decoder.pt --split futo/validation --out fused_s1_val.pkl > runs/export_s1_val.log 2>&1 || echo "export s1 FAILED" >> $L
echo "=== build clean caches $(date)" >> $L
PYTHONPATH=src:scripts $PY scripts/build_clean_caches.py > runs/build_clean_caches.log 2>&1 || echo "clean caches FAILED" >> $L
cat runs/build_clean_caches.log >> $L
for arm in "ar_clean_s1:futo_clean/train" "ar_mixed_s1:futo_clean/train,hws_clean/train"; do
  name=${arm%%:*}; spec=${arm#*:}
  echo "=== train $name $(date)" >> $L
  $PY scripts/train_ar_decoder.py --train-path $spec --seed 1 --out runs/$name > runs/$name.log 2>&1 || echo "train $name FAILED" >> $L
  echo "=== eval $name $(date)" >> $L
  $PY scripts/eval_ar_decoder.py --checkpoint runs/$name/ar_decoder.pt > runs/$name/beam_eval.log 2>&1 || echo "eval $name FAILED" >> $L
  grep -E 'beam top-1' runs/$name/beam_eval.log >> $L
  echo "=== export $name $(date)" >> $L
  PYTHONPATH=src $PY scripts/export_fused_bundle.py --checkpoint runs/$name/ar_decoder.pt --split futo/validation --out fused_${name}_val.pkl > runs/export_${name}_val.log 2>&1 || echo "export $name FAILED" >> $L
done
echo "=== sweep $(date)" >> $L
PYTHONPATH=src $PY scripts/sweep_prior_algebra.py --split val --arms s1=fused_s1_val.pkl,clean=fused_ar_clean_s1_val.pkl,mixed=fused_ar_mixed_s1_val.pkl --alphas 0.4,0.6 --lams 0,0.25 > runs/sweep_clean_mixed_val.log 2>&1
sed -n '/best:/,$p' runs/sweep_clean_mixed_val.log >> $L
echo "=== all done $(date)" >> $L

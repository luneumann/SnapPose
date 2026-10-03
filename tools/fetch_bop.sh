#!/bin/bash
# Download BOP models + test images (LM, TUD-L) from HuggingFace into data/bop/<name>/ (~2 GB).
# ITODD: test ground truth is not public; use the val split instead (see docs/VALIDATION.md).
cd "$(dirname "$0")/.." || exit 1
mkdir -p data/bop && cd data/bop || exit 1
for d in tudl lm; do
  mkdir -p $d && cd $d
  for part in models test_bop19; do
    f=${d}_${part}.zip
    curl -fsL -C - -o $f "https://huggingface.co/datasets/bop-benchmark/$d/resolve/main/$f" && unzip -qo $f && rm $f
  done
  cd ..
  echo "done $d"
done

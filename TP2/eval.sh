#! /bin/bash

python ../pstal-etu/lib/accuracy.py \
  -p prediction_file.conllu \
  -g ../pstal-etu/sequoia/sequoia-ud.parseme.frsemcor.simple.test \
  -c parseme:ne \
  -f form

  # probeleme ca fonctionne pas, mauvais format de fichier
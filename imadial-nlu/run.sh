#!/bin/bash 
illc_dir="./data/ILLC-IER"
work_dir="./work_dir"

echo "Step 1: make vocab"
python run.py vocab -t ${illc_dir}/train.abstract.txt -o ${illc_dir}/vocab.bin 

echo "Step 2: train LSTM tagger"
python run.py train --train ${illc_dir}/train.abstract.txt \
                    --valid ${illc_dir}/val.abstract.txt \
                    --vocab ${illc_dir}/vocab.bin \
                    --work-dir ${work_dir}

echo "Step 3: eval tagger"
python run.py eval --work-dir ${work_dir} \
                    --tag ${illc_dir}/test.abstract.txt 

echo "Step 4: serve tagger"
python run.py serve --work-dir ${work_dir}


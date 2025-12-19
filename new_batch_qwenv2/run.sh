#!/bin/bash


python /kaggle/working/new_batch_qwenv2/runner.py --mode nouns --phase private --batch_size 10 --output_dir /kaggle/working/output
# python submission_converter.py --input submission_nouns_private.json --output nouns.tsv

echo "Pipeline Finished."
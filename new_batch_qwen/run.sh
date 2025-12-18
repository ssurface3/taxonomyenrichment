#!/bin/bash
python runner.py --mode verbs --phase public --batch_size 42

python submission_converter.py --input submission_verbs_public.json --output verbs.tsv

echo "end"
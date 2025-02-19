#!/bin/bash
# # This should not be changed
#OAR -q production
# # Remove `!` for CPUs only
#OAR -p cluster='grele'
# # Adapt as desired
#OAR -l host=1/gpu=1,walltime=24:00:00

source ~/.bashrc
conda activate base
cd av_hubert/avhubert/

# Now you can do the job e.g.
python copy_and_extract_hubert_feature.py dummy
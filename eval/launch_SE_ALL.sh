#!/bin/bash

total_seg=10  ## divide the test set into "total_seg" segments, so that each segment is proceeded by a different gpu 

itotal_seg=$((total_seg-1))


# Run speech enhancement on each segment of test files

for i in $( seq 0 $itotal_seg ); do
  full_command="./eval/single_seg_launch_SE.sh $i $total_seg"
  eval "$full_command"
done

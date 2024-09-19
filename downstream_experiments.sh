#!/bin/bash

model=$1
output_dir=$2

./pred.sh ${model} 00 ${output_dir} 0 &
./pred.sh ${model} 01 ${output_dir} 0 &
./pred.sh ${model} 02 ${output_dir} 0 &
./pred.sh ${model} 03 ${output_dir} 0 &
./pred.sh ${model} 00 ${output_dir} 0 &
./pred.sh ${model} 01 ${output_dir} 0 &
./pred.sh ${model} 02 ${output_dir} 0 &
./pred.sh ${model} 03 ${output_dir} 0 &
./pred.sh ${model} 00 ${output_dir} 0 &
./pred.sh ${model} 01 ${output_dir} 0 &
wait

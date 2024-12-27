#!/bin/bash


latest_file = "/data/ai_input/cam/20240806/1000004-20240806171801.jpg"
# 输出最新文件的完整路径
echo "Latest file: $latest_file"
str_out="/data/ai_output/cam/"
output= "${str_out}${latest_file:19:8}"
echo "$output"

# 如果目录不存在，创建它
if [ ! -d "$output" ]; then
  mkdir -p "$output"
fi

# /root/anaconda3/envs/ai/bin/python  /usr/local/ai/soft/cuda/cuda_2/1/dist/main.py $latest_file /usr/local/ai/soft/cuda/cuda_2/1/dist/pytransform/__load__.py /usr/local/ai/soft/cuda_11_8_2.run --out-dir $output  --score-thr 500 --device cuda:0 --area-thr 50000000.0 

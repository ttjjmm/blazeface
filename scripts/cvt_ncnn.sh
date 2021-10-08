#!/bin/bash

ncnn_dir=/home/ubuntu/opt/ncnn/build/tools/onnx/
onnx_dir=$(pwd)/../onnx/
cd ${ncnn_dir}
echo $(pwd)
# convert ncnn model's .param .bin files
./onnx2ncnn ${onnx_dir}/blazeface_sim.onnx ${onnx_dir}/xxx.param ${onnx_dir}/xxx.bin
# optimize ncnn model
cd ..
./ncnnoptimize ${onnx_dir}/xxx.param ${onnx_dir}/xxx.bin ${onnx_dir}/xxx-opt.param ${onnx_dir}/xxx-opt.bin 65536

#!/bin/bash

curr_path=$(pwd)
converter_path="/home/ubuntu/opt/Tengine-Convert-Tools/build/install/bin/"
cd ${converter_path}
./tm_convert_tool -f onnx -m ${src} -o ${dst}


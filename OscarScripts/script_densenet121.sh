#!/bin/bash

echo "SCRIPT: onnx_second_inference.py. File available in $INPUT_FILE_PATH"
python /opt/DownloadONNXSplitter2/onnx_second_inference.py --onnx_path=/opt/DownloadONNXSplitter2/DenseNet121_SplittedModels --input_file="$INPUT_FILE_PATH" >> $TMP_OUTPUT_DIR/output.txt

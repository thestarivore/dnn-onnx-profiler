import onnxruntime as rt
import sclblonnx as so
import onnx
import numpy as np
import argparse, sys
import pickle
import os
import time
import numpy
import json
from json import JSONEncoder

'''
  Used to run at inference an ONNX DNN from a specified layer to the end. 
  The resulting classification tensor is written on a file.

  Arguments:
  -h, --help            show this help message and exit
  --onnx_path ONNX_PATH
                        Select the path were all the Splitted ONNX Models are stored
  --input_file INPUT_FILE
                        Insert the file that contains the input tensor (a
                        list) to be fed to the network
'''
def main():
    parser=argparse.ArgumentParser()

    #parser.add_argument('--onnx_file', help='Select the ONNX File')
    parser.add_argument('--onnx_path', help='Select the path were all the Splitted ONNX Models are stored')
    parser.add_argument('--input_file', help='Insert the file that contains the input tensor (a list) to be fed to the network')
    args=parser.parse_args()

    #onnx_run_second_half(args.onnx_file, args.input_file)
    onnx_search_and_run_second_half(args.onnx_path, args.input_file)

# Run at inference the Second Half of the Splitted Model
def onnx_run_second_half(onnx_file, input_file):
  #Get the input and output of the model
  onnx_model = onnx.load(onnx_file)
  model_input = onnx_model.graph.input[0].name 
  model_output = onnx_model.graph.output[0].name

  #Get the input tensor fromt the file
  with open(input_file, 'rb') as f:
    input_tensor = pickle.load(f)

  # Run the second model with the results from the first model as input
  g = so.graph_from_file(onnx_file)

  inputs = {model_input: input_tensor}
  result = so.run(g,
                  inputs=inputs,
                  outputs=[model_output]
                  )
  print(result)
  return result

# Search the correct Model and run at inference the Second Half of the Splitted Model
def onnx_search_and_run_second_half(onnx_models_path, input_file):
  startTime = time.perf_counter()

  #Get the input tensor from the file
  with open(input_file, 'rb') as f:
    data = pickle.load(f)
  input_tensor = data["result"]
  input_layer = data["splitLayer"].replace("/", '-')
  execTime1 = data["execTime1"]
  #print(data)

  #Iterate through the subdirectories to find the ONNX model splitted at our selected layer
  #print("Search for: " + input_layer)
  for dir in os.listdir(onnx_models_path):
    if dir.find("_on_") > 0:
      index = dir.index('_on_')
      d = dir[index+4:]
      #print("Check: " + d)
      if d == input_layer:
        #print("Found Layer: " + d)
        onnx_file = onnx_models_path + "/" + dir + "/second_half.onnx"
        break

  #Get the input and output of the model
  onnx_model = onnx.load(onnx_file)
  model_input = onnx_model.graph.input[0].name 
  model_output = onnx_model.graph.output[0].name

  # Run the second model with the results from the first model as input
  g = so.graph_from_file(onnx_file)

  inputs = {model_input: input_tensor}
  result = so.run(g,
                  inputs=inputs,
                  outputs=[model_output]
                  )
  endTime = time.perf_counter()
  dictData = {
    "splitLayer": input_layer,
    "execTime1": execTime1,
    "execTime2": endTime-startTime,
    "result": result[0],
  }

  # Convert into JSON:
  returnData = json.dumps(dictData, cls=NumpyArrayEncoder)

  print(returnData)
  return returnData

class NumpyArrayEncoder(JSONEncoder):
    def default(self, obj):
        if isinstance(obj, numpy.ndarray):
            return obj.tolist()
        return JSONEncoder.default(self, obj)

if __name__ == "__main__":
    main()

import sclblonnx as so
import onnx
from skl2onnx.helpers.onnx_helper import select_model_inputs_outputs
from skl2onnx.helpers.onnx_helper import save_onnx_model
from skl2onnx.helpers.onnx_helper import enumerate_model_node_outputs
from skl2onnx.helpers.onnx_helper import load_onnx_model
import numpy as np
import argparse, sys
import os

'''
  Manages operations on ONNX DNN Models such as layer visualizzation and splitting.

  Arguments:
  -h, --help            show this help message and exit
  --operation OPERATION
                        Select the operation to be performed on the ONNX Model
                        (list_layers | print_model | split_model |
                        split_model_all)
  --onnx_file ONNX_FILE
                        Select the ONNX File
  --split_layer SPLIT_LAYER
                        Select the layer where the slit must take place on the
                        ONNX Model
'''
def main():
    parser=argparse.ArgumentParser()

    parser.add_argument('--operation', help='Select the operation to be performed on the ONNX Model (list_layers | print_model | split_model | split_model_all)')
    parser.add_argument('--onnx_file', help='Select the ONNX File')
    parser.add_argument('--split_layer', help='Select the layer where the slit must take place on the ONNX Model')
    args=parser.parse_args()
    print ("Operation: " + args.operation)

    if args.operation == "list_layers":
        onnx_list_model_layers(args.onnx_file)
    elif args.operation == "print_layers":
        onnx_model_details(args.onnx_file)
    elif args.operation == "split_model":
        onnx_model_split(args.onnx_file, args.split_layer)
    elif args.operation == "split_model_all":
        onnx_model_split_all(args.onnx_file)


def onnx_list_model_layers(onnx_file):
  model_onnx = load_onnx_model(onnx_file)
  for out in enumerate_model_node_outputs(model_onnx):
      print(out)

def onnx_model_details(onnx_file):
  onnx_model = onnx.load(onnx_file)
  print(onnx_model)

def onnx_model_split(onnx_file, layer):
  print("Split at layer: " + layer)

  #Load the Onnx Model
  model_onnx = load_onnx_model(onnx_file)

  #Split and save the first half of the ONNX Model at the specified layer
  print("Split and get the first model..")
  num_onnx = select_model_inputs_outputs(model_onnx, layer)
  save_onnx_model(num_onnx, "first_half.onnx")

  #Split and save the second half of the ONNX Model
  print("Split and get the second model..")
  output_path = 'second_half.onnx'
  input_names = [layer]
  output_names = []
  onnx_model = onnx.load(onnx_file)
  for i in range(len(onnx_model.graph.output)):
    print("Output layer: " + onnx_model.graph.output[i].name)
    output_names.append(onnx_model.graph.output[i].name)

  onnx.utils.extract_model(onnx_file, output_path, input_names, output_names)
  print("Finished!")

def onnx_model_split_all(onnx_file):
  #Load the Onnx Model
  model_onnx = load_onnx_model(onnx_file)

  #Make a split for every layer in the model
  ln = 0
  for layer in enumerate_model_node_outputs(model_onnx):
    #Ignore the last layer
    if layer != list(enumerate_model_node_outputs(model_onnx))[-1]:
      folder = "split_" + str(ln) + "_on_" + layer.replace("/", '-')
      if not(os.path.exists(folder) and os.path.isdir(folder)):
        print("Create Folder: " + folder)
        os.mkdir(folder)
      
      #Split and save the first half of the ONNX Model at the specified layer
      print("Split at layer" + str(ln) + ": " + layer)
      print("Split and get the first model..")
      num_onnx = select_model_inputs_outputs(model_onnx, layer)
      save_onnx_model(num_onnx, folder+"/first_half.onnx")

      #Split and save the second half of the ONNX Model
      print("Split and get the second model..\n")
      output_path = folder+'/second_half.onnx'
      input_names = [layer]
      output_names = []
      onnx_model = onnx.load(onnx_file)
      for i in range(len(onnx_model.graph.output)):
        output_names.append(onnx_model.graph.output[i].name)

      onnx.utils.extract_model(onnx_file, output_path, input_names, output_names)
      ln = ln + 1
  print("Finished!")

if __name__ == "__main__":
    main()

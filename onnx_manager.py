from numpy.core.numeric import True_
import sclblonnx as so
import onnx
from skl2onnx.helpers.onnx_helper import select_model_inputs_outputs
from skl2onnx.helpers.onnx_helper import save_onnx_model
from skl2onnx.helpers.onnx_helper import enumerate_model_node_outputs
from skl2onnx.helpers.onnx_helper import load_onnx_model
import datetime
import numpy as np
import argparse, sys
import time
import json
import os
from onnx_first_inference import onnx_run_first_half

'''
  Manages operations on ONNX DNN Models such as layer visualizzation and splitting.

  Arguments:
  -h, --help            show this help message and exit
  --operation OPERATION
                        Select the operation to be performed on the ONNX Model
                        (list_layers | print_model | split_model | split_model_all | run)
  --onnx_file ONNX_FILE
                        Select the ONNX File
  --split_layer SPLIT_LAYER
                        Select the layer where the slit must take place on the
                        ONNX Model
  --onnx_path ONNX_PATH
                        Select the path were all the Splitted ONNX Models are stored
  --image_file IMAGE_FILE                     Select the Image File
  --image_size_x IMAGE_SIZE_X                 Select the Image Size X
  --image_size_y IMAGE_SIZE_Y                 Select the Image Size Y
  --image_is_grayscale IMAGE_IS_GRAYSCALE     Indicate if the Image is in grayscale
'''
def main():
    parser=argparse.ArgumentParser()

    parser.add_argument('--operation', help='Select the operation to be performed on the ONNX Model (list_layers | print_model | split_model | split_model_all)')
    parser.add_argument('--onnx_file', help='Select the ONNX File')
    parser.add_argument('--split_layer', help='Select the layer where the slit must take place on the ONNX Model')
    parser.add_argument('--onnx_path', help='Select the path were all the Splitted ONNX Models are stored')
    parser.add_argument('--image_file', help='Select the Image File')
    parser.add_argument('--image_size_x', help='Select the Image Size X')
    parser.add_argument('--image_size_y', help='Select the Image Size Y')
    parser.add_argument('--image_is_grayscale', help='Indicate if the Image is in grayscale')
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
    elif args.operation == "run":
        onnx_run_complete(args.onnx_path, args.split_layer, args.image_file, int(args.image_size_x), int(args.image_size_y), bool(args.image_is_grayscale))


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

def onnx_run_complete(onnx_path, split_layer, image_file, img_size_x, img_size_y, is_grayscale = False):
  #Iterate through the subdirectories to find the ONNX model splitted at our selected layer
  split_layer = split_layer.replace("/", '-').replace(":", '_')
  print("Search for: " + split_layer)
  for dir in os.listdir(onnx_path):
    if dir.find("_on_") > 0:
      index = dir.index('_on_')
      d = dir[index+4:]
      print("Check: " + d)
      if d == split_layer:
        print("Found Layer: " + d)
        onnx_file = onnx_path + "/" + dir + "/first_half.onnx"
        break
  
  print("\n ###Start the 1st Inference Execution Locally\n")

  #Run at Inference the First part of the ONNX DNN Model 
  result1 = onnx_run_first_half(onnx_file, image_file, img_size_x, img_size_y, is_grayscale)    
  #print(result1)

  #Clear the Jobs in the OSCAR Cloud before launching a new Job
  os.system("./oscar-cli service logs remove onnx-test3 --all")
  
  #Run at Inference the Second part of the ONNX DNN Model 
  print("\n ###Start the 2nd Inference Execution on Cloud (OSCAR)\n")
  outputStrBefore = os.system("./mc ls local/onnx3/output")
  res = os.system("./mc cp results.txt local/onnx3/input/results.txt")
  
  #Check when is the Cloud execution terminated on OSCAR
  execFinished = False
  import time
  while not execFinished:
    time.sleep(0.1)
    print(".")
    outputStrAfter = os.system("./mc ls local/onnx3/output")
    if outputStrBefore == outputStrAfter:
      execFinished = True
  
  #Get the result from the MinIO Bucket
  os.system("./mc cp local/onnx3/output/output.txt output.json")
  # Opening JSON file
  f = open('output.json')
  # Returns JSON object as a dictionary
  data = json.load(f)
  # Closing file
  f.close()

  print("\n ###execTime1: " + str(data["execTime1"]) + "sec")
  print("\n ###execTime2: " + str(data["execTime2"]) + "sec")

  ### Get the Cluster's timings via oscar-cli
  # Get OSCAR Job Name
  ret = ""
  succeeded = False
  while ret == "" or not succeeded:
    ret = os.popen("./oscar-cli service logs list onnx-test3").read()
    print(">" + ret)
    if not ret == "":
      resLines = ret.split("\n",2)
      jobStatus = resLines[1].split(None, 2)[1]
      print ("jobStatus: " + jobStatus)
      if jobStatus == "Succeeded":
        succeeded = True
    time.sleep(0.5)
    print(".")
  #print(">" + ret)
  resLines = ret.split("\n",2)
  jobName = resLines[1].split(None, 1)[0]
  print("jobName: " + jobName + "\n")

  # Get the Logs of the Job run on OSCAR
  ret = os.popen("./oscar-cli service logs get onnx-test3 " + jobName).read()
  print(">" + ret)
  firstTime = None
  secodTime = None
  for line in ret.split("\n"):
    if line[:3].isdigit():
      time = line.split(" - ", 2)[0]+"000"
      #print("time before parsing: " + time)
      time = datetime.datetime.strptime(time, '%Y-%m-%d %H:%M:%S,%f')  #format(datetime.datetime.now())
      if firstTime == None:
        firstTime = time
      else:
        secodTime = time
  execTime3 = (secodTime - firstTime).total_seconds() 
  print("\n ###execTime1: " + str(data["execTime1"]) + "sec")
  print("\n ###execTime2: " + str(data["execTime2"]) + "sec")
  print("\n ###execTime3: " + str(execTime3) + "sec")

if __name__ == "__main__":
    main()

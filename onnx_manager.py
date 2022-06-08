from numpy.core.numeric import True_
#import sclblonnx as so
import onnx
#from skl2onnx.helpers.onnx_helper import select_model_inputs_outputs
#from skl2onnx.helpers.onnx_helper import save_onnx_model
from skl2onnx.helpers.onnx_helper import enumerate_model_node_outputs
from skl2onnx.helpers.onnx_helper import load_onnx_model
import datetime
import numpy as np
import argparse, sys
import time
import json
import os
import csv
from onnx_first_inference import onnx_run_first_half
import matplotlib.pyplot as plt
import pandas as pd
import shutil
import pickle
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array
import onnxruntime
from onnxruntime.quantization import quantize_static, quantize_dynamic, CalibrationDataReader, QuantFormat, QuantType
from argparse import ArgumentParser
import asyncio
from onnx_opcounter import calculate_params
from onnx2json import convert

RESULTS_CSV_FILE = 'time_table.csv'
RESULTS_CSV_FILE2 = 'time_table_2.csv'
FINAL_RESULTS_CSV_FILE = 'time_table_final.csv'
AVG_RESULTS_CSV_FILE = 'time_table_avg.csv'
INPUT_PICKLE_FILE = 'input.txt'
OUTPUT_PICKLE_FILE = 'results.txt'
NEUTRON_INSTALLATION_PATH = '/snap/bin/neutron'

MODEL_SPLIT_FIRST_FILE = 'first_half.onnx'
MODEL_SPLIT_SECOND_FILE = 'second_half.onnx'

# Prefer ACL Execution Provider over CPU Execution Provider
ACL_EP_list       = ['ACLExecutionProvider']
# Prefer OpenVINO Execution Provider over CPU Execution Provider
TensorRT_EP_list  = ['TensorrtExecutionProvider']
# Prefer OpenVINO Execution Provider over CPU Execution Provider
OpenVINO_EP_list  = ['OpenVINOExecutionProvider']
# Prefer CUDA Execution Provider over CPU Execution Provider
GPU_EP_list       = ['CUDAExecutionProvider']
# Prefer CPU Execution Provider
CPU_EP_list       = ['CPUExecutionProvider']

dictTensors = {}
sshHost = ""
sshPsk = ""

def main():
  '''
    Manages operations on ONNX DNN Models such as layer visualizzation and splitting.

    Arguments:
    -h, --help            Show this help message and exit
    --operation OPERATION
                          Select the operation to be performed on the ONNX Model
                          (list_layers | print_model | split_model | split_model_all | multi_split_model | early_exit_split_model | data_processing | 
                           run | run_all | plot_results | quant_model | show_graph)
    --onnx_file ONNX_FILE                       Select the ONNX File
    --split_layer SPLIT_LAYER                   Select the layer where the slit must take place on the ONNX Model
    --split_layers SPLIT_LAYERS                 Select the list of layers where the slit must take place on the ONNX Model
    --outputs OUTPUTS                           Select the output and the early exits where the slit must take place on the ONNX Model (the actual split will take place above the early exit)
    --onnx_path ONNX_PATH                       Select the path were all the Splitted ONNX Models are stored
    --image_file IMAGE_FILE                     Select the Image File
    --image_file IMAGE_BATCH                    Select the Image Folder containing the Batch of images
    --image_size_x IMAGE_SIZE_X                 Select the Image Size X
    --image_size_y IMAGE_SIZE_Y                 Select the Image Size Y
    --image_is_grayscale IMAGE_IS_GRAYSCALE     Indicate if the Image is in grayscale
    --results_file RESULTS_FILE                 Select the Results File(.csv)
    --minio_bucket MINIO_BUCKET                 Insert the name of the MinIO Bucket
    --oscar_service OSCAR_SERVICE               Insert the name of the OSCAR Service
    --kube_namespace KUBE_NAMESPACE             Insert the namespace used in Kubernetes for the OSCAR Service
    --quant_type QUANT_TYPE                     Choose weight type used during model quantization
                                                Dependencies: https://www.tensorflow.org/install/source#gpu
    --platform PLATFORM                         Choose the platform where RUN_ALL/RUN is executed, in order to use the right client for MinIO, OSCAR and Kubernetes
                                Requirements for GPU: https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html#requirements
    --device_type Device_TYPE   Select DeviceType
    --exec_type EXEC_TYPE                       Select Execution Provider at inference: CPU (default) | GPU | OpenVINO | TensorRT | ACL
                                'CPU_FP32', 'GPU_FP32', 'GPU_FP16', 'MYRIAD_FP16', 'VAD-M_FP16', 'VAD-F_FP32',
                                Options are: (Any hardware target can be assigned if you have the access to it)
                                'HETERO:MYRIAD,CPU',  'MULTI:MYRIAD,GPU,CPU'
    --rep REP                                   Number of repetitions
    --kube_ssh_host KUBE_SSH_HOST               SSH Host to use when we need to run Kubernetes Client remotely (form: "username@ip_address")
    --kube_ssh_psk KUBE_SSH_PSK                 SSH Host\'s Password to use when we need to run Kubernetes Client remotely (form: "password")
  '''
  parser=argparse.ArgumentParser(
    description='''
ONNX Manager: Manages operations on ONNX DNN Models such as: 
  - the execution of the complete cycle edge-cloud;
  - the creation of splitted models;
  - model layer visualizzation;
  - data processing (of images or batches); 
  - the quantization of models;
  - plotting of results;
  - showing the graph on ONNX Models
    ''',
    epilog='''
Examples:
> python onnx_manager.py --operation run --split_layer sequential/dense_1/MatMul:0 
                         --onnx_path LENET_SplittedModels/ --image_file=images/mnist_test.jpg 
                         --image_size_x=32 --image_size_y=32 --image_is_grayscale=True
> python onnx_manager.py --operation run_all --onnx_file lenet.onnx --onnx_path LENET_SplittedModels/ 
                         --image_file=images/mnist_test.jpg --image_size_x=32 --image_size_y=32 --image_is_grayscale=True

> python onnx_manager.py --operation list_layers --onnx_file mobilenet_v2.onnx
> python onnx_manager.py --operation split_model --onnx_file mobilenet_v2.onnx 
                         --split_layer sequential/mobilenetv2_1.00_160/block_3_project_BN/FusedBatchNormV3:0

> python onnx_manager.py --operation run --split_layer sequential/mobilenetv2_1.00_160/block_5_add/add:0 
                         --onnx_path MobileNetV2_SplittedModles 
                         --image_file=images/mobilenet_misc/141340262_ca2e576490_jpg.rf.a9e7a7e679798619924bbc5cade9f806.jpg 
                         --image_size_x=160 --image_size_y=160 --image_is_grayscale=False --minio_bucket=onnx-test-mobilenet 
                         --oscar_service=onnx-test-mobilenet
> python onnx_manager.py --operation run --split_layer sequential/mobilenetv2_1.00_160/block_5_add/add:0 
                         --onnx_path MobileNetV2_SplittedModles --image_batch=images/mobilenet_batch 
                         --image_size_x=160 --image_size_y=160 --image_is_grayscale=False 
                         --minio_bucket=onnx-test-mobilenet --oscar_service=onnx-test-mobilenet

> python onnx_manager.py --operation run_all --onnx_file mobilenet_v2.onnx --onnx_path MobileNetV2_SplittedModles 
                         --image_file=images/mobilenet_misc/141340262_ca2e576490_jpg.rf.a9e7a7e679798619924bbc5cade9f806.jpg 
                         --image_size_x=160 --image_size_y=160 --image_is_grayscale=False 
                         --minio_bucket=onnx-test-mobilenet --oscar_service=onnx-test-mobilenet
> python onnx_manager.py --operation run_all --onnx_file mobilenet_v2.onnx --onnx_path MobileNetV2_SplittedModles 
                         --image_file=images/mobilenet_batch --image_size_x=160 --image_size_y=160 --image_is_grayscale=False 
                         --minio_bucket=onnx-test-mobilenet --oscar_service=onnx-test-mobilenet

> python onnx_manager.py --operation data_processing --image_file=images/mobilenet_misc/141340262_ca2e576490_jpg.rf.a9e7a7e679798619924bbc5cade9f806.jpg 
                         --image_size_x=160 --image_size_y=160 --image_is_grayscale=False
    ''',
    formatter_class=argparse.RawTextHelpFormatter
  )
  parser.add_argument('--operation', help='Select the operation to be performed on the ONNX Model',
                      choices=['list_layers', 'print_model', 'split_model', 'split_model_all', 'multi_split_model', 'early_exit_split_model',
                               'data_processing', 'run', 'run_all', 'plot_results', 'quant_model', 'show_graph'])
  parser.add_argument('--onnx_file', help='Select the ONNX File')
  parser.add_argument('--xml_file', help='Select the XML File (OpenVINO Optimized Model)')
  parser.add_argument('--split_layer', help='Select the layer where the slit must take place on the ONNX Model')
  parser.add_argument('--split_layers', help='Select the list of layers where the slit must take place on the ONNX Model', 
                                        dest='split_layers', type=str, nargs='+')
  parser.add_argument('--outputs',  help='Select the output and the early exits where the slit must take place on the ONNX Model (the actual split will take place above the early exit)', 
                                    dest='outputs', type=str, nargs='+')
  parser.add_argument('--onnx_path', help='Select the path were all the Splitted ONNX Models are stored')
  parser.add_argument('--image_file', help='Select the Image File')
  parser.add_argument('--image_batch', help='Select the Image Folder containing the Batch of images')
  parser.add_argument('--image_size_x', help='Select the Image Size X')
  parser.add_argument('--image_size_y', help='Select the Image Size Y')
  parser.add_argument('--image_is_grayscale', help='Indicate if the Image is in grayscale')
  parser.add_argument('--results_file', help='Select the Results File(.csv)')
  parser.add_argument('--minio_bucket', help='Insert the name of the MinIO Bucket')
  parser.add_argument('--oscar_service', help='Insert the name of the OSCAR Service')
  parser.add_argument('--kube_namespace', help='Insert the namespace used in Kubernetes for the OSCAR Service')
  parser.add_argument("--quant_type", help='Choose weight type used during model quantization', default=QuantType.QUInt8, choices=list(QuantType))
  parser.add_argument("--platform", help='Choose the platform where RUN_ALL/RUN is executed, in order to use the right client for MinIO, OSCAR and Kubernetes', 
                      choices=['AMD64', 'ARM64'])
  parser.add_argument('--exec_type', help='Select Execution Provider at inference', choices=['CPU', 'GPU', 'OpenVINO', 'TensorRT', 'ACL'])
  parser.add_argument('--device_type', help='Select DeviceType: (CPU_FP32, GPU_FP32, GPU_FP16, MYRIAD_FP16, VAD-M_FP16, VAD-F_FP32, \
                                             HETERO:MYRIAD,CPU,  MULTI:MYRIAD,GPU,CPU)')
  parser.add_argument("--rep", help='Number of repetitions', type=int)
  parser.add_argument('--kube_ssh_host', help='SSH Host to use when we need to run Kubernetes Client remotely (form: "username@ip_address")')
  parser.add_argument('--kube_ssh_psk', help='SSH Host\'s Password to use when we need to run Kubernetes Client remotely (form: "password")')
  args=parser.parse_args()
  print ("Operation: " + args.operation)

  #Get Execution Provider
  exec_provider = None
  if args.exec_type == "ACL":
    exec_provider = ACL_EP_list
  if args.exec_type == "TensorRT":
    exec_provider = TensorRT_EP_list
  elif args.exec_type == "OpenVINO":
    exec_provider = OpenVINO_EP_list
  elif args.exec_type == "GPU":
    oexec_provider = GPU_EP_list
  else:
    exec_provider = CPU_EP_list

  #Get SSH Credentials
  global sshHost
  global sshPsk
  sshHost = args.kube_ssh_host
  sshPsk = args.kube_ssh_psk

  #Choose the operation
  if args.operation == "list_layers":
      onnx_list_model_layers(args.onnx_file)
  elif args.operation == "print_layers":
      onnx_model_details(args.onnx_file)
  elif args.operation == "split_model":
      onnx_model_split(args.onnx_file, args.split_layer)
  elif args.operation == "split_model_all":
      onnx_model_split_all(args.onnx_file)
  elif args.operation == "multi_split_model":
      onnx_model_multi_split(args.onnx_file, args.split_layers)
  elif args.operation == "early_exit_split_model":
      onnx_model_early_exits_split(args.onnx_file, args.outputs)
  elif args.operation == "data_processing":
      onnx_import_data(args.image_file, 
                            args.image_batch,
                            int(args.image_size_x), 
                            int(args.image_size_y), 
                            args.image_is_grayscale == "True")
  elif args.operation == "run":
      onnx_run_complete(args.onnx_path, 
                        args.split_layer, 
                        args.image_file, 
                        args.image_batch,
                        int(args.image_size_x), 
                        int(args.image_size_y), 
                        args.image_is_grayscale == "True",
                        args.minio_bucket,
                        args.oscar_service,
                        args.kube_namespace,
                        args.platform,
                        exec_provider,
                        args.device_type,
                        args.xml_file)
  elif args.operation == "run_all":
      onnx_run_all_complete(args.onnx_file, 
                            args.onnx_path, 
                            args.image_file, 
                            args.image_batch,
                            int(args.image_size_x), 
                            int(args.image_size_y), 
                            args.image_is_grayscale == "True",
                            args.minio_bucket,
                            args.oscar_service,
                            args.kube_namespace,
                            args.platform,
                            args.rep,
                            exec_provider,
                            args.device_type,
                            args.xml_file)
  elif args.operation == "plot_results":
      plot_results(args.results_file)
  elif args.operation == "quant_model":
      quantize_dynamic(args.onnx_file, 'quant_'+args.onnx_file, weight_type=args.quant_type)
  elif args.operation == "show_graph":
      onnx_show_graph(args.onnx_file)

def onnx_list_model_layers(onnx_file):
  '''
  List all the layers of an ONNX DNN Model

  :param onnx_file: the ONNX file to analize
  '''
  model_onnx = load_onnx_model(onnx_file)
  for out in enumerate_model_node_outputs(model_onnx):
      print(out)

def onnx_model_details(onnx_file):
  '''
  Print the details of an ONNX DNN Model

  :param onnx_file: the ONNX file to analize
  '''
  onnx_model = onnx.load(onnx_file)
  print(onnx_model)

def onnx_model_split(onnx_file, layer):
  '''
  Split an ONNX Model into two Models and save the ONNX Files

  :param onnx_file: the ONNX file to split (it can also be an already splitted model)
  :param layer: the name of the DNN layer where the split has to be performed on the model choosen
  '''
  print("Split at layer: " + layer)

  #Load the Onnx Model
  model_onnx = load_onnx_model(onnx_file)

  #Split and save the first half of the ONNX Model at the specified layer
  print("Split and get the first model..")
  ##num_onnx = select_model_inputs_outputs(model_onnx, layer)
  ##save_onnx_model(num_onnx, MODEL_SPLIT_FIRST_FILE)
  output_path = MODEL_SPLIT_FIRST_FILE
  input_names = []
  onnx_model = onnx.load(onnx_file)
  for i in range(len(onnx_model.graph.input)):
    input_names.append(onnx_model.graph.input[i].name)
  output_names = [layer]
  onnx.utils.extract_model(onnx_file, output_path, input_names, output_names)

  #Split and save the second half of the ONNX Model
  print("Split and get the second model..")
  output_path = MODEL_SPLIT_SECOND_FILE
  input_names = [layer]
  output_names = []
  onnx_model = onnx.load(onnx_file)
  for i in range(len(onnx_model.graph.output)):
    print("Output: " + onnx_model.graph.output[i].name)
    output_names.append(onnx_model.graph.output[i].name)

  onnx.utils.extract_model(onnx_file, output_path, input_names, output_names)
  print("Finished!")

  print("\nGraph of the two models splitted..")
  onnx_show_graph(MODEL_SPLIT_FIRST_FILE + ' ' + MODEL_SPLIT_SECOND_FILE);

def onnx_model_split_all(onnx_file):
  '''
  For every layer in the ONNX Model, split the ONNX File into two Models and save them in a directory named after the layer label.

  :param onnx_file: the ONNX file to split (it can also be an already splitted model)
  '''
  #Load the Onnx Model
  model_onnx = load_onnx_model(onnx_file)

  #Make a split for every layer in the model
  ln = 0
  for layer in enumerate_model_node_outputs(model_onnx):
    #Ignore the last layer and RELU Layers
    if layer != list(enumerate_model_node_outputs(model_onnx))[-1] and "relu" not in layer.lower():
      folder = "split_" + str(ln) + "_on_" + layer.replace("/", '-').replace(":", '_')
      if not(os.path.exists(folder) and os.path.isdir(folder)):
        print("Create Folder: " + folder)
        os.mkdir(folder)
      
      #Split and save the first half of the ONNX Model at the specified layer
      print("Split at layer" + str(ln) + ": " + layer)
      print("Split and get the first model..")
      ##num_onnx = select_model_inputs_outputs(model_onnx, layer)
      ##save_onnx_model(num_onnx, folder+"/first_half.onnx")
      output_path = folder+'/first_half.onnx'
      input_names = []
      onnx_model = onnx.load(onnx_file)
      for i in range(len(onnx_model.graph.input)):
        input_names.append(onnx_model.graph.input[i].name)
      output_names = [layer]
      onnx.utils.extract_model(onnx_file, output_path, input_names, output_names)

      #Split and save the second half of the ONNX Model
      print("Split and get the second model..\n")
      output_path = folder+'/second_half.onnx'
      input_names = [layer]
      output_names = []
      onnx_model = onnx.load(onnx_file)
      for i in range(len(onnx_model.graph.output)):
        output_names.append(onnx_model.graph.output[i].name)

      try:
        onnx.utils.extract_model(onnx_file, output_path, input_names, output_names)
      except Exception as e:
        print(e)
        shutil.rmtree(folder)

      ln = ln + 1
  print("Finished!")

def onnx_model_multi_split(onnx_file, layers):
  '''
  Split an ONNX Model into MULTIPLE Models and save the ONNX Files

  :param onnx_file: the ONNX file to split (it can also be an already splitted model)
  :param layers: the list of names of the DNN layers where the splits have to be performed on the choosen model
  '''
  print("1st Split at layer: " + layers[0])

  #Load the Onnx Model
  #model_onnx = load_onnx_model(onnx_file)
  onnx_model = onnx.load(onnx_file)

  #Split and save the first part of the ONNX Model at the specified layer
  print("Split and get the first model..")
  #num_onnx = select_model_inputs_outputs(model_onnx, layers[0])
  #save_onnx_model(num_onnx, "part1.onnx")
  output_path = "part1.onnx"
  input_names = []
  for i in range(len(onnx_model.graph.input)):
    input_names.append(onnx_model.graph.input[i].name)
  output_names = [layers[0]]
  onnx.utils.extract_model(onnx_file, output_path, input_names, output_names)

  onnx_files = "part1.onnx"
  for i in range(1, len(layers)+1):
    #Split and save another part of the ONNX Model
    print("Split and get the " +str(i+1)+ " model..")
    output_path = 'part'+str(i+1)+'.onnx'
    onnx_files = onnx_files + ' ' + output_path
    input_names = [layers[i-1]]
    if i == len(layers):
      output_names = [] 
      for j in range(len(onnx_model.graph.output)):
        print("Output layer: " + onnx_model.graph.output[j].name)
        output_names.append(onnx_model.graph.output[j].name)
    else:
      output_names = [layers[i]]
      print("Output layer: " + output_names[0])

    onnx_model = onnx.load(onnx_file)
    onnx.utils.extract_model(onnx_file, output_path, input_names, output_names)
    print("Finished this part!")

  print("\nGraph of the " + str(len(layers)+1) + " models splitted..")
  onnx_show_graph(onnx_files)

def onnx_model_early_exits_split(onnx_file, outputs):
  '''
  Split an ONNX Model with EarlyExits into MULTIPLE Models and save the ONNX Files

  :param onnx_file: the ONNX file to split (it can also be an already splitted model)
  :param outputs: the list of names of the DNN nodes that represent the output or the early exits
  '''
  print("Outputs: " + str(outputs))

  #Load the Onnx Model
  model_onnx = load_onnx_model(onnx_file)

  #Get all the outputs of the model from the onnx graph
  outputs_read = []
  for i in range(len(model_onnx.graph.output)):
    print("Output layer: " + model_onnx.graph.output[i].name)
    outputs_read.append(model_onnx.graph.output[i].name)

  #Check that the outputs passed as argument actually exist in the model
  for output in outputs:
    if output not in outputs_read:
      print("Error: the output '"+output+"' passed as argument does not belong to the model!")
      return

  #Split the model for each output passed as argument
  num_splits = 0
  remaining_onnx = None
  split_nodes = []
  for output in outputs:
    if num_splits == 0:
      split_node = onnxGetSplitNodeForEarlyExit(model_onnx.graph, output)
      print("Split at node: " + split_node)
      split_nodes.append(split_node)

      #Perform the Split
      onnx_model_early_exits_singlesplit(onnx_file, model_onnx, split_node, num_splits, output)
    elif num_splits > 0 and num_splits < len(outputs)-1:
      split_node = onnxGetSplitNodeForEarlyExit(model_onnx.graph, output)
      print("Split at node: " + split_node)
      split_nodes.append(split_node)

      #Perform the Split
      onnx_model_early_exits_singlesplit(onnx_file, model_onnx, split_node, num_splits, output)
    elif num_splits == len(outputs)-1:
      split_node = onnxGetSplitNodeForEarlyExit(model_onnx.graph, output)
      print("Split at node: " + split_node)
      split_nodes.append(split_node)

      #Perform the Split
      onnx_model_early_exits_singlesplit(onnx_file, model_onnx, split_node, num_splits, output)

    num_splits = num_splits + 1
  
  #Split and save the BaseModel of the Full ONNX Model at the specified layers (the outputs)
  #base_model_onnx = select_model_inputs_outputs(model_onnx, split_nodes)
  #save_onnx_model(base_model_onnx, "ee_base_model.onnx")
  output_path = "ee_base_model.onnx"
  input_names = []
  onnx_model = onnx.load(onnx_file)
  for i in range(len(onnx_model.graph.input)):
    input_names.append(onnx_model.graph.input[i].name)
  output_names = split_nodes 
  onnx.utils.extract_model(onnx_file, output_path, input_names, output_names)

  # Graph the Base Model and all the other models splitted (one for each output/early-exit)
  modelsToGraph = "ee_base_model.onnx"
  for i in range(len(model_onnx.graph.output)):
    modelsToGraph = modelsToGraph + " ee_"+model_onnx.graph.output[i].name+".onnx"
  print("\nGraph all the models..")
  onnx_show_graph(modelsToGraph)

def onnx_model_early_exits_singlesplit(onnx_file, model_onnx, split_node, num_splits, output):
  '''
  Perform one Split on a ONNX Model with EarlyExits
  '''
  #Split and save the first half of the ONNX Model at the specified layer
  #remaining_onnx = select_model_inputs_outputs(model_onnx, split_node)
  #save_onnx_model(remaining_onnx, "ee_base_model.onnx")

  #Split and save the second half of the ONNX Model
  if num_splits == 0:
    print("Split and get the first model for output: " + output)
  else:
    print("Split and get the "+str(num_splits+1)+"° model for output: " + output)
  output_path = "ee_"+output+".onnx"
  input_names = [split_node]
  output_names = [output]
  onnx.utils.extract_model(onnx_file, output_path, input_names, output_names)
  print("Done..")
  #print("\nGraph of the model..")
  #onnx_show_graph(output_path)

def onnxGetSplitNodeForEarlyExit(graph, output):
  '''
  Find and return the node where the split has to be made before an output or early-exit.
  This will tipically be a couple of layers far away from the output, just before a concatenation.

  :param graph: the graph of the onnx model to inspect
  :param output: the name of the output or early-exit to insp
  :returns: split layer name
  '''
  #First find the node before the output passed as argument
  outputNode = None
  prevNode = None
  for node in graph.node:
    if output in node.output:
      outputNode = node   
      prevNode = node.input
      print(prevNode)

  #Then traferse the graph backwords
  while prevNode not in graph.input:
    for node in graph.node:
      for prev in prevNode:
        if prev in node.output:
          prevNode = node.input 
          print(prevNode)

          # the first concatenation node represents our limit, we split just after
          if 'concat' in node.input[0]:
            return node.output[0]
          break

def onnx_model_split_all_singlenode(onnx_file, tensors):
  '''
  For every layer in the ONNX Model, split the ONNX File into a single node/layer model

  :param onnx_file: the ONNX file to split (it can also be an already splitted model)
  :param tensors: dictionary containing all the the intermediate tensors
  '''
  #Load the Onnx Model
  model_onnx = load_onnx_model(onnx_file)

  #Make a Folder
  folder = "SingleLayerSplits"
  if not(os.path.exists(folder) and os.path.isdir(folder)):
    print("Create Folder: " + folder)
    os.mkdir(folder)

  #Get the list of all the layers where we must split
  listLayers = []
  for layer in tensors:
    listLayers.append(layer)

  # For each layer get extract from the full model a single layer model
  for i in range(0, len(listLayers)-1):
    layer = listLayers[i]
    nextLayer = listLayers[i+1]
    output_path = folder+'/'+layer.replace("/", '-').replace(":", '_')+'.onnx'
    #output_path = folder+'/'+nextLayer.replace("/", '-').replace(":", '_')+'.onnx'    #use the output layer name as ModelName.onnx 
    input_names = [layer]
    output_names = [nextLayer]
    try:
      onnx.utils.extract_model(onnx_file, output_path, input_names, output_names)
    except Exception as e:
      print(e)

  #Also extract the model for the last layer
  layer = listLayers[len(listLayers)-1]
  output_path = folder+'/'+layer.replace("/", '-').replace(":", '_')+'.onnx'
  input_names = [layer]
  output_names = []
  onnx_model = onnx.load(onnx_file)
  for i in range(len(onnx_model.graph.output)):
    output_names.append(onnx_model.graph.output[i].name)
  try:
    onnx.utils.extract_model(onnx_file, output_path, input_names, output_names)
  except Exception as e:
    print(e)

  print("Finished!")

def onnx_run_complete(onnx_path, split_layer, image_file, image_batch, img_size_x, img_size_y, is_grayscale, 
                      minio_bucket,  oscar_service, kube_namespace, platform, exec_provider, device_type, xml_file = None):
  '''
  Run a complete cycle of inference, meaning running the first half of the model locally, getting the results, load them on the cloud(minio) and 
  then schedule the second part of the model on the cloud(OSCAR) and getting the results. The OSCAR job will be closely monitored and the execution
  timings will be returned.

  :param onnx_path: the path to the collection of models were to find the correct one to use for the inference
  :param split_layer: the name of the DNN layer where the split has been performed on the model choosen
  :param image_file: the path to the image if using a single image
  :param image_batch: the path to the folder containing the batch of images if using a batch
  :param img_size_x: the horrizontal size of the images
  :param img_size_y: the vertical size of the images
  :param is_grayscale: true if the image is grayscale, false otherwise
  :param minio_bucket: MinIO Bucket where the input and output files used by OSCAR must be placed
  :param oscar_service: the name of the OSCAR Service to use when we create a new job to run the second inference
  :param kube_namespace: the namespace in Kubernetes used for the OSCAR Service
  :param platform: the platform where the script is executed, in order to use the right client for MinIO, OSCAR and Kubernetes
  :param exec_provider: the Execution Provider used at inference (CPU (default) | GPU | OpenVINO | TensorRT | ACL)
  :param device: specifies the device type such as 'CPU_FP32', 'GPU_FP32', 'GPU_FP16', etc..
  :return: the 1° inference execution time, the 2° inference execution time, the 2° inference OSCAR execution time and the 2nd inference Kubernetes pod execution time
  '''
  #Default Argument Values
  if minio_bucket == None: minio_bucket = "onnx-test-mobilenet"  
  if oscar_service == None: oscar_service = "onnx-test-mobilenet"
  if kube_namespace == None: kube_namespace = "oscar-svc"
  if is_grayscale == None: is_grayscale = False
  if platform == None: platform = "AMD64"
  import time

  global dictTensors
  global sshHost
  global sshPsk

  #Default Clients
  MINIO_CLI = "./mc"
  OSCAR_CLI = "./oscar-cli"
  KUBECTL = "kubectl"

  #Platform Specific Clients for MinIO, OSCAR and Kubernetes
  if platform == "AMD64":
    MINIO_CLI = "./mc"
    OSCAR_CLI = "./oscar-cli"
    KUBECTL = "kubectl"
  elif platform == "ARM64":
    MINIO_CLI = "./mc-arm64"
    OSCAR_CLI = "./oscar-cli-arm64"
    #KUBECTL = "sshpass -p \"password\" ssh hostname@ip_address kubectl"
    KUBECTL = "sshpass -p \""+sshPsk+"\" ssh "+sshHost+" kubectl"

  #Iterate through the subdirectories to find the ONNX model splitted at our selected layer
  onnx_file = None
  if split_layer == "NO_SPLIT":
    #Since we skip the local execution and don't use splits, the full model is required instead of the onnx_path
    onnx_file = onnx_path
  else:
    split_layer = split_layer.replace("/", '-').replace(":", '_')
    print("Search for: " + split_layer)
    for dir in os.listdir(onnx_path):
      if dir.find("_on_") > 0:
        index = dir.index('_on_')
        d = dir[index+4:]
        #print("Check: " + d)
        if d == split_layer:
          print("Found Layer: " + d)
          onnx_file = onnx_path + "/" + dir + "/first_half.onnx"
          break
  
  # Only proceed if an onnx file is found for this layer
  if onnx_file != None:
    print("\n ###Start the 1st Inference Execution Locally\n")

    #Load the Onnx Model    --    Got to do it just to have the input tensor shape for data_processing
    model_onnx = load_onnx_model(onnx_file)

    # Process input data (image or batch of images)
    inputData = data_processing(image_file, image_batch, img_size_x, img_size_y, is_grayscale, model_onnx.graph.input[0])

    #Save the first input tensor
    #dictTensors["first"] = inputData
    #dictTensors[model_onnx.graph.input[0]] = inputData    #"first" won't be recognized, use the input of the model

    # Check if we have to SKIP the 1st Inference Execution Locally
    if split_layer == "NO_SPLIT":
      print("\n ###SKIP the 1st Inference Execution Locally, run directly the whole model on the Cloud..\n")
      print(" Create a results.txt file with the whole image instead of the tensor..")
      data = {
        "splitLayer": "NO_SPLIT",
        "fullModelFile": onnx_file,
        "execTime1": 0,   #1st Inference Execution Time
        "result": inputData,
        "tensorLenght": inputData.size,
        "tensorSaveTime": 0
      }

      #Save the first input tensor (input)
      #dictTensors["first"] = inputData
      dictTensors[model_onnx.graph.input[0].name] = inputData    #"first" won't be recognized, use the input of the model

      # Save the Tensor on a file
      with open(OUTPUT_PICKLE_FILE, 'wb') as f:
        pickle.dump(data, f)
    else:
      #Run at Inference the First part of the ONNX DNN Model (on single image OR batch)
      resData = onnx_run_first_half(onnx_file, inputData, True, exec_provider, device_type, profiling=False, xml_file=xml_file)   
      #I don't need to save the file since it's already saved in onnx_first_inference.py 

      #Save the Intermediate Tensors
      dictTensors[resData["splitLayer"]] = resData["result"]

    #Clear the Jobs in the OSCAR Cloud before launching a new Job
    os.system(OSCAR_CLI+" service logs remove "+oscar_service+" --all")
    
    #Run at Inference the Second part of the ONNX DNN Model 
    print("\n ###Start the 2nd Inference Execution on Cloud (OSCAR)\n")
    outputStrBefore = os.system(MINIO_CLI+" ls local/"+minio_bucket+"/output")
    print("Start Uploading file on MinIO..")
    startNetworkingTime = time.perf_counter()
    res = os.system(MINIO_CLI+" cp results.txt local/"+minio_bucket+"/input/results.txt")
    endNetworkingTime = time.perf_counter()
    networkingTime = endNetworkingTime-startNetworkingTime
    print("File upload done in: " + str(networkingTime))
    
    '''#Check when is the Cloud execution terminated on OSCAR
    execFinished = False
    import time
    while not execFinished:
      time.sleep(0.1)
      print(".")
      outputStrAfter = os.system(MINIO_CLI+" ls local/"+minio_bucket+"/output")
      if outputStrBefore == outputStrAfter:
        execFinished = True
    
    #Get the result from the MinIO Bucket
    os.system(MINIO_CLI+" cp local/"+minio_bucket+"/output/output.txt output.json")
    # Opening JSON file
    f = open('output.json')
    # Clean the JSON file
    jsonLines = []
    with open(r'output.json', 'r') as fp:
      # read an store all lines into list
      jsonLines = fp.readlines()
    for i in range(0, len(jsonLines)):
      if not jsonLines[0].startswith('{'): 
        jsonLines.pop(0)
    # Returns JSON object as a dictionary
    jsonStr = ''.join(jsonLines)
    data = json.loads(jsonStr)
    # Closing file
    f.close()

    print("\n ####1st Inference Execution Time: " + str(data["execTime1"]) + "sec") #1st Inference Execution Time
    print("\n ####2nd Inference Execution Time: " + str(data["execTime2"]) + "sec") #2nd Inference Execution Time'''

    ### Get the Cluster's timings via oscar-cli
    # Get OSCAR Job Name
    ret = ""
    succeeded = False
    while ret == "" or not succeeded:
      ret = os.popen(OSCAR_CLI+" service logs list "+oscar_service+"").read()
      print(">" + ret)
      try:
        if ret != "" and ret != "This service has no logs":
          resLines = ret.split("\n",2)
          jobStatus = resLines[1].split(None, 2)[1]
          print ("jobStatus: " + jobStatus)
          if jobStatus == "Succeeded":
            succeeded = True
      except Exception as e:
        print(e)
      time.sleep(1)
      print(".")
    #print(">" + ret)
    resLines = ret.split("\n",2)
    jobName = resLines[1].split(None, 1)[0]
    print("jobName: " + jobName + "\n")

    # Get the Logs of the Job run on OSCAR
    ret = os.popen(OSCAR_CLI+" service logs get "+oscar_service+" " + jobName).read()
    print(">" + ret)
    firstTime = None
    secodTime = None
    for line in ret.split("\n"):
      if line[:3].isdigit():
        time = line.split(" - ", 2)[0]+"000"
        #print("time before parsing: " + time)
        try:
          time = datetime.datetime.strptime(time, '%Y-%m-%d %H:%M:%S,%f')  #format(datetime.datetime.now())
        except:
          break
        if firstTime == None:
          firstTime = time
        else:
          secodTime = time
    execTime3 = (secodTime - firstTime).total_seconds() 

    #Check when is the Cloud execution terminated on OSCAR
    execFinished = False
    import time
    while not execFinished:
      time.sleep(0.1)
      print(".")
      outputStrAfter = os.system(MINIO_CLI+" ls local/"+minio_bucket+"/output")
      if outputStrBefore == outputStrAfter:
        execFinished = True


    #Get the result from the MinIO Bucket
    os.system(MINIO_CLI+" cp local/"+minio_bucket+"/output/output.txt output.json")
    # Opening JSON file
    f = open('output.json')
    # Clean the JSON file
    jsonLines = []
    with open(r'output.json', 'r') as fp:
      # read an store all lines into list
      jsonLines = fp.readlines()
    for i in range(0, len(jsonLines)):
      if not jsonLines[0].startswith('{'): 
        jsonLines.pop(0)
    # Returns JSON object as a dictionary
    jsonStr = ''.join(jsonLines)
    data = json.loads(jsonStr)
    # Closing file
    f.close()

    print("\n ####1st Inference Execution Time: " + str(data["execTime1"]) + "sec") #1st Inference Execution Time
    print("\n ####2nd Inference Execution Time: " + str(data["execTime2"]) + "sec") #2nd Inference Execution Time


    ### Get the Pod's timings on Kubernetes via kubectl
    # Get POD Name
    print("Find POD's Name, print the list of pods..")
    podList = os.popen(KUBECTL+" get pods --namespace="+kube_namespace).read()
    print(podList)
    podName = None
    for line in podList.split("\n"):
      if line.startswith(jobName):
        podName = line.split(" ", 2)[0]
    
    if podName != None:
      # Get the Logs of the Pod run on Kubernetes(Kind)
      podCreationTimestamp = os.popen(KUBECTL+" get pod "+podName+" --namespace="+kube_namespace+" -o jsonpath='{.metadata.creationTimestamp}'").read()
      print("podCreationTimestamp:" + podCreationTimestamp)
      podTerminationTimestamp = os.popen(KUBECTL+" get pod "+podName+" --namespace="+kube_namespace+" -o jsonpath='{.status.containerStatuses[0].state.terminated.finishedAt}'").read()
      print("podTerminationTimestamp:" + podTerminationTimestamp)

      #Get Timestamp objects and calculate the pod's execution time
      podCreationTimeObj = datetime.datetime.strptime(podCreationTimestamp, '%Y-%m-%dT%H:%M:%SZ')
      podTerminationTimeObj = datetime.datetime.strptime(podTerminationTimestamp, '%Y-%m-%dT%H:%M:%SZ')
      execTime4 = (podTerminationTimeObj - podCreationTimeObj).total_seconds() 
    else:
      execTime4 = 0

    print("\n ####1st Inference Execution Time: " + str(data["execTime1"]) + "sec")
    print("\n ####2nd Inference Execution Time: " + str(data["execTime2"]) + "sec")
    print("\n ###2nd Inf. OSCAR JOB Exec. Time: " + str(execTime3) + "sec")
    print("\n ###2nd Inf. Kubernetes POD Exec. Time: " + str(execTime4) + "sec")
    print("\n ---------------------------------------------------")
    print("\n ####Tensor Lenght: " + str(data["tensorLenght"]))
    print("\n ####1st Inf. Tensor Save Time: " + str(data["tensorSaveTime"]) + "sec")
    print("\n ####Networking Time: " + str(networkingTime))
    print("\n ####2nd Inf. Tensor Load Time: " + str(data["tensorLoadTime"]) + "sec")
    return data["execTime1"], data["execTime2"], execTime3, execTime4, data["tensorLenght"], data["tensorSaveTime"], data["tensorLoadTime"], networkingTime
  return -1,-1,-1,-1,-1,-1,-1,-1

def onnx_run_all_complete(onnx_file, onnx_path, image_file, image_batch, img_size_x, img_size_y, is_grayscale, 
                          minio_bucket, oscar_service, kube_namespace, platform, repetitions, exec_provider, device_type, xml_file = None):
  '''
  Run a complete cycle of inference for every splitted pair of models in the folder passed as argument, save the results in a CSV File and Plot the results.
  To run a complete cycle means to run the first half of the model locally, get the results, load them on the cloud(minio) and then schedule 
  the second part of the model on the cloud(OSCAR) and get the results.

  :param onnx_file: the full unsplitted ONNX file (used to gather usefull information)
  :param onnx_path: the path to the collection of models were to find the correct one to use for the inference
  :param image_file: the path to the image if using a single image
  :param image_batch: the path to the folder containing the batch of images if using a batch
  :param img_size_x: the horrizontal size of the images
  :param img_size_y: the vertical size of the images
  :param is_grayscale: true if the image is grayscale, false otherwise
  :param minio_bucket: MinIO Bucket where the input and output files used by OSCAR must be placed
  :param oscar_service: the name of the OSCAR Service to use when we create a new job to run the second inference
  :param kube_namespace: the namespace in Kubernetes used for the OSCAR Service
  :param repetition: specifies the number of repetitions to execute
  :param exec_provider: the Execution Provider used at inference (CPU (default) | GPU | OpenVINO | TensorRT | ACL)
  :param device: specifies the device type such as 'CPU_FP32', 'GPU_FP32', 'GPU_FP16', etc..
  :param platform: the platform where the script is executed, in order to use the right client for MinIO, OSCAR and Kubernetes
  '''
  #Default Argument Values
  if minio_bucket == None: minio_bucket = "onnx-test-mobilenet"  
  if oscar_service == None: oscar_service = "onnx-test-mobilenet"
  if kube_namespace == None: kube_namespace = "oscar-svc"
  if is_grayscale == None: is_grayscale = False
  if platform == None: platform = "AMD64"
  if repetitions == None: repetitions = 1

  global dictTensors

  #######################
  '''model_onnx = load_onnx_model(onnx_file)

  #Load tensor dictionary 
  with open("tensor_dict.pkl", "rb") as tf:
    tensors = pickle.load(tf)

  #Iterate through inputs of the graph and get operation types
  dictOperations = {}
  for node in model_onnx.graph.node:
    name = node.output[0]
    op_type = node.op_type
    if name in dictTensors:
      dictOperations[name] = op_type
      #print (op_type)

  #Now we split the layer in single blocks and run them individually in order to get the inference time per layer
  onnx_model_split_all_singlenode(onnx_file, tensors)'''

  #Run at inference all the single layer models generated and get the inference execution time and the Nr. of Parameters
  '''layerTime = {}
  dictNrParams = {}
  dictFLOPS = {}
  #Load tensor dictionary 
  with open("tensor_dict.pkl", "rb") as tf:
    dictTensors = pickle.load(tf)

  for layer in dictTensors:
    #Ignore the first layer since it will be the input
    if True: #layer != list(dictTensors.keys())[0]:
      file = "SingleLayerSplits/"+layer.replace("/", '-').replace(":", '_')+'.onnx'

      if os.path.exists(file):
        #Execute at inference the whole model locally AND Use profiling (for now disabled)
        try:
          #t_inf = onnx_run_first_half(file, dictTensors[layer], False, exec_provider, device_type, profiling=False)["execTime1"] 
          #t_inf = onnx_run_first_half(file, dictTensors[layer], False, CPU_EP_list, device_type, profiling=False)["execTime1"]  
          inputData = dictTensors[layer]
          resData = onnx_run_first_half(file, inputData, True, exec_provider, device_type, profiling=False, xml_file=xml_file)  
          resData = onnx_run_first_half(file, inputData, True, exec_provider, device_type, profiling=False, xml_file=xml_file) 
          resData = onnx_run_first_half(file, inputData, True, exec_provider, device_type, profiling=False, xml_file=xml_file) 
          t_inf = resData["execTime1"]  
        except Exception as e:
          print(e)
        layerTime[layer] = t_inf
        time.sleep(3.5) #it's only needed for OpenVINO, cuz NCS2 runs out of memory'''

  
  #Get the data from the first two cvs files
  '''list1 = []
  list2 = []
  list1_avg = []
  list2_avg = []
  with open(RESULTS_CSV_FILE, 'r', newline='') as csvfile1:
    reader = csv.reader(csvfile1, delimiter=",")
    for i, line in enumerate(reader):
      list1.append(line)
  with open(RESULTS_CSV_FILE2, 'r', newline='') as csvfile2:
    reader = csv.reader(csvfile2, delimiter=",")
    for i, line in enumerate(reader):
      list2.append(line)

  #Calc the sum of all the time of execution of all the singleLayer models
  singleLayerTimeSum = [0]*repetitions
  rowsPerRepetition = int(len(list2)/repetitions)

  #AVERAGE all the times
  list1_avg = list1
  list2_avg = list2
  for i in range(1, rowsPerRepetition): 
    list1_avg[i][1] = float(list1[i][1])   #1stInfTime
    list1_avg[i][2] = float(list1[i][2])   #2ndInfTime
    list1_avg[i][3] = float(list1[i][3])   #oscarJobTime
    list1_avg[i][4] = float(list1[i][4])   #kubePodTime
    list2_avg[i][1] = float(list2[i][1])   #singleLayerInfTime

    for rep in range(1, repetitions):
      list1_avg[i][1] = float(list1_avg[i][1]) + float(list1[i+rep*rowsPerRepetition][1])   #1stInfTime
      list1_avg[i][2] = float(list1_avg[i][2]) + float(list1[i+rep*rowsPerRepetition][2])   #2ndInfTime
      list1_avg[i][3] = float(list1_avg[i][3]) + float(list1[i+rep*rowsPerRepetition][3])   #oscarJobTime
      list1_avg[i][4] = float(list1_avg[i][4]) + float(list1[i+rep*rowsPerRepetition][4])   #kubePodTime
      list2_avg[i][1] = float(list2_avg[i][1]) + float(list2[i+rep*rowsPerRepetition][1])   #singleLayerInfTime
  for i in range(1, rowsPerRepetition): 
    list1_avg[i][1] = str(list1_avg[i][1]/repetitions)   #1stInfTime
    list1_avg[i][2] = str(list1_avg[i][2]/repetitions)   #2ndInfTime
    list1_avg[i][3] = str(list1_avg[i][3]/repetitions)   #oscarJobTime
    list1_avg[i][4] = str(list1_avg[i][4]/repetitions)   #kubePodTime
    list2_avg[i][1] = str(list2_avg[i][1]/repetitions)   #singleLayerInfTime

  #Unite the two tables into a fourth cvs file
  import math
  with open(AVG_RESULTS_CSV_FILE, 'w', newline='') as csvfile3:
    fieldnames = ['SplitLayer', '1stInfTime', '2ndInfTime', 'oscarJobTime', 'kubePodTime', 'tensorSaveTime', 'tensorLoadTime', 'tensorLenght', 
                  'networkingTime', 'singleLayerInfTime', 'OpType', 'NrParameters', 'FLOPS']
    cvswriter = csv.DictWriter(csvfile3, fieldnames=fieldnames)
    cvswriter.writeheader()

    for i in range(1, rowsPerRepetition): 
      cvswriter.writerow({"SplitLayer":list1[i][0], "1stInfTime":list1_avg[i][1], "2ndInfTime":list1_avg[i][2], "oscarJobTime":list1_avg[i][3], "kubePodTime":list1_avg[i][4],
                          "tensorSaveTime":list1[i][5], "tensorLoadTime":list1[i][6], "tensorLenght":list1[i][7], "networkingTime":list1[i][8], 
                          "singleLayerInfTime":list2_avg[i][1], "OpType":list2[i][2], "NrParameters":list2[i][3], "FLOPS":list2[i][4]})'''
  #####################

  #Load the Onnx Model
  model_onnx = load_onnx_model(onnx_file)

  #Open an cvs file to save the results
  with open(RESULTS_CSV_FILE, 'w', newline='') as csvfile:
    with open(RESULTS_CSV_FILE2, 'w', newline='') as csvfile2:
      # Repeat the whole cycle the specified number of times
      for rep in range(0, repetitions):
        #fieldnames = ['SplitLayer', 'Time1', 'Time2', 'Time3', 'Time4']
        fieldnames = ['SplitLayer', '1stInfTime', '2ndInfTime', 'oscarJobTime', 'kubePodTime', 
                      'tensorSaveTime', 'tensorLoadTime', 'tensorLenght', 'networkingTime']
        cvswriter = csv.DictWriter(csvfile, fieldnames=fieldnames)
        cvswriter.writeheader()

        # Process input data (image or batch of images)
        inputData = data_processing(image_file, image_batch, img_size_x, img_size_y, is_grayscale, model_onnx.graph.input[0])
        batchSize = inputData[0]

        #Execute at inference the whole model locally AND Use profiling (for now disabled)
        t_1st_inf = onnx_run_first_half(onnx_file, inputData, True, exec_provider, device_type, profiling=False, xml_file=xml_file)["execTime1"]  
        print("Finished inference of the whole layer locally..")
        #cvswriter.writerow({'SplitLayer':"NO_SPLIT", "Time1":t_1st_inf, "Time2":0, "Time3":0, "Time4":0})
        cvswriter.writerow({"SplitLayer":"NO_SPLIT", "1stInfTime":t_1st_inf, "2ndInfTime":0, "oscarJobTime":0, "kubePodTime":0, 
                            "tensorSaveTime":0, "tensorLoadTime":0, "tensorLenght":0, "networkingTime":0})
        print("Saved results..")  

        #Execute at inference the whole model on the Cloud (OSCAR)
        try:
          (t_1st_inf, t_2nd_inf, t_oscar_job, t_kube_pod, 
            tensor_lenght, t_tensor_save, t_tensor_load, t_networking) = onnx_run_complete(onnx_file,  #it should be onnx_path, but since we skip the local execution and don't use splits, we pass the full model
                                                                                          "NO_SPLIT", 
                                                                                          image_file, 
                                                                                          image_batch, 
                                                                                          img_size_x, 
                                                                                          img_size_y, 
                                                                                          is_grayscale,
                                                                                          minio_bucket,
                                                                                          oscar_service,
                                                                                          kube_namespace,
                                                                                          platform,
                                                                                          exec_provider,
                                                                                          device_type,
                                                                                          xml_file=xml_file) 
        except Exception as e:
          print("Error on executin RUN Complete cycle: " + e)  

        print("Finished inference of the whole layer on the Cloud (OSCAR)..")
        #cvswriter.writerow({'SplitLayer':"NO_SPLIT", "Time1":0, "Time2":t_2nd_inf, "Time3":t_oscar_job, "Time4":t_kube_pod})
        cvswriter.writerow({"SplitLayer":"NO_SPLIT", "1stInfTime":0, "2ndInfTime":t_2nd_inf, "oscarJobTime":t_oscar_job, "kubePodTime":t_kube_pod,
                            "tensorSaveTime":t_tensor_save, "tensorLoadTime":t_tensor_load, "tensorLenght":tensor_lenght, "networkingTime":t_networking})
        print("Saved results..")  

        #Make a split for every layer in the model
        ln = 0
        for layer in enumerate_model_node_outputs(model_onnx):
          #Ignore the first and the last layer
          if layer != list(enumerate_model_node_outputs(model_onnx))[0] and layer != list(enumerate_model_node_outputs(model_onnx))[-1]:
          #if layer != list(enumerate_model_node_outputs(model_onnx))[-1]:
            splitLayer = layer.replace("/", '-').replace(":", '_')

            #TODO:delete this when mobilenet_splittedmodels is updated on OSCAR
            if splitLayer == "sequential-mobilenetv2_1.00_160-Conv1-Conv2D__7426_0":
              splitLayer = "sequential-mobilenetv2_1.00_160-Conv1-Conv2D__6_0"

            print("Splitting at layer: " + splitLayer)

            # Make a complete Inference Run of the whole model by splitting at this particular layer
            print("Run..")
            try:
              (t_1st_inf, t_2nd_inf, t_oscar_job, t_kube_pod, 
                tensor_lenght, t_tensor_save, t_tensor_load, t_networking) = onnx_run_complete(onnx_path, 
                                                                                              splitLayer, 
                                                                                              image_file, 
                                                                                              image_batch, 
                                                                                              img_size_x, 
                                                                                              img_size_y, 
                                                                                              is_grayscale,
                                                                                              minio_bucket,
                                                                                              oscar_service,
                                                                                              kube_namespace,
                                                                                              platform,
                                                                                              exec_provider,
                                                                                              device_type,
                                                                                              xml_file=xml_file)
            except Exception as e:
              print("Error on executin RUN Complete cycle: " + e) 

            # t_1st_inf = 1st Inference Execution Time
            # t_2nd_inf = 2nd Inference Execution Time
            # t_oscar_job = 2nd Inf. OSCAR JOB Exec. Time
            # t_kube_pod = 2nd Inf. Kubernetes POD Exec. Time
            # tensor_lenght = 1st Inf. Tensor Lenght:
            # t_tensor_save = 1st Inf. Tensor Save Time
            # t_tensor_load = 2nd Inf. Tensor Load Time
            if t_1st_inf != -1 and t_2nd_inf != -1 and t_oscar_job != -1 and t_kube_pod != -1:
              print("Finished inference after splitting at layer: " + splitLayer)
              #cvswriter.writerow({'SplitLayer':splitLayer, "Time1":t_1st_inf, "Time2":t_2nd_inf, "Time3":t_oscar_job, "Time4":t_kube_pod})
              cvswriter.writerow({"SplitLayer":splitLayer, "1stInfTime":t_1st_inf, "2ndInfTime":t_2nd_inf, "oscarJobTime":t_oscar_job, "kubePodTime":t_kube_pod,
                                  "tensorSaveTime":t_tensor_save, "tensorLoadTime":t_tensor_load, "tensorLenght":tensor_lenght, "networkingTime":t_networking})
              print("Saved results..")

        #Save tensor dictionary 
        with open("tensor_dict.pkl", "wb") as tf:
          pickle.dump(dictTensors,tf)

        #Load tensor dictionary 
        with open("tensor_dict.pkl", "rb") as tf:
          tensors = pickle.load(tf)

        #Iterate through inputs of the graph and get operation types
        dictOperations = {}
        for node in model_onnx.graph.node:
          name = node.output[0]
          op_type = node.op_type
          if name in dictTensors:
            dictOperations[name] = op_type
            #print (op_type)

        #Now we split the layer in single blocks and run them individually in order to get the inference time per layer
        onnx_model_split_all_singlenode(onnx_file, tensors)

        #Run at inference all the single layer models generated and get the inference execution time and the Nr. of Parameters
        layerTime = {}
        dictNrParams = {}
        dictFLOPS = {}
        for layer in dictTensors:
          #Ignore the first layer since it will be the input
          if True: #layer != list(dictTensors.keys())[0]:
            file = "SingleLayerSplits/"+layer.replace("/", '-').replace(":", '_')+'.onnx'

            if os.path.exists(file):
              #Get the Nr. of Parameters
              single_layer_model_onnx = load_onnx_model(file)
              params = calculate_params(single_layer_model_onnx)
              dictNrParams[layer] = params
              #print('Number of params:', params)

              #For every model calc FLOPS
              '''onnx_json = convert(
                onnx_graph=single_layer_model_onnx.graph,
              )'''
              onnx_json = convert(
                input_onnx_file_path=file,
                output_json_path="test.json",
                json_indent=2,
              )
              dictNodeFLOPS = calc_flops(onnx_json, batchSize)

              #Sum all the FLOPS of the nodes inside the Single Layer Model
              dictFLOPS[layer] = 0
              for node in dictNodeFLOPS:
                dictFLOPS[layer] = dictFLOPS[layer] + dictNodeFLOPS[node]
              print(file + " FLOPS: " + str(dictFLOPS[layer]))

              #Execute at inference the whole model locally AND Use profiling (for now disabled)
              try:
                #t_inf = onnx_run_first_half(file, dictTensors[layer], False, exec_provider, device_type, profiling=False)["execTime1"] 
                #t_inf = onnx_run_first_half(file, dictTensors[layer], False, CPU_EP_list, device_type, profiling=False)["execTime1"]  
                inputData = dictTensors[layer]
                #first inference is just to get the model loaded into ram
                resData = onnx_run_first_half(file, inputData, True, exec_provider, device_type, profiling=False, xml_file=xml_file)  
                #sencond inference with the same model is faster (mimicing a case whre we don't have to load the model in ram)
                resData = onnx_run_first_half(file, inputData, True, exec_provider, device_type, profiling=False, xml_file=xml_file, ignore_onnx_load_time = True)  
                t_inf = resData["execTime1"] 
              except Exception as e:
                print(e)
              layerTime[layer] = t_inf
              time.sleep(3.5) #it's only needed for OpenVINO, cuz NCS2 runs out of memory

        #Save the inf times in a different csv file     
        fieldnames2 = ['SplitLayer', 'singleLayerInfTime', 'OpType', 'NrParameters', 'FLOPS']
        cvswriter2 = csv.DictWriter(csvfile2, fieldnames=fieldnames2)
        cvswriter2.writeheader()
        cvswriter2.writerow({"SplitLayer":"NO_SPLIT", "singleLayerInfTime":0})
        cvswriter2.writerow({"SplitLayer":"NO_SPLIT", "singleLayerInfTime":0})

        #dictOperations was saved by using the name of the output node for each model, while
        #layerTime, dictNrParams, dictFLOPS were saved by using the name of the input node for each model
        index = 0
        for layer in layerTime:
          #Ignore the last layer name since layerTime was saved by using the name of the input node
          if layer != list(layerTime.keys())[len(list(layerTime.keys()))-1]:
            #Get next layer
            nextLayer = list(layerTime.keys())[index+1]
            try:
              cvswriter2.writerow({"SplitLayer":nextLayer.replace("/", '-').replace(":", '_'), "singleLayerInfTime":layerTime[layer], 
                                  "OpType":dictOperations[nextLayer], "NrParameters":dictNrParams[layer], 'FLOPS':dictFLOPS[layer]})
            except Exception as e:
              print(e)
            index = index + 1

  #Get the data from the first two cvs files
  list1 = []
  list2 = []
  with open(RESULTS_CSV_FILE, 'r', newline='') as csvfile1:
    reader = csv.reader(csvfile1, delimiter=",")
    for i, line in enumerate(reader):
      list1.append(line)
  with open(RESULTS_CSV_FILE2, 'r', newline='') as csvfile2:
    reader = csv.reader(csvfile2, delimiter=",")
    for i, line in enumerate(reader):
      list2.append(line)

  #Calc the sum of all the time of execution of all the singleLayer models
  singleLayerTimeSum = [0]*repetitions
  rowsPerRepetition = int(len(list2)/repetitions)
  for rep in range(0, repetitions):
    startRow = int(rep*rowsPerRepetition + 1)
    for i in range(startRow,startRow+rowsPerRepetition-1): 
      singleLayerTimeSum[rep] = singleLayerTimeSum[rep] + float(list2[i][1])
    print("Sum of the Single Layer Models time: " + str(singleLayerTimeSum[rep]) + " | rep: " + str(rep))

  #AVERAGE all the times
  list1_avg = list1
  list2_avg = list2
  for i in range(1, rowsPerRepetition): 
    list1_avg[i][1] = float(list1[i][1])   #1stInfTime
    list1_avg[i][2] = float(list1[i][2])   #2ndInfTime
    list1_avg[i][3] = float(list1[i][3])   #oscarJobTime
    list1_avg[i][4] = float(list1[i][4])   #kubePodTime
    list2_avg[i][1] = float(list2[i][1])   #singleLayerInfTime

    for rep in range(1, repetitions):
      list1_avg[i][1] = float(list1_avg[i][1]) + float(list1[i+rep*rowsPerRepetition][1])   #1stInfTime
      list1_avg[i][2] = float(list1_avg[i][2]) + float(list1[i+rep*rowsPerRepetition][2])   #2ndInfTime
      list1_avg[i][3] = float(list1_avg[i][3]) + float(list1[i+rep*rowsPerRepetition][3])   #oscarJobTime
      list1_avg[i][4] = float(list1_avg[i][4]) + float(list1[i+rep*rowsPerRepetition][4])   #kubePodTime
      list2_avg[i][1] = float(list2_avg[i][1]) + float(list2[i+rep*rowsPerRepetition][1])   #singleLayerInfTime
  for i in range(1, rowsPerRepetition): 
    list1_avg[i][1] = str(list1_avg[i][1]/repetitions)   #1stInfTime
    list1_avg[i][2] = str(list1_avg[i][2]/repetitions)   #2ndInfTime
    list1_avg[i][3] = str(list1_avg[i][3]/repetitions)   #oscarJobTime
    list1_avg[i][4] = str(list1_avg[i][4]/repetitions)   #kubePodTime
    list2_avg[i][1] = str(list2_avg[i][1]/repetitions)   #singleLayerInfTime

  #Unite the two tables into a third cvs file
  import math
  with open(FINAL_RESULTS_CSV_FILE, 'w', newline='') as csvfile3:
    fieldnames = ['SplitLayer', '1stInfTime', '2ndInfTime', 'oscarJobTime', 'kubePodTime', 'tensorSaveTime', 'tensorLoadTime', 'tensorLenght', 
                  'networkingTime', 'singleLayerInfTime', 'OpType', 'NrParameters', 'FLOPS', 'SingleLayerSum-Splitted']
    cvswriter = csv.DictWriter(csvfile3, fieldnames=fieldnames)
    cvswriter.writeheader()

    for i in range(1,len(list1)): 
      if i % rowsPerRepetition == 0:
        cvswriter.writeheader()
      else:
        cvswriter.writerow({"SplitLayer":list1[i][0], "1stInfTime":list1[i][1], "2ndInfTime":list1[i][2], "oscarJobTime":list1[i][3], "kubePodTime":list1[i][4],
                          "tensorSaveTime":list1[i][5], "tensorLoadTime":list1[i][6], "tensorLenght":list1[i][7], "networkingTime":list1[i][8], 
                          "singleLayerInfTime":list2[i][1], "OpType":list2[i][2], "NrParameters":list2[i][3], "FLOPS":list2[i][4],
                          "SingleLayerSum-Splitted": str(singleLayerTimeSum[math.floor(i/rowsPerRepetition)] - (float(list1[i][1]) + float(list1[i][2])))})

  #Unite the two tables into a fourth cvs file, averaging the time measurements
  import math
  with open(AVG_RESULTS_CSV_FILE, 'w', newline='') as csvfile3:
    fieldnames = ['SplitLayer', '1stInfTime', '2ndInfTime', 'oscarJobTime', 'kubePodTime', 'tensorSaveTime', 'tensorLoadTime', 'tensorLenght', 
                  'networkingTime', 'singleLayerInfTime', 'OpType', 'NrParameters', 'FLOPS']
    cvswriter = csv.DictWriter(csvfile3, fieldnames=fieldnames)
    cvswriter.writeheader()

    for i in range(1, rowsPerRepetition): 
      cvswriter.writerow({"SplitLayer":list1[i][0], "1stInfTime":list1_avg[i][1], "2ndInfTime":list1_avg[i][2], "oscarJobTime":list1_avg[i][3], "kubePodTime":list1_avg[i][4],
                          "tensorSaveTime":list1[i][5], "tensorLoadTime":list1[i][6], "tensorLenght":list1[i][7], "networkingTime":list1[i][8], 
                          "singleLayerInfTime":list2_avg[i][1], "OpType":list2[i][2], "NrParameters":list2[i][3], "FLOPS":list2[i][4]})

  print("Plotting the results..")
  plot_results(AVG_RESULTS_CSV_FILE)

def calc_flops(onnx_json, batchSize):
  '''
  Calculate the FLOPS of a given onnx model. It expects in input the JSON version of the onnx's graph.

  :param onnx_json: the JSON version of the onnx's graph
  :returns: a dictionay with the flops for every node in the onnx model
  '''
  dictNodeFLOPS = {}
  #Iterate all the nodes of the Single Layer Model
  for node in onnx_json['graph']['node']:
    node_name = node['name']
    node_inputs = node['input']     #there might be more than one input
    node_output = node['output'][0]
    node_op_type = node['opType']

    #Calculate FLOPS differently based on the OperationType
    if (node_op_type == "Clip" or 
        node_op_type == "Relu" or 
        node_op_type == "LeakyRelu" or
        node_op_type == "Sigmoid" or
        node_op_type == "Tanh" or
        node_op_type == "BatchNormalization"):
      #FLOPS = 3+Cout (for forward pass)

      #Get Cout,Hout,Wout dimensions - ONNX handle internal tensors in Channel Firstformat, which is a (N,C,H,W)
      for info in onnx_json['graph']['valueInfo']:
        #Get the valueInfo instance that corresponds with the current node
        if info['name'] == node_output:
          Cout = int(info['type']['tensorType']['shape']['dim'][1]['dimValue'])

          #Calc FLOPS
          dictNodeFLOPS[node_name] = 3*Cout
          break
    elif node_op_type == "Conv":
      #FLOPS = Hf*Wf*Cin*Cout (for forward pass)
      Hf,Wf,Cin,Cout = 1,1,1,1#default

      #Get KernelShape (here we can also get the pads, strides, ecc)
      for attr in node['attribute']:
        if attr['name'] == 'kernel_shape':
          Hf = int(attr['ints'][0])
          Wf = int(attr['ints'][1])
          break

      #Get Cin,Hin,Win dimensions - ONNX handle internal tensors in Channel Firstformat, which is a (N,C,H,W)
      for info in onnx_json['graph']['valueInfo']:
        #Check for each node input if it corresponds with a valueInfo instance
        #That way we are basically searching for the output info of the node that is at the input of the current node
        for input in node_inputs:
          if info['name'] == input:
            Cin = int(info['type']['tensorType']['shape']['dim'][1]['dimValue'])
            break

      #Get Cout,Hout,Wout dimensions - ONNX handle internal tensors in Channel Firstformat, which is a (N,C,H,W)
      for info in onnx_json['graph']['valueInfo']:
        #Get the valueInfo instance that corresponds with the current node
        if info['name'] == node_output:
          Cout = int(info['type']['tensorType']['shape']['dim'][1]['dimValue'])
          break

      #Calc FLOPS
      dictNodeFLOPS[node_name] = Hf*Wf*Cin*Cout
    elif (node_op_type == "MaxPool" or 
          node_op_type == "LpPool" or 
          node_op_type == "AveragePool" or
          node_op_type == "GlobalMaxPool" or 
          node_op_type == "GlobalAveragePool"):
      #FLOPS = Hf*Wf*Cout (for forward pass)
      Hf,Wf,Cout = 1,1,1#default

      #Get KernelShape (here we can also get the pads, strides, ecc)
      if 'attribute' in node:
        for attr in node['attribute']:
          if attr['name'] == 'kernel_shape':
            Hf = int(attr['ints'][0])
            Wf = int(attr['ints'][1])
            break
      else:
        #No attribute in node, we get the kernel dimentions from the previous node
        for info in onnx_json['graph']['valueInfo']:
          #Check for each node input if it corresponds with a valueInfo instance
          #That way we are basically searching for the output info of the node that is at the input of the current node
          for input in node_inputs:
            if info['name'] == input:
              Hf = int(info['type']['tensorType']['shape']['dim'][2]['dimValue']) #only in this case Hf=Hin
              Wf = int(info['type']['tensorType']['shape']['dim'][3]['dimValue']) #only in this case Wf=Win
              break

      #Get Cout,Hout,Wout dimensions - ONNX handle internal tensors in Channel Firstformat, which is a (N,C,H,W)
      for info in onnx_json['graph']['valueInfo']:
        #Get the valueInfo instance that corresponds with the current node
        if info['name'] == node_output:
          Cout = int(info['type']['tensorType']['shape']['dim'][1]['dimValue'])
          break

      #Calc FLOPS
      dictNodeFLOPS[node_name] = Hf*Wf*Cout
    elif (node_op_type == "BatchNormalization" or 
          node_op_type == "LpNormalization"):
      #FLOPS = 5*Cout + Cn - 2 (for forward pass)
      Cout, Cn = 1,1  #default

      #Get Cout,Hout,Wout dimensions - ONNX handle internal tensors in Channel Firstformat, which is a (N,C,H,W)
      for info in onnx_json['graph']['valueInfo']:
        #Get the valueInfo instance that corresponds with the current node
        if info['name'] == node_output:
          Cout = int(info['type']['tensorType']['shape']['dim'][1]['dimValue'])
          if 'dimValue' in info['type']['tensorType']['shape']['dim'][0]:
            Cn = info['type']['tensorType']['shape']['dim'][0]['dimValue']    #TODO: try dimValue first and if it's not working then dimParam
          else:
            Cn = info['type']['tensorType']['shape']['dim'][0]['dimParam']

          if Cn.startswith("unk_"):
            Cn = batchSize    #Use the dimension of the batch used to run the RUN ALL comand instead of 1
          else:
            Cn = int(Cn)
          break

      #Calc FLOPS
      dictNodeFLOPS[node_name] = 5*Cout + Cn - 2
    elif (node_op_type == "SoftmaxCrossEntropyLoss" or 
          node_op_type == "NegativeLogLikelihoodLoss"):
      #FLOPS = 4*Cout - 1 (for forward pass)
      Cout, Cn = 1,1,1  #default

      #Get Cout,Hout,Wout dimensions - ONNX handle internal tensors in Channel Firstformat, which is a (N,C,H,W)
      for info in onnx_json['graph']['valueInfo']:
        #Get the valueInfo instance that corresponds with the current node
        if info['name'] == node_output:
          Cout = int(info['type']['tensorType']['shape']['dim'][1]['dimValue'])
          break

      #Calc FLOPS
      dictNodeFLOPS[node_name] = 4*Cout - 1
    #MatMul
    elif (node_op_type == "MatMul"): #or node_op_type == "FC"
      #FLOPS = Hin*Win*Cin*Cout (for forward pass)
      Hin,Win,Cin,Cout = 1,1,1,1#default

      #Get Cin,Hin,Win dimensions - ONNX handle internal tensors in Channel Firstformat, which is a (N,C,H,W)
      for info in onnx_json['graph']['valueInfo']:
        #Check for each node input if it corresponds with a valueInfo instance
        #That way we are basically searching for the output info of the node that is at the input of the current node
        for input in node_inputs:
          if info['name'] == input:
            Cin = int(info['type']['tensorType']['shape']['dim'][1]['dimValue'])
            if len(info['type']['tensorType']['shape']['dim']) == 4:
              Hin = int(info['type']['tensorType']['shape']['dim'][2]['dimValue'])
              Win = int(info['type']['tensorType']['shape']['dim'][3]['dimValue'])
            break

      #Get Cout,Hout,Wout dimensions - ONNX handle internal tensors in Channel Firstformat, which is a (N,C,H,W)
      for info in onnx_json['graph']['valueInfo']:
        #Get the valueInfo instance that corresponds with the current node
        if info['name'] == node_output:
          Cout = int(info['type']['tensorType']['shape']['dim'][1]['dimValue'])
          break

      #Calc FLOPS
      dictNodeFLOPS[node_name] = Hin*Win*Cin*Cout
    #Add          
    elif (node_op_type == "Add" or 
          node_op_type == "Mul" or
          node_op_type == "Div" or
          node_op_type == "Sub"):
      #FLOPS = Hout*Wout*Cout (for forward pass)
      Hout,Wout,Cout = 1,1,1#default

      #Get Cout,Hout,Wout dimensions - ONNX handle internal tensors in Channel Firstformat, which is a (N,C,H,W)
      for info in onnx_json['graph']['valueInfo']:
        #Get the valueInfo instance that corresponds with the current node
        if info['name'] == node_output:
          Cout = int(info['type']['tensorType']['shape']['dim'][1]['dimValue'])
          Hout = int(info['type']['tensorType']['shape']['dim'][2]['dimValue'])
          Wout = int(info['type']['tensorType']['shape']['dim'][3]['dimValue'])
          break

      #Calc FLOPS
      dictNodeFLOPS[node_name] = Hout*Wout*Cout
    else:
      print("WARNING! This type of Opeartion hasn't been recognized by the FLOPS calcultion algorithm! Please add support for: " + node_op_type)
      jsonFile = open(node_op_type+".json", "w")
      jsonString = json.dumps(onnx_json)
      jsonFile.write(jsonString)
      jsonFile.close()

  return dictNodeFLOPS

def onnx_import_data(image_file, image_batch, img_size_x, img_size_y, is_grayscale = False):
  '''
  Import Data (Images) - Imports the Image or Image Batch, turns it into an np array and saves ot
  on a pickle file that can be later used as input by onnx_first_inference.py

  :param image_file: the path to the image if using a single image
  :param image_batch: the path to the folder containing the batch of images if using a batch
  :param img_size_x: the horrizontal size of the images
  :param img_size_y: the vertical size of the images
  :param is_grayscale: true if the image is grayscale, false otherwise
  '''
  array = data_processing(image_file, image_batch, img_size_x, img_size_y, is_grayscale)
  data = {
    "inputData": array,
  }
  
  # Save the array on a pickle file
  with open(INPUT_PICKLE_FILE, 'wb') as f:
    pickle.dump(data, f)

def data_processing(image_file, image_batch, img_size_x, img_size_y, is_grayscale = False, input_tensor = None):
  '''
  Input Data Proproccessing - Imports the Image or Image Batch and turns it into an np array

  :param image_file: the path to the image if using a single image
  :param image_batch: the path to the folder containing the batch of images if using a batch
  :param img_size_x: the horrizontal size of the images
  :param img_size_y: the vertical size of the images
  :param is_grayscale: true if the image is grayscale, false otherwise
  :returns: the np array of the image or batch of images
  '''
  # Import the single Image
  if image_file != None:
    # Load an image from file
    image = load_img(image_file, target_size=(img_size_x, img_size_y), grayscale=is_grayscale)

    # convert the image pixels to a numpy array
    image = img_to_array(image)

    #Get the input's tensor shape first
    if input_tensor != None:
      input_tensor_shape = [list(input_tensor.type.tensor_type.shape.dim)[0].dim_value,
                            list(input_tensor.type.tensor_type.shape.dim)[1].dim_value,
                            list(input_tensor.type.tensor_type.shape.dim)[2].dim_value,
                            list(input_tensor.type.tensor_type.shape.dim)[3].dim_value]
    else:
      input_tensor_shape = [1,img_size_x,img_size_y,3]
    #input_tensor_shape = [1,3,img_size_x,img_size_y]    #test force another tensor shape

    #Reshape the Image based on the input's tensor shape
    if input_tensor_shape[3] == 3:
      print("Image of shape: (1,x,y,3)")

      # reshape data for the model
      image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
    elif input_tensor_shape[1] == 3:
      print("Image of shape: (1,3,y,x)")

      # reshape data for the model
      image = image.reshape((1, image.shape[2], image.shape[1], image.shape[0]))
    else:
      print("Default Image shape: (1,x,y,3)")

      # reshape data for the model
      image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))

    # prepare the image for the model
    #image = preprocess_input(image)
    input = np.array(image).astype(np.float32)  # Note the extra brackets to create 1x10
    input_img = np.array(image).astype(np.float32)  # Note the extra brackets to create 1x10
    print(image.shape) 
    return input_img   
  # Import the Batch of Images
  elif image_batch != None:
    i = 0
    list_img = []
    for img_name in os.listdir(image_batch):
      #print(img_name)
      new_img = load_img(image_batch+'/'+img_name, target_size=(img_size_x, img_size_y), grayscale=is_grayscale)
      new_img = img_to_array(new_img)
      new_img = new_img.reshape((1, new_img.shape[0], new_img.shape[1], new_img.shape[2]))
      list_img.append(new_img)
      i = i + 1

    tuple_img = tuple(list_img)
    batch_img = np.vstack(tuple_img)
    print(batch_img.shape)
    return batch_img
  else:
    return None

def plot_results(results_file):
  '''
  Plots the results of the Inference Cycle that are saved on a CSV file (only the first cycle if there are multiple repetitions)
  ResultsFile - FieldNames:'SplitLayer', '1stInfTime', '2ndInfTime', 'oscarJobTime', 'kubePodTime', 'tensorSaveTime', 'tensorLoadTime', 'tensorLenght'

  :param results_file: the path to the CSV file where the results are saved
  '''
  N = 0
  xTicks = []
  data_inf1_oscar_job = []
  data_inf1_inf2 = []
  data_inf1_kube_pod = []
  with open(results_file, 'r') as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',') 
    for row in csv_reader:
      if not N == 0:
        # Discard repetitions
        if row[0] == 'SplitLayer':  
          break
        #print(f'\t{row[0]},  {row[1]}, {row[2]}, {row[3]}, {row[4]}.')
        xTicks.append(row[0])
        t_1st_inf = np.float(row[1])
        t_2nd_inf = np.float(row[2])
        t_oscar_job = np.float(row[3])
        t_kube_pod = np.float(row[4])
        t_tensor_save = np.float(row[5])
        t_tensor_load = np.float(row[6])
        tensor_lenght = np.float(row[7])
        t_networking = np.float(row[6])
        data_inf1_oscar_job.append([t_1st_inf, t_networking, t_oscar_job])
        data_inf1_inf2.append([t_1st_inf, t_tensor_save, t_networking, t_2nd_inf, t_tensor_load])
        data_inf1_kube_pod.append([t_1st_inf, t_networking, t_kube_pod])
      N += 1

  print(data_inf1_oscar_job)
  # t_1st_inf = 1st Inference Execution Time
  # t_2nd_inf = 2nd Inference Execution Time
  # t_oscar_job = 2nd Inf. OSCAR JOB Exec. Time
  # t_kube_pod = 2nd Inf. Kubernets POD Exec. Time
  # tensor_lenght = 1st Inf. Tensor Lenght:
  # t_tensor_save = 1st Inf. Tensor Save Time
  # t_tensor_load = 2nd Inf. Tensor Load Time

  print("Plot the first graph where we consider also the cluster execution time..")
  # Dummy dataframe
  df = pd.DataFrame(data=data_inf1_oscar_job, columns=['1st Exec Time', 'Networking Time', '2nd Exec Time(OSCAR JOB)'])

  # Plot a stacked barchart
  ax = df.plot.bar(stacked=True)

  # Place the legend
  ax.legend(bbox_to_anchor=(1.1, 1.05))
  plt.xticks(ticks=range(0,N-1), labels=xTicks, rotation=90)
  #plt.ylim(0, 100)
  plt.title('Execution time by layer divided between Edge and Cloud (considering cluster execution time)')
  plt.xlabel('Layer')
  plt.ylabel('Time (sec)')
  plt.show()

  print("Plot the second graph where we consider also the kubernetes pod execution time..")
  # Dummy dataframe
  df = pd.DataFrame(data=data_inf1_kube_pod, columns=['1st Exec Time', 'Networking Time', '2nd Exec Time(Kubernetes POD)'])

  # Plot a stacked barchart
  ax = df.plot.bar(stacked=True)

  # Place the legend
  ax.legend(bbox_to_anchor=(1.1, 1.05))
  plt.xticks(ticks=range(0,N-1), labels=xTicks, rotation=90)
  #plt.ylim(0, 100)
  plt.title('Execution time by layer divided between Edge and Cloud (considering pod execution time)')
  plt.xlabel('Layer')
  plt.ylabel('Time (sec)')
  plt.show()

  print("Plot the third graph where we don't consider the cluster execution time..")
  # Dummy dataframe
  df = pd.DataFrame(data=data_inf1_inf2, columns=['1st Inf Exec Time', 'TensorSave Time', 'Networking Time', '2nd Exec Time', 'TensorLoad Time'])

  # Plot a stacked barchart
  ax = df.plot.bar(stacked=True)

  # Place the legend
  ax.legend(bbox_to_anchor=(1.1, 1.05))
  plt.xticks(ticks=range(0,N-1), labels=xTicks, rotation=90)
  #plt.ylim(0, 100)
  plt.title('Execution time by layer divided between Edge and Cloud')
  plt.xlabel('Layer')
  plt.ylabel('Time (sec)')
  plt.show()

def onnx_show_graph(onnx_file):
  '''
  Show/Print an ONNX Model's Graph with Neutron (https://github.com/lutzroeder/netron)

  :param onnx_file: the ONNX file to split (it can also be an already splitted model)
  '''
  #Graph the ONNX File with Neutron
  #res = os.popen(NEUTRON_INSTALLATION_PATH + " " + onnx_file, 'r').read()
  #pipe = os.popen(NEUTRON_INSTALLATION_PATH + " " + onnx_file, 'r', 100) 	
  res = os.system("/snap/bin/netron " + onnx_file)
  if res != 0:
    print("\n\nNeutron is NOT INSTALLED or installed in the wrong place.")
    print("Please install it in the following path: " + NEUTRON_INSTALLATION_PATH)
    print("Docs: https://github.com/lutzroeder/netron")
    print("\n\n")

if __name__ == "__main__":
    main()

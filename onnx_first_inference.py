#import onnxruntime as rt
#import sclblonnx as so
import onnxruntime
import onnx
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array
import numpy as np
import argparse, sys
import pickle
import time
import os
import collections
import json
from mlprodict.onnxrt.ops_whole.session import OnnxWholeSession
from pandas import DataFrame
#import openvino.runtime as ov
#from openvino.inference_engine import IECore, Blob, TensorDesc
from pathlib import Path
#from openvino.preprocess import PrePostProcessor, ResizeAlgorithm
#from openvino.runtime import Core, Layout, Type

#Default files for input/output
INPUT_PICKLE_FILE = 'input.txt'
OUTPUT_PICKLE_FILE = 'results.txt'
MAX_NUMBER_OF_INPUTS = 10

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

def main():
  '''
    Used to run at inference an ONNX DNN up to a specified layer. 
    The resulting tensor is written on a file, along with some additional info about the layer used.

    Arguments:
    -h, --help                  Show this help message and exit
    --onnx_file ONNX_FILE       Select the ONNX File
    --xml_file XML_FILE         Select the XML File (OpenVINO Optimized Model)
    --exec_type EXEC_TYPE       Select Execution Provider at inference: CPU (default) | GPU | OpenVINO | TensorRT | ACL
                                Requirements for GPU: https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html#requirements
    --device_type Device_TYPE   Select DeviceType
                                Options are: (Any hardware target can be assigned if you have the access to it)
                                'CPU_FP32', 'GPU_FP32', 'GPU_FP16', 'MYRIAD_FP16', 'VAD-M_FP16', 'VAD-F_FP32',
                                'HETERO:MYRIAD,CPU',  'MULTI:MYRIAD,GPU,CPU'
  '''
  parser=argparse.ArgumentParser(
    description='''
ONNX First Inference: Manages the execution of the first part of a splitted DNN Model (in ONNX format).
    ''',
    epilog='''
Examples:
X python onnx_first_inference.py --onnx_file MobileNetV2_SplittedModles/split_8_on_sequential-mobilenetv2_1.00_160-block_1_project_BN-FusedBatchNormV3_0/first_half.onnx 
                                 --image_file=images/mobilenet_misc/141340262_ca2e576490_jpg.rf.a9e7a7e679798619924bbc5cade9f806.jpg 
                                 --image_size_x=160 --image_size_y=160 --image_is_grayscale=False
X python onnx_first_inference.py --onnx_file MobileNetV2_SplittedModles/split_8_on_sequential-mobilenetv2_1.00_160-block_1_project_BN-FusedBatchNormV3_0/first_half.onnx 
                                 --image_batch images/mobilenet_batch --image_size_x=160 --image_size_y=160 --image_is_grayscale=False

> python onnx_first_inference.py --onnx_file MobileNetV2_SplittedModles/split_8_on_sequential-mobilenetv2_1.00_160-block_1_project_BN-FusedBatchNormV3_0/first_half.onnx 
    ''',
    formatter_class=argparse.RawTextHelpFormatter                                         
  )
  parser.add_argument('--onnx_file', help='Select the ONNX File')
  parser.add_argument('--xml_file', help='Select the XML File (OpenVINO Optimized Model)')
  parser.add_argument('--exec_type', help='Select Execution Provider at inference', choices=['CPU', 'GPU', 'OpenVINO', 'TensorRT', 'ACL'])
  parser.add_argument('--device_type', help='Select DeviceType: (CPU_FP32, GPU_FP32, GPU_FP16, MYRIAD_FP16, VAD-M_FP16, VAD-F_FP32, \
                                             HETERO:MYRIAD,CPU,  MULTI:MYRIAD,GPU,CPU)')
  args=parser.parse_args()

  #onnx_run_first_half_from_file(args.onnx_file)
  if args.exec_type == "ACL":
    onnx_run_first_half_from_file(args.onnx_file, ACL_EP_list, args.device_type)   #TODO: use c program
  if args.exec_type == "TensorRT":
    onnx_run_first_half_from_file(args.onnx_file, TensorRT_EP_list, args.device_type)
  elif args.exec_type == "OpenVINO":
    onnx_run_first_half_from_file(args.onnx_file, OpenVINO_EP_list, args.device_type, args.xml_file)
  elif args.exec_type == "GPU":
    onnx_run_first_half_from_file(args.onnx_file, GPU_EP_list, args.device_type)
  else:
    onnx_run_first_half_from_file(args.onnx_file, CPU_EP_list, args.device_type)

# Run at inference the First Half of the Splitted Model on the data(in array form) contained in a pickle file
def onnx_run_first_half_from_file(onnx_file, EP_list = CPU_EP_list, device = None, xml_file = None):
  '''
  Run at inference the First Half of the Splitted Model on the data(in array form) contained in a pickle file (input.txt)

  :param onnx_file: the ONNX file to use for the inference
  :param EP_list: the Execution Provider used at inference (CPU (default) | GPU | OpenVINO | TensorRT | ACL)
  :device: specifies the device type such as 'CPU_FP32', 'GPU_FP32', 'GPU_FP16', etc..
  '''
  #Get the input tensor from the pickle file
  with open(INPUT_PICKLE_FILE, 'rb') as f:
    data = pickle.load(f)
  inputData = data["inputData"]

  '''  # Single input case --> numpy array
  if not isinstance(data["inputData"], collections.Mapping):
    inputData = data["inputData"]
  # Multi-input case --> dict of numpy arrays
  else:
    inputData = {}
    inputData["input1"] = data["inputData"]
    for i in range[2, MAX_NUMBER_OF_INPUTS]:
      inputKey = "inputData"+str(i)
      if inputKey in data:
        inputData["input"+str(i)] = data[inputKey]
      else:
        break'''

  # Run at inference the First Half of the Splitted Model
  onnx_run_first_half(onnx_file, inputData, True, EP_list, device, xml_file=xml_file)

def onnx_run_first_half(onnx_file, inputData, saveOutput = False, EP_list = CPU_EP_list, device = None, profiling = False, xml_file = None, ignore_onnx_load_time = False):
  '''
  Run at inference the First Half of the Splitted Model on an input array and save the results on an output file

  :param onnx_file: the ONNX file to use for the inference
  :param inputData: the input data in array format (image or batch already proccessed), or a disct of arrays if passing multiple inputs (with keys "input1", "input2", ecc..)
  :param saveOutput: flag indicating if the function has to save the pickle file or not (it returns the results anyway)
  :param EP_list: the Execution Provider used at inference (CPU (default) | GPU | OpenVINO | TensorRT | ACL)
  :param device: specifies the device type such as 'CPU_FP32', 'GPU_FP32', 'GPU_FP16', etc..
  :param profiling: if True enables profiling, which generates a file with all the 
  :returns: the results of the inference, the execution time and split layer used
  '''
  startTime = time.perf_counter()

  #Get the input and output of the model
  onnx_model = onnx.load(onnx_file)
  model_output = onnx_model.graph.output[0].name 
  #model_num_inputs = len(onnx_model.graph.input)

  #Start counting the time after the model load if ignore_onnx_load_time is True
  if ignore_onnx_load_time:
    startTime = time.perf_counter()

  #To get the inputs we must ignore the initializers, otherwise it would seem like we have a lot of inputs in some cases
  input_all = [node.name for node in onnx_model.graph.input]
  input_initializer =  [node.name for node in onnx_model.graph.initializer]
  net_feed_input = list(set(input_all)  - set(input_initializer))
  print('Inputs: ', net_feed_input)
  model_num_inputs = len(net_feed_input)

  # Single Input ONNX File --> string
  if model_num_inputs == 1:
    #model_input = onnx_model.graph.input[0].name 
    model_input = net_feed_input[0]
  # Multi-Input ONNX File --> list of string
  else:
    model_input = net_feed_input
    #for i in range(0, model_num_inputs):
    #  model_input.append(onnx_model.graph.input[i].name)

  # Run the first model and get the intermediate results (that we will use as input for the second model) (Using sclblonnx)
  '''g = so.graph_from_file(onnx_file)
  inputs = {model_input: input_img}
  result = so.run(g,
                  inputs=inputs,
                  outputs=[model_output]
                  )'''

  # prefer CUDA Execution Provider over CPU Execution Provider
  #EP_list = ['CPUExecutionProvider']
  openVinoPreparationTime = 0

  #If the model is a file .XML(Optimized Model) use the OpenVINO Libraries otherwise, if it's a .ONNX file, use ONNXRUNTIME Library
  if xml_file != None:
    unusedInputs = {}#multiple inputs are not supported yet

    try:
      result, openVinoPreparationTime = runOnOpenVINO(xml_file, inputData)
    except Exception as e:
      print(e)
    except:
      print("Inference Failed..")

  else:  
    #Enable Profiling via ONNXRUNTIME
    so = onnxruntime.SessionOptions()
    if profiling:
      so.enable_profiling = True

    #Run the first model and get the intermediate results (that we will use as input for the second model) (Using onnxruntime)
    if device == None:
      #ort_session = onnxruntime.InferenceSession(onnx_file, so, providers=EP_list, provider_options=[{'enable_vpu_fast_compile':True}]) 
      ort_session = onnxruntime.InferenceSession(onnx_file, so, providers=EP_list) 
    elif device == "MYRIAD_FP16":
      so.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_DISABLE_ALL
      ort_session = onnxruntime.InferenceSession(onnx_file, so, providers=EP_list, provider_options=[{'device_type' : device, 'num_of_threads': 4}])
    else: 
      #ort_session = onnxruntime.InferenceSession(onnx_file, so, providers=EP_list, provider_options=[{'device_type' : device}])
      #print(onnxruntime.capi._pybind_state.get_available_openvino_device_ids())
      ort_session = onnxruntime.InferenceSession(onnx_file, so, providers=EP_list, provider_options=[{'device_type' : device, 'enable_vpu_fast_compile':True}])

    # RUN on Single Input
    unusedInputs = {}
    if model_num_inputs == 1:
      startOVTime = time.perf_counter()
      result = [[0]]

      #Two inference executions are needed for OpenVino, since the first one is really slow
      if EP_list == OpenVINO_EP_list:
        try:
          result = ort_session.run(None, {model_input: inputData})   
        except Exception as e:
          print(e)
        except:
          print("Inference Failed..")

      endOVTime = time.perf_counter()
      openVinoPreparationTime = endOVTime - startOVTime

      try:
        result = ort_session.run(None, {model_input: inputData})   
      except Exception as e:
        print(e)
      except:
        print("Inference Failed..")

    # RUN on Multiple Inputs
    else:
      #Build a dictionary where we assign/match input names to input awways
      '''inputsDict = {}
      for i in range[0, model_num_inputs-1]:
        inputsDict[model_input[i]] = inputData[model_input[i]]'''

      startOVTime = time.perf_counter()
      result = [[0]]

      #Two inference executions are needed for OpenVino, since the first one is really slow
      if EP_list == OpenVINO_EP_list:
        try:
          result = ort_session.run(None, {model_input: inputData})   
        except Exception as e:
          print(e)
        except:
          print("Inference Failed..")

      endOVTime = time.perf_counter()
      openVinoPreparationTime = endOVTime - startOVTime

      try:
        result = ort_session.run(None, inputData) 
      except Exception as e:
        print(e)
      except:
        print("Inference Failed..")

      #Also add unused inputs to the result pickle file if any
      for input in inputData:
        if input not in model_input:
          unusedInputs[input] = inputData[input]

  endTime = time.perf_counter()

  # Profiling ends
  if profiling:
    prof = ort_session.end_profiling()
    # and is collected in that file:
    print(prof)

    # what does it look like?
    with open(prof, "r") as f:
        js = json.load(f)
    print(js[:3])

    # a tool to convert it into a table and then into a csv file
    df = DataFrame(OnnxWholeSession.process_profiling(js))
    df.to_csv("inference_profiling.csv", index=False)

  #Build Return Data
  returnData = {
    "splitLayer": model_output,
    "execTime1": endTime-startTime - openVinoPreparationTime,   #1st Inference Execution Time
    "result": result[0],
    "tensorLenght": result[0].size,
    "unusedInputs": unusedInputs,
    "tensorSaveTime": 0
  }

  print("ExecTime: " + str(returnData["execTime1"])) 

  #Also save the Profiling File Name
  if profiling:
    returnData["profilingFile"] = prof

  # Save the Tensor on a file
  startSTime = time.perf_counter()
  if saveOutput:
    with open(OUTPUT_PICKLE_FILE, 'wb') as f:
      pickle.dump(returnData, f)
  
  #Get the TensorSave time and save it again in the pickle file (WARNING: this doubles the salvation time even if it will not be visible in the resultData timings)
  endSTime = time.perf_counter()
  returnData["tensorSaveTime"] = endSTime - startSTime
  if saveOutput:
    with open(OUTPUT_PICKLE_FILE, 'wb') as f:
      pickle.dump(returnData, f)
      
  return returnData

def runOnOpenVINO(xml_file, inputData): #todo:finish description
  # --------------------------- Step 1. Initialize OpenVINO Runtime Core ------------------------------------------------
  print('Creating OpenVINO Runtime Core')
  core = Core()

  # --------------------------- Step 2. Read a model --------------------------------------------------------------------
  print(f'Reading the model: {xml_file}')
  # (.xml and .bin files) or (.onnx file)
  model = core.read_model(xml_file)

  # --------------------------- Step 3. Set up input --------------------------------------------------------------------
  input_tensor = inputData

  # --------------------------- Step 4. Apply preprocessing -------------------------------------------------------------
  ppp = PrePostProcessor(model)

  _, h, w, _ = input_tensor.shape

  # 1) Set input tensor information:
  # - input() provides information about a single model input
  # - reuse precision and shape from already available `input_tensor`
  # - layout of data is 'NHWC'
  ppp.input().tensor() \
      .set_from(input_tensor) \
      .set_layout(Layout('NHWC'))  # noqa: ECE001, N400

  # 2) Adding explicit preprocessing steps:
  # - apply linear resize from tensor spatial dims to model spatial dims
  ppp.input().preprocess().resize(ResizeAlgorithm.RESIZE_LINEAR)

  # 3) Here we suppose model has 'NCHW' layout for input
  ppp.input().model().set_layout(Layout('NHWC'))

  # 4) Set output tensor information:
  # - precision of tensor is supposed to be 'f32'
  ppp.output().tensor().set_element_type(Type.f32)

  # 5) Apply preprocessing modifying the original 'model'
  model = ppp.build()

  # --------------------------- Step 5. Loading model to the device -----------------------------------------------------
  print('Loading the model to the plugin')
  startOVTime = time.perf_counter()
  compiled_model = core.compile_model(model, 'MYRIAD')
  endOVTime = time.perf_counter()
  openVinoPreparationTime = endOVTime - startOVTime

  # --------------------------- Step 6. Create infer request and do inference synchronously -----------------------------
  print('Starting inference in synchronous mode')
  results = compiled_model.infer_new_request({0: input_tensor})

  # --------------------------- Step 7. Process output ------------------------------------------------------------------
  return next(iter(results.values())), openVinoPreparationTime

if __name__ == "__main__":
    main()

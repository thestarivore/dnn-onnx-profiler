from numpy.core.numeric import True_
import numpy as np
import json
import os
import csv
import matplotlib.pyplot as plt
import pandas as pd
import pickle
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array

RESULTS_CSV_FILE = 'time_table.csv'
RESULTS_CSV_FILE2 = 'time_table_2.csv'
FINAL_RESULTS_CSV_FILE = 'time_table_final.csv'
AVG_RESULTS_CSV_FILE = 'time_table_avg.csv'
PROFILER_RESULTS_CSV_FILE = 'profiler_time_table.csv'
INPUT_PICKLE_FILE = 'input.txt'
OUTPUT_PICKLE_FILE = 'results.txt'
NEUTRON_INSTALLATION_PATH = '/snap/bin/neutron'

MODEL_SPLIT_FIRST_FILE = 'first_half.onnx'
MODEL_SPLIT_SECOND_FILE = 'second_half.onnx'

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

def calc_flops(onnx_json, batchSize):
  '''
  Calculate the FLOPS of a given onnx model. It expects in input the JSON version of the onnx's graph.

  :param onnx_json: the JSON version of the onnx's graph
  :returns: a dictionay with the flops for every node in the onnx model
  '''
  dictNodeFLOPS = {}
  #Iterate all the nodes of the Single Layer Model
  for node in onnx_json['graph']['node']:
    valid_node = True
    if 'input' in node.keys() and 'output' in node.keys():
      node_inputs = node['input']     #there might be more than one input
      node_output = node['output'][0]
    else:
      valid_node = False #skip this node as it's invalid for flops calculation (usually a constant or similar)
    node_op_type = node['opType']
    if 'name' in node.keys():
      node_name = node['name']
    else:
      node_name = node_output #in some models, nodes don't have a name..

    if not valid_node:
      dictNodeFLOPS[node_name] = 0
      continue

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
          if 'shape' in info['type']['tensorType']:
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
            if 'shape' in info['type']['tensorType']:
              Cin = int(info['type']['tensorType']['shape']['dim'][1]['dimValue'])
              break

      #Get Cout,Hout,Wout dimensions - ONNX handle internal tensors in Channel Firstformat, which is a (N,C,H,W)
      for info in onnx_json['graph']['valueInfo']:
        #Get the valueInfo instance that corresponds with the current node
        if info['name'] == node_output:
          if 'shape' in info['type']['tensorType']:
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
              if 'shape' in info['type']['tensorType']:
                Hf = int(info['type']['tensorType']['shape']['dim'][2]['dimValue']) #only in this case Hf=Hin
                Wf = int(info['type']['tensorType']['shape']['dim'][3]['dimValue']) #only in this case Wf=Win
                break

      #Get Cout,Hout,Wout dimensions - ONNX handle internal tensors in Channel Firstformat, which is a (N,C,H,W)
      for info in onnx_json['graph']['valueInfo']:
        #Get the valueInfo instance that corresponds with the current node
        if info['name'] == node_output:
          if 'shape' in info['type']['tensorType']:
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
          if 'shape' in info['type']['tensorType']:
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
          if 'shape' in info['type']['tensorType']:
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
            if 'shape' in info['type']['tensorType']:
              Cin = int(info['type']['tensorType']['shape']['dim'][1]['dimValue'])
              if len(info['type']['tensorType']['shape']['dim']) == 4:
                Hin = int(info['type']['tensorType']['shape']['dim'][2]['dimValue'])
                Win = int(info['type']['tensorType']['shape']['dim'][3]['dimValue'])
              break

      #Get Cout,Hout,Wout dimensions - ONNX handle internal tensors in Channel Firstformat, which is a (N,C,H,W)
      for info in onnx_json['graph']['valueInfo']:
        #Get the valueInfo instance that corresponds with the current node
        if info['name'] == node_output:
          if 'shape' in info['type']['tensorType']:
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
          if 'shape' in info['type']['tensorType']:
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

def onnx_get_true_inputs(onnx_model):
  '''
  Get the list of TRUE inputs of the ONNX model passed as argument. 
  The reason for this is that sometimes "onnx.load" interprets some of the static initializers 
  (such as weights and biases) as inputs, therefore showing a large list of inputs and misleading for instance
  the fuctions used for splitting.

  :param onnx_model: the already imported ONNX Model
  :returns: a list of the true inputs
  '''
  input_names = []

  # Iterate all inputs and check if they are valid
  for i in range(len(onnx_model.graph.input)):
    nodeName = onnx_model.graph.input[i].name
    # Check if input is not an initializer, if so ignore it
    if isNodeAnInitializer(onnx_model, nodeName):
      continue
    else:
      input_names.append(nodeName)
  
  return input_names

def isNodeAnInitializer(onnx_model, node):
  '''
  Check if the node passed as argument is an initializer in the network.

  :param onnx_model: the already imported ONNX Model
  :param node: node's name
  :returns: True if the node is an initializer, False otherwise
  '''
  # Check if input is not an initializer, if so ignore it
  for i in range(len(onnx_model.graph.initializer)):
    if node == onnx_model.graph.initializer[i].name:
      return True

  return False
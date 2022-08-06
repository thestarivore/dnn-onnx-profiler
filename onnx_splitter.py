from numpy.core.numeric import True_
#import sclblonnx as so
import onnx
#from skl2onnx.helpers.onnx_helper import select_model_inputs_outputs
#from skl2onnx.helpers.onnx_helper import save_onnx_model
from skl2onnx.helpers.onnx_helper import enumerate_model_node_outputs
from skl2onnx.helpers.onnx_helper import load_onnx_model
import os
import shutil
from onnx_utils import *

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
  #input_names = []
  onnx_model = onnx.load(onnx_file)
  input_names = onnx_get_true_inputs(onnx_model)
  '''for i in range(len(onnx_model.graph.input)):
    input_names.append(onnx_model.graph.input[i].name)'''
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
  full_model_onnx = load_onnx_model(onnx_file)

  #Get list of outputs
  outputsList = []
  for i in range(len(full_model_onnx.graph.output)):
    outputsList.append(full_model_onnx.graph.output[i].name)

  #Make a split for every layer in the model
  ln = 0
  for layer in enumerate_model_node_outputs(full_model_onnx):
    #Ignore the output layers and RELU Layers
    if layer not in outputsList and "relu" not in layer.lower():
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
      #input_names = []
      onnx_model = onnx.load(onnx_file)
      input_names = onnx_get_true_inputs(onnx_model)
      '''for i in range(len(onnx_model.graph.input)):
        input_names.append(onnx_model.graph.input[i].name)'''
      output_names = [layer]
      try:
        onnx.utils.extract_model(onnx_file, output_path, input_names, output_names)
      except Exception as e:
        print(e)
        shutil.rmtree(folder)
        continue

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
  input_names = input_names = onnx_get_true_inputs(onnx_model)
  '''for i in range(len(onnx_model.graph.input)):
    input_names.append(onnx_model.graph.input[i].name)'''
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
  #input_names = []
  onnx_model = onnx.load(onnx_file)
  input_names = onnx_get_true_inputs(onnx_model)
  '''for i in range(len(onnx_model.graph.input)):
    input_names.append(onnx_model.graph.input[i].name)'''
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

def onnx_model_split_all_singlenode(onnx_file, tensors):    #tensors should probably be named listOfLayers instead
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

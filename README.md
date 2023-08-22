# DNN ONNX Profiler: A tool for the performance profiling and prediction of partitioned Deep Neural Networks in Computing Continua Environments

## 1) Introduction

A Tool for splitting ONNX models and running & profiling inference divided between the Edge and Cloud. We split the model so that the first part can be executed on the Edge device, while the second part is executed on Cloud.

The tool accepts models in the form of **.onnx** files, so after designing and training your network with your preferred framework (*Tensorflow, Caffe, PyTorch, etc.*.) you must convert it into onnx. In fact the advantage of using onnx (apart from inference speed) is that it's framework independent. Some examples of Jupiter Notebooks can be found on the "*Notebooks*" folder.

*IMPORTANT*: For PyTorch, the model must be exported with **export_params=True**, otherwise profiling will fail for lack of node information. Example:

```
torch.onnx.export(net, dummy_input, model_path, export_params=True, verbose=False, input_names=['input'], output_names=['scores', 'boxes'])
```

Notes and Tutorial about how to setup OSCAR and install everything on Embedded devices can be found in the "*Documentation*" folder.

For the Cloud architecture we use [OSCAR](https://github.com/grycap/oscar), while for the edge it might be any AMD64 or ARM64 based device and any hardware accelerator (tested on: *Intel's NeuralComputeStick2, GoogleCoral, Jetson Nano, Jetson TX2, Jetson AGX Xavier*).

The profiler is a tool that allows us to estimate what is the split that achieves the best trade-of in the Edge-Cloud architecture.



## 2) Usage Summary

The tool is divided into 3 files:

1. **onnx_first_inference.py**: which manages allows to run models at inference on the Edge;
2. **onnx_second_inference.py**: which manages allows to run models at inference on the Cloud;
3. **onnx_manager.py**: which manages:
   - Data Processing of the input (onnx_first_inference and onnx_second_inference only receive  tensors as input);
   - Data Visualization: ResultsPlotting and Model Graph Visualization (via Netron);
   - Splitting of the onnx models;
   - Profiling of the performances over multiple repetitions;

The tool's usage can be read by running any script with the **--help** argument. Bellow we have it's output and the indication about the argument dependencies.

### 2.1) onnx_first_inference

**usage**: onnx_first_inference.py [-h] [--onnx_file ONNX_FILE] [--xml_file XML_FILE]
                             						   [--exec_type {CPU,GPU,OpenVINO,TensorRT,ACL}] [--device_type                            													    DEVICE_TYPE]

**ONNX First Inference**: Manages the execution of the first part of a splitted DNN Model (in ONNX format).

**optional arguments**:
  -h, --help            						Show this help message and exit
  --onnx_file ONNX_FILE			Select the ONNX File
  --xml_file XML_FILE   			   Select the XML File (OpenVINO Optimized Model)
  --exec_type {CPU,GPU,OpenVINO,TensorRT,ACL}
                        						     Select Execution Provider at inference
  --device_type DEVICE_TYPE   Select DeviceType: (CPU_FP32, GPU_FP32, GPU_FP16, MYRIAD_FP16,           	                     																						VAD-M_FP16, VAD-F_FP32, HETERO:MYRIAD,CPU,  			 																						MULTI:MYRIAD,GPU,CPU)



### 2.2) onnx_second_inference

**usage**: onnx_second_inference.py [-h] [--onnx_path ONNX_PATH] [--onnx_file ONNX_FILE]   															  [--input_file INPUT_FILE] [--save_results SAVE_RESULTS] 															  [--exec_type {CPU,GPU,OpenVINO,TensorRT,ACL}]
                                							  [--device_type DEVICE_TYPE]

**ONNX Second Inference**: Manages the execution of the second part of a splitted DNN Model (in ONNX format).  Also used for multiple splits, running the 2nd, 3rd, ecc parts. The resulting classification tensor is written on a file.

**optional arguments**:
  -h, --help            							Show this help message and exit
  --onnx_path ONNX_PATH		   Select the path were all the Splitted ONNX Models are stored
  --onnx_file ONNX_FILE				Select the ONNX File (only choose onnx_path or onnx_file)
  --input_file INPUT_FILE				Insert the file that contains the input tensor (a list) to be fed to the network
  --save_results SAVE_RESULTS    Set the salvation of the Results in a specified Pickle file (if not specified just 														  return the results)
  --exec_type {CPU,GPU,OpenVINO,TensorRT,ACL}																										 	 													Select Execution Provider at inference
  --device_type DEVICE_TYPE	   Select DeviceType: (CPU_FP32, GPU_FP32, GPU_FP16, MYRIAD_FP16, 																							VAD-M_FP16, VAD-F_FP32,  HETERO:MYRIAD, CPU,  																							MULTI:MYRIAD,GPU,CPU)

**Examples**:

> python onnx_second_inference.py --onnx_path MobileNetV2_SplittedModles --input_file results.txt

**Example on multi-split** (two splits): 

> python onnx_first_inference.py --onnx_file part1.onnx
> python onnx_second_inference.py --onnx_file part2.onnx --input_file results.txt --save_results results2.txt
> python onnx_second_inference.py --onnx_file part3.onnx --input_file results2.txt 



### 2.3) onnx_manager

**usage**: onnx_manager.py [-h]
                       [--operation {list_layers,print_model,split_model,split_model_all,multi_split_model,early_exit_split_model,data_processing,run,run_all,plot_results,quant_model,show_graph}]
                       [--onnx_file ONNX_FILE] [--xml_file XML_FILE] [--split_layer SPLIT_LAYER]
                       [--split_layers SPLIT_LAYERS [SPLIT_LAYERS ...]] [--outputs OUTPUTS [OUTPUTS ...]]
                       [--onnx_path ONNX_PATH] [--image_file IMAGE_FILE] [--image_batch IMAGE_BATCH]
                       [--image_size_x IMAGE_SIZE_X] [--image_size_y IMAGE_SIZE_Y]
                       [--image_is_grayscale IMAGE_IS_GRAYSCALE] [--results_file RESULTS_FILE]
                       [--minio_bucket MINIO_BUCKET] [--oscar_service OSCAR_SERVICE] [--kube_namespace KUBE_NAMESPACE]
                       [--quant_type {QInt8,QUInt8}] [--platform {AMD64,ARM64}]
                       [--exec_type {CPU,GPU,OpenVINO,TensorRT,ACL}] [--device_type DEVICE_TYPE] [--rep REP]
                       [--kube_ssh_host KUBE_SSH_HOST] [--kube_ssh_psk KUBE_SSH_PSK]

**ONNX Manager**: Manages operations on ONNX DNN Models such as: 

  - the execution of the complete cycle edge-cloud;
  - the creation of splitted models;
  - model layer visualizzation;
  - data processing (of images or batches); 
  - the quantization of models;
  - plotting of results;
  - showing the graph on ONNX Models
    

**optional arguments**:
  -h, --help      Show this help message and exit
  --operation {list_layers,print_model,split_model,split_model_all,multi_split_model,early_exit_split_model,data_processing,run,run_all,plot_results,quant_model,show_graph}
                        Select the operation to be performed on the ONNX Model
  --onnx_file ONNX_FILE
                        Select the ONNX File
  --xml_file XML_FILE   Select the XML File (OpenVINO Optimized Model)
  --split_layer SPLIT_LAYER
                        Select the layer where the slit must take place on the ONNX Model
  --split_layers SPLIT_LAYERS [SPLIT_LAYERS ...]
                        Select the list of layers where the slit must take place on the ONNX Model
  --outputs OUTPUTS [OUTPUTS ...]
                        Select the output and the early exits where the slit must take place on the ONNX Model (the actual split will take place above the early exit)
  --onnx_path ONNX_PATH
                        Select the path were all the Splitted ONNX Models are stored
  --image_file IMAGE_FILE
                        Select the Image File
  --image_batch IMAGE_BATCH
                        Select the Image Folder containing the Batch of images
  --image_size_x IMAGE_SIZE_X
                        Select the Image Size X
  --image_size_y IMAGE_SIZE_Y
                        Select the Image Size Y
  --image_is_grayscale IMAGE_IS_GRAYSCALE
                        Indicate if the Image is in grayscale
  --results_file RESULTS_FILE
                        Select the Results File(.csv)
  --minio_bucket MINIO_BUCKET
                        Insert the name of the MinIO Bucket
  --oscar_service OSCAR_SERVICE
                        Insert the name of the OSCAR Service
  --kube_namespace KUBE_NAMESPACE
                        Insert the namespace used in Kubernetes for the OSCAR Service
  --quant_type {QInt8,QUInt8}
                        Choose weight type used during model quantization
  --platform {AMD64,ARM64}
                        Choose the platform where RUN_ALL/RUN is executed, in order to use the right client for MinIO, OSCAR and Kubernetes
  --exec_type {CPU,GPU,OpenVINO,TensorRT,ACL}
                        Select Execution Provider at inference
  --device_type DEVICE_TYPE
                        Select DeviceType: (CPU_FP32, GPU_FP32, GPU_FP16, MYRIAD_FP16, VAD-M_FP16, VAD-F_FP32,                                              HETERO:MYRIAD,CPU,  MULTI:MYRIAD,GPU,CPU)
  --rep REP             Number of repetitions
  --kube_ssh_host KUBE_SSH_HOST
                        SSH Host to use when we need to run Kubernetes Client remotely (form: "username@ip_address")
  --kube_ssh_psk KUBE_SSH_PSK
                        SSH Host's Password to use when we need to run Kubernetes Client remotely (form: "password")

**Examples**:

> ```
> > python onnx_manager.py --operation run --split_layer sequential/dense_1/MatMul:0 
>                          --onnx_path LENET_SplittedModels/ --image_file=images/mnist_test.jpg 
>                          --image_size_x=32 --image_size_y=32 --image_is_grayscale=True
> > python onnx_manager.py --operation run_all --onnx_file lenet.onnx --onnx_path LENET_SplittedModels/ 
>                          --image_file=images/mnist_test.jpg --image_size_x=32 --image_size_y=32 --image_is_grayscale=True
> 
> > python onnx_manager.py --operation list_layers --onnx_file mobilenet_v2.onnx
> > python onnx_manager.py --operation split_model --onnx_file mobilenet_v2.onnx 
>                          --split_layer sequential/mobilenetv2_1.00_160/block_3_project_BN/FusedBatchNormV3:0
> 
> > python onnx_manager.py --operation run --split_layer sequential/mobilenetv2_1.00_160/block_5_add/add:0 
>                          --onnx_path MobileNetV2_SplittedModles 
>                          --image_file=images/mobilenet_misc/141340262_ca2e576490_jpg.rf.a9e7a7e679798619924bbc5cade9f806.jpg 
>                          --image_size_x=160 --image_size_y=160 --image_is_grayscale=False --minio_bucket=onnx-test-mobilenet 
>                          --oscar_service=onnx-test-mobilenet
> > python onnx_manager.py --operation run --split_layer sequential/mobilenetv2_1.00_160/block_5_add/add:0 
>                          --onnx_path MobileNetV2_SplittedModles --image_batch=images/mobilenet_batch 
>                          --image_size_x=160 --image_size_y=160 --image_is_grayscale=False 
>                          --minio_bucket=onnx-test-mobilenet --oscar_service=onnx-test-mobilenet
> 
> > python onnx_manager.py --operation run_all --onnx_file mobilenet_v2.onnx --onnx_path MobileNetV2_SplittedModles 
>                          --image_file=images/mobilenet_misc/141340262_ca2e576490_jpg.rf.a9e7a7e679798619924bbc5cade9f806.jpg 
>                          --image_size_x=160 --image_size_y=160 --image_is_grayscale=False 
>                          --minio_bucket=onnx-test-mobilenet --oscar_service=onnx-test-mobilenet
> > python onnx_manager.py --operation run_all --onnx_file mobilenet_v2.onnx --onnx_path MobileNetV2_SplittedModles 
>                          --image_file=images/mobilenet_batch --image_size_x=160 --image_size_y=160 --image_is_grayscale=False 
>                          --minio_bucket=onnx-test-mobilenet --oscar_service=onnx-test-mobilenet
> 
> > python onnx_manager.py --operation data_processing --image_file=images/mobilenet_misc/141340262_ca2e576490_jpg.rf.a9e7a7e679798619924bbc5cade9f806.jpg 
>                          --image_size_x=160 --image_size_y=160 
>                          --image_is_grayscale=False
> 
> ```



## 3) Debugging

If using VisualStudioCode with the Python extension, one can use all the execution templates, which can be found in **.vscode/launch.json**. Additional templates can be added.


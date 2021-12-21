import onnxruntime as rt
import sclblonnx as so
import onnx
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array
import numpy as np
import argparse, sys
import pickle

'''
  Used to run at inference an ONNX DNN up to a specified layer. 
  The resulting tensor is written on a file, along with some additional info about the layer used.

  Arguments:
  -h, --help                                  Show this help message and exit
  --onnx_file ONNX_FILE                       Select the ONNX File
  --image_file IMAGE_FILE                     Select the Image File
  --image_size_x IMAGE_SIZE_X                 Select the Image Size X
  --image_size_y IMAGE_SIZE_Y                 Select the Image Size Y
  --image_is_grayscale IMAGE_IS_GRAYSCALE     Indicate if the Image is in grayscale
'''
def main():
    parser=argparse.ArgumentParser()

    parser.add_argument('--onnx_file', help='Select the ONNX File')
    parser.add_argument('--image_file', help='Select the Image File')
    parser.add_argument('--image_size_x', help='Select the Image Size X')
    parser.add_argument('--image_size_y', help='Select the Image Size Y')
    parser.add_argument('--image_is_grayscale', help='Indicate if the Image is in grayscale')
    args=parser.parse_args()

    onnx_run_first_half(args.onnx_file, args.image_file, int(args.image_size_x), int(args.image_size_y), bool(args.image_is_grayscale))

# Run at inference the First Half of the Splitted Model
def onnx_run_first_half(onnx_file, image_file, img_size_x, img_size_y, is_grayscale = False):
  #print(onnx_file + " , " + image_file + " , " + str(img_size_x) + " , " + str(img_size_y) + " , " + str(is_grayscale))

  # load an image from file
  image = load_img(image_file, target_size=(img_size_x, img_size_y), grayscale=is_grayscale)

  # convert the image pixels to a numpy array
  image = img_to_array(image)

  # reshape data for the model
  image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))

  # prepare the image for the model
  #image = preprocess_input(image)
  print(image.shape)

  #Get the input and output of the model
  onnx_model = onnx.load(onnx_file)
  model_input = onnx_model.graph.input[0].name 
  model_output = onnx_model.graph.output[0].name 

  # Run the first model and get the intermediate results (that we will use as input for the second model)
  g = so.graph_from_file(onnx_file)

  input_img = np.array(image).astype(np.float32)  # Note the extra brackets to create 1x10
  inputs = {model_input: input_img}
  result = so.run(g,
                  inputs=inputs,
                  outputs=[model_output]
                  )
  #print(result[0])
  returnData = {
    "splitLayer": model_output,
    "result": result[0],
  }
  print(returnData)
  
  # Save the Tensor on a file
  with open("results.txt", 'wb') as f:
    pickle.dump(returnData, f)


if __name__ == "__main__":
    main()

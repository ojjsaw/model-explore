import argparse
import sys
from openvino.runtime import Core
import cv2
import numpy as np
import time



def build_argparser():
    """ Build argument parser.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model', required=True, type=str,
                        help="Required. Path to an .xml file with a trained model")
    parser.add_argument('-i', '--input', type=str,
                        help='Required. Path to the dataset file in .pickle format.')
    parser.add_argument("-o", "--output", type = str, required=True,
                        help = "Path to output directory")
    parser.add_argument("-d", "--device", type=str,
        help="specify a device to infer on (the list of available devices is shown below). Default is CPU")
  
    return parser

def main(args):
    input_mean=0
    input_std=255
    ie = Core()
    model = ie.read_model(args.model)
    compiled_model = ie.compile_model(model=model, device_name=args.device)

    input_layer = compiled_model.input(0)
    output_layer = compiled_model.output(0)

    # image_filename = "./inference_images/401.jpg"
    image = cv2.imread(args.input)
    N, H, W, C = input_layer.shape
    resized_image = cv2.resize(src=image, dsize=(W, H))
    resized_image = np.asarray(resized_image, np.float32)
    normalized_image = (resized_image - input_mean) / input_std
    input_data = np.expand_dims(resized_image, 0)

    # warmup iterations
    for _ in range(5):
        results = compiled_model([input_data])[output_layer]

    start = time.time()
    results = compiled_model([input_data])[output_layer]
    elapsed = time.time() - start
    fps = 1 / elapsed
    print(f'FPS: %.2f ' % fps)
    print(f'Inference time: %.2f ms' % (elapsed * 1000))

    index = np.argmax(results)
    if index == 0:
        print("Found Cat")
    else:
        print("Found Dog")
        
if __name__ == "__main__":
    args = build_argparser().parse_args()
    sys.exit(main(args) or 0)
    main(args)
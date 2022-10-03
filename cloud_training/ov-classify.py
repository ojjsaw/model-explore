import argparse
import sys
from openvino.runtime import Core
import cv2
import numpy as np
import time
import os
import json
import glob



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

def append_value(dict_obj, key, value):
    # Check if key exist in dict or not
    if key in dict_obj:
        # Key exist in dict.
        # Check if type of value of key is list or not
        if not isinstance(dict_obj[key], list):
            # If type is not list then make it list
            dict_obj[key] = [dict_obj[key]]
        # Append the value in list
        dict_obj[key].append(value)
    else:
        # As key is not in dict,
        # so, add key-value pair
        dict_obj[key] = value

def main(args):
    input_mean=0
    input_std=255
    ie = Core()
    model = ie.read_model(args.model)
    compiled_model = ie.compile_model(model=model, device_name=args.device)
    job_id = str(os.environ['PBS_JOBID']).split('.')[0]
    output_path = args.output
    print("This is output path: ", output_path)
    if not os.path.exists(output_path):
        try:
            os.makedirs(output_path)
        except OSError:
            print(" Failed to Create Directory %s" % output_path)
        else:
            print("Output directory %s was successfully created" % output_path)

    input_layer = compiled_model.input(0)
    output_layer = compiled_model.output(0)
    
    main_dir = args.input
    all_classes_folders = []
    images_path = []
    for all_classes in os.listdir(main_dir):
        all_classes_folders.append(main_dir+all_classes+'/')
    for x in all_classes_folders:
        for file in glob.glob(f'{x}*.jpg'):
            images_path.append(file)
            
    # prepare input images
    ts = []
    for n in images_path:
        image = cv2.imread(n)
        N, H, W, C = input_layer.shape
        resized_image = cv2.resize(src=image, dsize=(W, H))
        resized_image = np.asarray(resized_image, np.float32)
        normalized_image = (resized_image - input_mean) / input_std
        ts.append(np.expand_dims(resized_image, 0))

    label_data = {}
    with open('000000001/labels.txt') as f:
        lines = f.read().splitlines()
    for i in range(0, len(lines)):
        append_value(label_data, 'labels', lines[i])        

    # warmup iterations
    for _ in range(5):
        results = compiled_model([ts[0]])[output_layer]

    # measure inference
    total_elapsed = 0
    for i in range(len(images_path)):
        start = time.time()
        results = compiled_model([ts[i]])[output_layer]
        elapsed = time.time() - start
        total_elapsed += elapsed
        index = np.argmax(results[0])
        prediction = lines[index]
        print('Prediction for {}: {}'.format(images_path[i],lines[index]))
        print('Score for {}: {:.3g}'.format(images_path[i],max(results[0])))
        image = cv2.imread(images_path[i])
        x,y,w,h = 0,0,image.shape[0], int(image.shape[0]/8)
        cv2.rectangle(image, (x, x), (x + w, y + h), (0,0,0), -1)
        cv2.putText(image, "Prediction: {}, {}".format(prediction, max(results[0])), (x + int(w/20),y + int(h/2)), cv2.FONT_HERSHEY_SIMPLEX, (image.shape[0]/500), (255, 255, 255), 1)
        cv2.imwrite(os.path.join(output_path, f'prediction_{i}.jpg'), image)
        
    avg_elapsed = total_elapsed/len(ts)
    fps = 1 / avg_elapsed
    print(f'Average FPS: %.2f ' % fps)
    print(f'Average Inference time: %.2f ms' % (avg_elapsed * 1000))    
    with open(os.path.join(output_path, f'stats_{job_id}.txt'), 'w') as f:
        f.write('{:.3g} \n'.format(fps))
        f.write('{:.3g} \n'.format(avg_elapsed * 1000))
     
        
if __name__ == "__main__":
    args = build_argparser().parse_args()
    sys.exit(main(args) or 0)
    main(args)
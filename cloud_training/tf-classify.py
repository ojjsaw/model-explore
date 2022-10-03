import cv2
import time
import numpy as np
import tensorflow as tf
from PIL import Image
import argparse
import sys
import os
import json
import glob


def build_argparser():
    """ Build argument parser.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', type=str,
                        help='Required. Path to the dataset file in .pickle format.')
    parser.add_argument("-o", "--output", type = str, required=True,
                        help = "Path to output directory")
  
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

def preprocess_image(frame, input_height, input_width, input_mean=0, input_std=255):
    image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    resized_image = np.asarray(image.resize((input_height, input_width)), np.float32)
    return np.expand_dims(resized_image, 0)

    
def main(args):

    model = tf.saved_model.load("000000001")
    # input_img_path = "./inference_images/401.jpg"
    output_path = args.output    
    print("This is output path: ", output_path)
    if not os.path.exists(output_path):
        try:
            os.makedirs(output_path)
        except OSError:
            print(" Failed to Create Directory %s" % output_path)
        else:
            print("Output directory %s was successfully created" % output_path)
    main_dir = args.input
    all_classes_folders = []
    images_path = []
    for all_classes in os.listdir(main_dir):
        all_classes_folders.append(main_dir+all_classes+'/')
    for x in all_classes_folders:
        for file in glob.glob(f'{x}*.jpg'):
            images_path.append(file)
            
    ts = []
    for n in images_path:
        np_img = preprocess_image(cv2.imread(n), input_height=180, input_width=180)
        ts.append(tf.convert_to_tensor(np_img))

    # load labels
    with open('000000001/labels.txt') as f:
        label_data = [line.rstrip() for line in f]

   
    # warmup iterations
    for _ in range(5):
        results = model(ts[0])

# measure inference
    total_elapsed = 0
    for i in range(len(ts)):
        start = time.time()
        results = model(ts[i])
        elapsed = time.time() - start
        total_elapsed += elapsed
        index = np.argmax(results)
        print('Prediction for {}: {}'.format(images_path[i],label_data[index]))
        print('Score for {}: {:.3g}'.format(images_path[i],max(results[0])))
        
        image = cv2.imread(images_path[i])
        x,y,w,h = 0,0,image.shape[0], int(image.shape[0]/8)
        cv2.rectangle(image, (x, x), (x + w, y + h), (0,0,0), -1)
        cv2.putText(image, "Prediction: {}, {:.3g}".format(label_data[index], max(results[0])), (x + int(w/20),y + int(h/2)), cv2.FONT_HERSHEY_SIMPLEX, (image.shape[0]/500), (255, 255, 255), 1)
        cv2.imwrite(os.path.join(output_path, f'prediction_{i}.jpg'), image)
    
    avg_elapsed = total_elapsed/len(ts)
    fps = 1 / avg_elapsed
    print(f'Average FPS: %.2f ' % fps)
    print(f'Average Inference time: %.2f ms' % (avg_elapsed * 1000))    
    job_id = str(os.environ['PBS_JOBID']).split('.')[0]
    with open(os.path.join(output_path, f'stats_{job_id}.txt'), 'w') as f:
        f.write('{:.3g} \n'.format(fps))
        f.write('{:.3g} \n'.format(elapsed * 1000))
        

if __name__ == "__main__":
    args = build_argparser().parse_args()
    sys.exit(main(args) or 0)
    main(args)
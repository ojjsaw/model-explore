import cv2
import time
import numpy as np
import tensorflow as tf
from PIL import Image
import argparse
import sys
import os
import json

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

def preprocess_image(frame,
                     input_height=299,
                     input_width=299,
                     input_mean=0,
                     input_std=255):
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(image)
    resized_image = image.resize((input_height, input_width))
    resized_image = np.asarray(resized_image, np.float32)
    normalized_image = (resized_image - input_mean) / input_std
    result = np.expand_dims(normalized_image, 0)
    return result

def main(args):

    model = tf.saved_model.load("000000001")
    # input_img_path = "./inference_images/401.jpg"
    frame = cv2.imread(args.input)

    t = tf.convert_to_tensor(preprocess_image(frame, input_height=180, input_width=180))
    
    label_data = {}
    with open('model_classes.txt') as f:
        lines = f.read().splitlines()
    for i in range(0, len(lines)):
        append_value(label_data, 'labels', lines[i])

    # warmup iterations
    for _ in range(5):
        results = model(t)

    start = time.time()
    results = model(t)
    # predict = model.predict(t)    
    scores, class_ids = tf.math.top_k(results, k=len(label_data['labels']), sorted=True)
    elapsed = time.time() - start
    fps = 1 / elapsed
    print(f'FPS: %.2f fps' % fps)
    print(f'Inference time: %.2f ms' % (elapsed * 1000))
    for score, class_id in zip(scores[0], class_ids[0]):
        score = score.numpy()
        # print('Prediction: {}, {}'.format(label_data['labels'][class_id], score))
    index = np.argmax(scores)
    print('Prediction {}'.format(lines[index]))
    
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
    with open(os.path.join(output_path, f'stats_{job_id}.txt'), 'w') as f:
        f.write('{:.3g} \n'.format(fps))
        f.write('{:.3g} \n'.format(elapsed * 1000))
        f.write('{} \n'.format(lines[index]))

    
if __name__ == "__main__":
    args = build_argparser().parse_args()
    sys.exit(main(args) or 0)
    main(args)
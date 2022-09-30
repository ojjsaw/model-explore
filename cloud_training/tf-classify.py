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
        

def preprocess_image(image,
                     input_height=299,
                     input_width=299,
                     input_mean=0,
                     input_std=255):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(image)
    resized_image = image.resize((input_height, input_width))
    resized_image = np.asarray(resized_image, np.float32)
    normalized_image = (resized_image - input_mean) / input_std
    result = np.expand_dims(normalized_image, 0)
    return result

def main(args):

    model = tf.saved_model.load("000000001")
    # input_img_path = "./inference_images/401.jpg"
    
    main_dir = args.input
    all_classes_folders = []
    images_path = []
    for all_classes in os.listdir(main_dir):
        all_classes_folders.append(main_dir+all_classes+'/')
    for x in all_classes_folders:
        for file in glob.glob(f'{x}*.jpg'):
            images_path.append(file)
    # print(images_path)
    
    
    # image = cv2.imread(args.input)
    
    # prepare input images
    ts = []
    for x in INPUT_IMG_PATHS:
        np_img = preprocess_image(cv2.imread(x), input_height=180, input_width=180)
        ts.append(tf.convert_to_tensor(np_img))

    # t = tf.convert_to_tensor(preprocess_image(image, input_height=180, input_width=180))
    
    label_data = {}
    with open('000000001/labels.txt') as f:
        lines = f.read().splitlines()
    for i in range(0, len(lines)):
        append_value(label_data, 'labels', lines[i])

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
    
    
    # start = time.time()
    # results = model(t)
    # print('results: {}'.format(results))
    # scores, class_ids = tf.math.top_k(results, k=len(label_data['labels']), sorted=True)
    # elapsed = time.time() - start
    # fps = 1 / elapsed
    # print(f'FPS: %.2f fps' % fps)
    # print(f'Inference time: %.2f ms' % (elapsed * 1000))
    # for score, class_id in zip(scores[0], class_ids[0]):
    #     score = score.numpy()
        # print('Prediction: {}, {}'.format(label_data['labels'][class_id], score))
        index = np.argmax(results)
        print ('Results for {}: {}'.format(ts[i],results))
        print('Score for {}: {}'.format(ts[i],max(results[0])))
        print('Predictionfor {}: {}'.format(ts[i],lines[index]))
    
    avg_elapsed = total_elapsed/len(ts)
    fps = 1 / avg_elapsed
    print(f'FPS: %.2f ' % fps)
    print(f'Inference time: %.2f ms' % (avg_elapsed * 1000))    
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
        # f.write('{} \n'.format(lines[index]))
        
        x,y,w,h = 0,0,image.shape[0], 20
        cv2.rectangle(image, (x, x), (x + w, y + h), (0,0,0), -1)
        cv2.putText(image, "Prediction: {}, {:.3g}".format(lines[index], max(results[0])), (x + int(w/20),y + int(h/2)), cv2.FONT_HERSHEY_SIMPLEX, (image.shape[0]/500), (255, 255, 255), 1)
        cv2.imwrite(os.path.join(output_path, f'prediction_{ts[i]}.jpg'), image)

    
if __name__ == "__main__":
    args = build_argparser().parse_args()
    sys.exit(main(args) or 0)
    main(args)
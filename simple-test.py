import cv2, time, json, os, sys
import numpy as np
import tensorflow as tf
from PIL import Image

input_img_paths = ["/home/u76485/for-nihar/test_images/glacier/60.jpg",
                   "/home/u76485/for-nihar/test_images/forest/98.jpg",
                   "/home/u76485/for-nihar/test_images/buildings/96.jpg"]
#input_img_paths = ["/home/u76485/for-nihar/test_images/forest/98.jpg"]
model_path = "/home/u76485/for-nihar/000000001"
labels_info_path = "/home/u76485/for-nihar/000000001/labels.txt"
input_w = 180
input_h = 180

def preprocess_image(frame, input_height, input_width, input_mean=0, input_std=255):
    image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    resized_image = np.asarray(image.resize((input_height, input_width)), np.float32)
    return np.expand_dims(resized_image, 0)
    #normalized_image = (resized_image - input_mean) / input_std <------- model does not require mean or scale shift, does in layer
    #return np.expand_dims(normalized_image, 0)
    
# load model
model = tf.saved_model.load(model_path)

# load labels
with open(labels_info_path) as f:
    label_data = [line.rstrip() for line in f]

# prepare input images
ts = []
for x in input_img_paths:
    np_img = preprocess_image(cv2.imread(x), 
                              input_height=input_h, 
                              input_width=input_w)
    ts.append(tf.convert_to_tensor(np_img))
    
# warmup iterations
for _ in range(5):
    results = model(ts[0])

for i in range(len(label_data)):
    print(label_data[i])

# measure inference
total_elapsed = 0
for i in range(len(ts)):
    start = time.time()
    results = model(ts[i])
    elapsed = time.time() - start
    total_elapsed += elapsed
    
    print("\nInput: " + input_img_paths[i])

    scores, class_ids = tf.math.top_k(tf.nn.softmax(results), 
                                      k=len(label_data), 
                                      sorted=True)
    for score, class_id in zip(scores[0], class_ids[0]):
        score = score.numpy()
        print(label_data[class_id], score)

avg_elapsed = total_elapsed/len(ts)
print(f'\nAVG_FPS: {(1 / avg_elapsed):.2f} , AVG_LATENCY: {(avg_elapsed * 1000):.2f} ms')
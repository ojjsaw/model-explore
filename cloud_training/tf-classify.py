import cv2
import time
import numpy as np
import tensorflow as tf
from PIL import Image
import argparse
import sys

def build_argparser():
    """ Build argument parser.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', type=str,
                        help='Required. Path to the dataset file in .pickle format.')
    parser.add_argument("-o", "--output", type = str, required=True,
                        help = "Path to output directory")
  
    return parser

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

    # warmup iterations
    for _ in range(5):
        results = model(t)

    start = time.time()
    results = model(t)
    elapsed = time.time() - start
    fps = 1 / elapsed
    print(f'FPS: %.2f fps' % fps)
    print(f'Inference time: %.2f ms' % (elapsed * 1000))
    
if __name__ == "__main__":
    args = build_argparser().parse_args()
    sys.exit(main(args) or 0)
    main(args)
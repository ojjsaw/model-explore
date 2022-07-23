import cv2
import time
import numpy as np
import tensorflow as tf
from PIL import Image

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

model = tf.saved_model.load("000000001")
input_img_path = "cat1.jpg"
frame = cv2.imread(input_img_path)

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
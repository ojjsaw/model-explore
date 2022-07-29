from openvino.runtime import Core
import cv2
import numpy as np
import time

input_mean=0
input_std=255

ie = Core()
classification_model_xml = "./models/resnet_50/FP32/saved_model.xml"

model = ie.read_model(model=classification_model_xml)
compiled_model = ie.compile_model(model=model, device_name="CPU")

input_layer = compiled_model.input(0)
output_layer = compiled_model.output(0)

image_filename = "./inference_images/401.jpg"
image = cv2.imread(image_filename)
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
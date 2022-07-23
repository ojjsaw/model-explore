```shell
pip install opencv-python-headless

pip install openvino-dev[tensorflow2]
```

```shell
tar -xf model.tar.gz
```

```shell
conda create -n py38 python=3.8
conda activate py38
conda deactivate

```

```shell
mo --saved_model_dir 000000001 --input_shape [1,180,180,3]
mo --saved_model_dir 000000001 --input_shape [1,180,180,3] --data_type FP16
```

```shell
python tf-classify.py
FPS: 14.72 fps
Inference time: 67.94 ms
FPS: 19.91 fps
Inference time: 50.23 ms
FPS: 20.33 fps
Inference time: 49.20 ms
```

```shell
python ov-classify.py
FPS: 21.27 fps
Inference time: 47.02 ms
Found Cat
FPS: 21.31 fps
Inference time: 46.92 ms
Found Cat
FPS: 21.37 fps
Inference time: 46.80 ms
Found Cat
```
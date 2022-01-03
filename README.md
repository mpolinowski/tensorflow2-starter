# Tensorflow Boiler Plate

This is based on the [Tensor Flow Object Detection Boiler Plate](https://github.com/mpolinowski/TFODCourse). This 
version is "freed" of the Jupyter Notebook dependency.

## Setup

### Arch Linux / PACMAN

```bash
sudo pacman -Syu bazel tensorflow-cuda python-tensorflow-cuda cuda cudnn protobuf
```

### Python / PiP

Create a virtual environment:

```bash
python -m venv tfod
```

Re-enter with `source tfod/bin/activate`.

Install the Python dependencies inside your virtual environment:

```bash
pip install -r dependencies.txt
```

## Collect Trainings Images

_01_collecting_training_data.py_

The script requires an RTSP Stream (e.g. an IP Camera) that you can configure here:

```python
RTSP_URL = 'rtsp://admin:instar@192.168.2.19/livestream/12'
```

Define the objects you want to detect, e.g. hand gestures, here:

```python
labels = ['thumbsup', 'thumbsdown', 'metal', 'ok']
```

Let the script run, and it will tell you what kind of object it is expecting to see - provide sample images. 
__Note__: that there can be a delay from the RTSP stream. You might have to adjust the sleep time to shift the timed 
captures.

The images will have to be labeled and divided on the __training__ and __test__ folders. See [labelImg](https://mpolinowski.github.io/devnotes/2021-11-08--tensorflow-crash-course-part-i#data-collection) for details. 


## Train your Model

_02_training_the_model.py_

Run the script to run a training based on your images using the [ssd_mobilenet_v2_fpnlite_320x320_coco17_tpu-8](http://download.tensorflow.org/models/object_detection/tf2/20200711/ssd_mobilenet_v2_fpnlite_320x320_coco17_tpu-8.tar.gz) pre-trained model.


## Test your Model

_03_run_object_detection_from_file.py_

Pick one of your images from the test folder to verify that it is recognized correctly:

```python
TEST_IMAGE = 'metal.tyrxdf6-zzggg-RGdgfc-zdfg-1cDGF17f.jpg'
```

_04_run_object_detection_from_stream.py_

Alternatively, test the model on your live RTSP stream:

```python
RTSP_URL = 'rtsp://admin:instar@192.168.2.19/livestream/12'
```

## Freeze and Convert

_05_freeze_and_convert_models.py_

Export your model and convert it to use with [TensorFlow.js](https://www.tensorflow.org/js) and [TensorFlow Lite](https://www.tensorflow.org/lite/)
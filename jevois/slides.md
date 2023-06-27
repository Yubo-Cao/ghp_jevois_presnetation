---
theme: academic
highlighter: prism
lineNumbers: false
drawings:
  persist: false
transition: slide-left

class: text-white
title: GHP JeVois Presentation
layout: cover
coverBackgroundUrl: background.jpg
coverAuthor: Yubo Cao, Krish Saxena, Austin Yang
coverDate: Jun 27, 2023

fonts:
  sans: Inter
  serif: EB Garamond
  mono: Cascadia Code
---

# GHP JeVois Presentation

PyDetectionDNN module

---

## Purpose

The purpose of this tutorial is to detect face using OpenCV deep neutral networks. The presentation is based on the [JeVois](http://jevois.org/basedoc/PyDetectionDNN_8py_source.html).

<div class="flex justify-center">
<img src="example.png" alt="PyDetection DNN Example Image" style="width: 40%; height: auto;" />
</div>

---
layout: cols
---

## Initialization Code

```python {all|1|3-6|7|8}
import pyjevois

if pyjevois.pro:
    import libjevoispro as jevois
else:
    import libjevois as jevois
import cv2
import numpy as np
```

::right::

<v-clicks at="0">

- `pyjevois` is a library that allows us to use the JeVois camera
- `libjevois` is a library that allows us to use the JeVois camera. Since we are using the JeVois A33, we use `libjevois` instead of `libjevoispro`
- `cv2` is the OpenCV library
- `numpy` is a library that allows us to use arrays

</v-clicks>

---

## `__init__` I

```python {all|1|3|4|5-10}
def __init__(self):
    # Hyperparameters
    self.confThreshold = 0.5
    self.nmsThreshold = 0.4
    self.inpWidth = 160
    self.inpHeight = 120

    self.scale = 1.0
    self.mean = [104.0, 177.0, 123.0]
    self.rgb = False
```

<v-clicks at="0">

- `confThreshold` is the min confidence that a bbox need to have to be considered as a detection
- `nmsThreshold` is for non-maximum-suppression, which we shall discuss later
- Input image parameters
  - `inpWidth` and `inpHeight` are the shape of the input image
  - `scale` is the scale factor for the image
  - `mean` is used to normalize the input into `[0, 1]`
  - `rgb` is whether the input is RGB or not

</v-clicks>

---

## `__init__` II

```python {all|1-2|3-4|5-|all}
# Load face model
model = "Face"
backend = cv.dnn.DNN_BACKEND_OPENCV
target = cv.dnn.DNN_TARGET_CPU

path = pyjevois.share + "/opencv-dnn/detection/"
classnames = path + "opencv_face_detector.classes"
modelname = path + "opencv_face_detector.caffemodel"
configname = path + "opencv_face_detector.prototxt"
```

<v-clicks at="0">

- `model` is the model that will be used to detect the object
- `backend` is the OpenCV DNN backend
- Finally, the path to the model and the config file are set

</v-clicks>

---

## `__init__` III

```python {all|1-4|5-9|10-13}
# Load classification vocab
if classnames:
    with open(classnames, "rt") as f:
        self.classes = f.read().rstrip("\n").split("\n")

# Load a network
self.net = cv.dnn.readNet(modelname, configname)
self.net.setPreferableBackend(backend)
self.net.setPreferableTarget(target)
self.timer = jevois.Timer("Neural detection", 10, jevois.LOG_DEBUG)
self.model = model
self.outNames = self.net.getUnconnectedOutLayersNames()
```

<v-clicks at=1>

- `classes` are the vocabularies of the class detected, as model will return 0, 1, etc. rather than name of the object detected.
- `net` is the neural network that will be used to detect the object.
- `timer` are used to evaluate the time to process the image.
- `outNames` are the names of the output layers.

</v-clicks>

---
layout: cols
---

## `process` I

```python {all|1-5|6-7|8-10|all}
def process(
    self,
    inframe: jevois.InputFrame,
    outframe: jevois.OutputFrame,
):
    frame = inframe.getCvBGR()
    self.timer.start()

    frameHeight = frame.shape[0]
    frameWidth = frame.shape[1]
```

::right::

<v-clicks at="0">

- `process` is the function that will be called when the camera is processing the image, *i.e.,* forever until the camera is turned off.
  - `inframe` can be used to get the image from the camera.
  - `outframe` can be used to send the image to the host.
- `frame` is the BGR format `np.ndarray` of the image.
- `frameHeight` and `frameWidth` are the height and width of the image, obtained by checking first and second dimension of the `frame` array.

</v-clicks>

---

## `process` II

```python
# Create a 4D blob from a frame.
blob = cv.dnn.blobFromImage(
    frame,
    self.scale,
    (self.inpWidth, self.inpHeight),
    self.mean,
    self.rgb,
    crop=False,
)

# Run a model
self.net.setInput(blob)

# Stack bottom panel below main image:
frame = np.vstack((frame, msgbox))

# Send output frame to host:
outframe.sendCv(frame)
```

---

## `postprocess` I

```python
def postprocess(self, frame: np.ndarray, outs: list):
    frameHeight = frame.shape[0]
    frameWidth = frame.shape[1]

    def drawPred(classId, conf, left, top, right, bottom):
        # Draw a bounding box.
        cv.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)

        label = "%.2f" % (conf * 100)

        # Print a label of class.
        if self.classes:
            if classId >= len(self.classes):
                label = "Oooops id=%d: %s" % (classId, label)
            else:
                label = "%s: %s" % (self.classes[classId], label)
```

---

## Credits

```python
# @author Laurent Itti
# @videomapping YUYV 640 502 20.0 YUYV 640 480 20.0 JeVois PyDetectionDNN
# @email itti@usc.edu
# @address 880 W 1st St Suite 807, Los Angeles CA 90012, USA
# @copyright Copyright (C) 2018 by Laurent Itti
# @mainurl http://jevois.org
# @supporturl http://jevois.org
# @otherurl http://jevois.org
# @license GPL v3
# @distribution Unrestricted
# @restrictions None
# @ingroup modules
```

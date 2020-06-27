# DL Face Detector using OpenCV and SSD Framework

This project consists of 2 components :-
1. **Face Detection in _images_** using OpenCV
2. **Face Detection in _videos_** using OpenCV

This project uses the latest _OpenCV v3.3_ and a _Caffe-based_ Face Detector implementation, which is available in the `face_detector` sub-directory inside the [OpenCV](https://github.com/opencv/opencv/tree/master/samples/dnn/face_detector) Repository.
**OpenCV’s deep learning face detector is based on the Single Shot Detector (SSD) framework with a ResNet base network.**

I used two files in my code :-
- The **.prototxt** file, which defines the model architecture (no. of layers, types of layers etc.)
- The **.caffemodel** file, which contains the weights for the model.
</br>

In the first notebook file, [detect_faces.py](DL-Face-Detector-using-OpenCV-and-SSD-Framework/detect_faces.py), I applied face detection with OpenCV to single input images.

In the second notebook file, [detect_faces_video.py](DL-Face-Detector-using-OpenCV-and-SSD-Framework/detect_faces_video.py), I tweaked the code a little bit and applied face detection with OpenCV to videos, video streams, and webcams.

The code was written on _Spyder (Python 3.7)_.
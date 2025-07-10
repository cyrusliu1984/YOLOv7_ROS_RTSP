# YOLOv7_ROS_RTSP
This project subscribes to a real-time video stream through ROS topics, accelerates the inference of each frame of images via the Huawei Ascend TPU. After that, it draws the detected bounding boxes from the inference results onto the original images and then pushes the processed video out through RTSP streaming.

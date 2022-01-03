import cv2
import uuid
import os
import errno
import time

# Create labels and # of images to record for each label
labels = ['thumbsup', 'thumbsdown', 'metal', 'ok']
number_imgs = 5
# Set path where to store images
IMAGES_PATH = os.path.join('Tensorflow', 'workspace', 'images', 'collection')
# Make path with all parent directories
try:
    os.makedirs(IMAGES_PATH)
except OSError as exc:
    if exc.errno == errno.EEXIST and os.path.isdir(IMAGES_PATH):
        pass
# Create dir for each label in collection
for label in labels:
    path = os.path.join(IMAGES_PATH, label)
    if not os.path.exists(path):
        os.makedirs(path)
# Grab images for each label from RTSP stream
# This is causing some issues - because of the stream delay
# You probably just have to adjust the sleep timer accordingly TODO
for label in labels:
    RTSP_URL = 'rtsp://admin:instar@192.168.2.19/livestream/12'
    os.environ['OPENCV_FFMPEG_CAPTURE_OPTIONS'] = 'rtsp_transport;udp'
    cap = cv2.VideoCapture(RTSP_URL, cv2.CAP_FFMPEG)
    
    print('Collecting images for {}'.format(label))
    time.sleep(5)

    if not cap.isOpened():
        print('Cannot open RTSP stream')
        exit(-1)

    for imgnum in range(number_imgs):
        print('Collecting image {}'.format(imgnum))
        ret, frame = cap.read()
        imgname = os.path.join(IMAGES_PATH, label, label + '.' + '{}.jpg'.format(str(uuid.uuid1())))
        cv2.imwrite(imgname, frame)
        # preview = cv2.resize(frame, (640, 480))
        # cv2.imshow('Captured Frame', preview)
        time.sleep(2)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


cap.release()
# cv2.destroyAllWindows

import cv2

video_capture = cv2.VideoCapture('../VID_20200525_102606.mp4')
i = 0
while True:
    ret, frame = video_capture.read()  # 捕获一帧图像
    if ret:
        cv2.imwrite('wrist_{}.jpg'.format(i), frame)
    else:
        break
    i += 1
video_capture.release()  # 关闭相机
cv2.destroyAllWindows()  # 关闭窗口

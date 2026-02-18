import cv2

def find_cameras(max_index=10):
    available = []
    for i in range(max_index):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            available.append(i)
            cap.release()
    return available

print("Available camera indexes:", find_cameras())


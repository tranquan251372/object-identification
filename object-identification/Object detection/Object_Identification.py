import numpy as np
import argparse
import cv2
from tkinter import filedialog, Button, Tk, Label
import os

# Phân tích đối số
ap = argparse.ArgumentParser()
ap.add_argument("-c", "--confidence", type=float, default=0.2,
                help="Độ xác suất tối thiểu để lọc các phát hiện yếu")
args = vars(ap.parse_args())

# Định nghĩa các lớp đối tượng
CLASSES = ["nen", "may bay", "xe dap", "chim", "thuyen",
           "chai", "xe bus", "oto", "meo", "ghe", "bo",
           "ban an", "cho", "ngua", "xe may", "nguoi",
           "cay trong trong chau", "cuu", "ghe sofa", "tau hoa", "TiVi"]
COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))

# Tải mô hình
net = cv2.dnn.readNetFromCaffe(
    "C:\\object-identification\\Object detection\\MobileNetSSD_deploy.prototxt.txt",
    "C:\\object-identification\\Object detection\\MobileNetSSD_deploy.caffemodel"
)

# Hàm làm nét ảnh
def sharpen_image(image, alpha=1.5, beta=-0.5):
    """
    Làm sắc nét hình ảnh đầu vào bằng cách sử dụng mặt nạ unsharp.
    
    Args:
    image (np.array): Hình ảnh đầu vào.
    alpha (float): Weight của hình ảnh đầu vào.
    beta (float): Weight của hình ảnh mờ.

    Returns:
    np.array: Hình ảnh sắc nét.
    """
    blurred = cv2.GaussianBlur(image, (0, 0), sigmaX=2, sigmaY=2)
    sharpened = cv2.addWeighted(image, alpha, blurred, beta, 0)
    return sharpened

# Hàm nhận diện đối tượng
def detect_objects(frame):
    """
   Phát hiện các đối tượng trong hình ảnh bằng mô hình SSD Mobilenet được đào tạo trước.
    
    Args:
    frame (np.array): Hình ảnh đầu vào.

    Returns:
    np.array: hình ảnh với các đối tượng được phát hiện và các hộp giới hạn được vẽ.
    """
    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 0.007843, (300, 300), 127.5)
    net.setInput(blob)
    detections = net.forward()

    for i in np.arange(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > args["confidence"]:
            idx = int(detections[0, 0, i, 1])
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")
            label = "{}: {:.2f}%".format(CLASSES[idx], confidence * 100)
            cv2.rectangle(frame, (startX, startY), (endX, endY), COLORS[idx], 2)
            y = startY - 15 if startY - 15 > 15 else startY + 15
            cv2.putText(frame, label, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS[idx], 2)
    return frame

# Xử lý khung hình
def process_frame(frame):
    frame = sharpen_image(frame)  # Làm nét ảnh
    frame = detect_objects(frame)  # Nhận diện đối tượng
    return frame

# Xử lý video
def process_video(filename):
    video_capture = cv2.VideoCapture(filename)
    pause = False
    
    while True:
        if not pause:
            ret, frame = video_capture.read()
            if not ret:
                break
            frame = process_frame(frame)
            resized_frame = cv2.resize(frame, (800, 500))
            cv2.imshow("Output", resized_frame)
        # phím điều khiển
        key = cv2.waitKey(30) & 0xFF
        if key == ord('e'):
            break
        elif key == ord('s'):
            pause = not pause
      
    video_capture.release()
    cv2.destroyAllWindows()

# Xử lý hình ảnh
def process_image(filename):
    frame = cv2.imread(filename)
    frame = process_frame(frame)
    resized_frame = cv2.resize(frame, (800, 500))
    cv2.imshow("Output", resized_frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# GUI để chọn tệp
def select_file():
    root.filename = filedialog.askopenfilename(
        initialdir=".",
        title="Chọn tệp hình ảnh hoặc video",
        filetypes=(("MP4 files", "*.mp4"), ("Image files", "*.jpg *.jpeg *.png"), ("all files", "*.*"))
    )
    input_file = root.filename
    if input_file.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
        process_video(input_file)
    elif input_file.lower().endswith(('.jpg', '.jpeg', '.png')):
        process_image(input_file)

# Khởi tạo GUI
root = Tk()
root.title("Nhận diện đối tượng")

label = Label(root, text="Chọn tệp hình ảnh hoặc video:")
label.pack(pady=10)

button = Button(root, text="Chọn tệp", command=select_file)
button.pack(pady=20)

root.mainloop()

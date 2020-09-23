import cv2
import glob

def webcamImage(device):
    succes, img = cv2.VideoCapture(device).read()
    if(not succes):
        raise Exception('webcam not available')
    return img

def view_image(image, name_of_window):
    cv2.namedWindow(name_of_window, cv2.WINDOW_NORMAL)
    cv2.imshow(name_of_window, image)
    cv2.waitKey(100)

def create_video(path, video_name, fps):
    img_array = []
    for filename in glob.glob(path):
        img = cv2.imread(filename)
        img_array.append(img)

    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(video_name, fourcc, 20.0, (640, 480))

    for i in range(len(img_array)):
        out.write(img_array[i])
    out.release()


def play_video(video_name):
    cap = cv2.VideoCapture(video_name)

    while (cap.isOpened()):
        ret, frame = cap.read()
        if(not ret):
            break

        cv2.imshow('frame', frame)
        if cv2.waitKey(1000) & 0xFF == ord('q'):
            break

    cap.release()



img_webcam = webcamImage(0)

view_image(img_webcam, 'Image from webcam')

cv2.imwrite("./images/webcamImage.jpg", img_webcam)

img_store = cv2.imread('./images/webcamImage.jpg')

ret, img_gray = cv2.threshold(img_store, 127, 255, 0)

cv2.imwrite("./images/grayImage.jpg", img_gray)
view_image(img_gray, 'Image gray')

img_with_line = cv2.line(img_gray, (0, 0), (img_gray.shape[1], img_gray.shape[0]), (0, 255, 0), 3)
cv2.imwrite("./images/lineImage.jpg", img_with_line)

img_with_rect = cv2.rectangle(img_with_line, (0, 0), (300, 300), (0, 0, 255), 3)
cv2.imwrite("./images/rectImage.jpg", img_with_rect)

video_name = './video/test.avi'

create_video('./images/*.jpg', video_name, 1)

play_video(video_name)


cv2.destroyAllWindows()


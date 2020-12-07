import cv2
import numpy as np
import glob
import time as tm
import math

def get_keypoint_and_descriptor_SIFT(img):
    sift = cv2.SIFT_create()
    kp_img, des_img = sift.detectAndCompute(img, None)

    return kp_img, des_img


def get_matches_BF_SIFT(des_img1, des_img2):
    bf = cv2.BFMatcher()
    return bf.knnMatch(des_img1, des_img2, k=2)




def get_good_points(matches, threshold):
    good_points = []
    try:
        for m, n in matches:
            if m.distance < threshold * n.distance:
                good_points.append(m)
    except(ValueError):
        return good_points

    return good_points



def replace_obj(good_points, kp_img1, kp_img2, origin_obj_img, current_img, cover_obj_img, MIN_MATCH_COUNT):
    imgAug = current_img.copy()
    if len(good_points) > MIN_MATCH_COUNT:
        img1_pts = np.float32([kp_img1[m.queryIdx].pt for m in good_points]).reshape(-1, 1, 2)
        img2_pts = np.float32([kp_img2[m.trainIdx].pt for m in good_points]).reshape(-1, 1, 2)
        matrix, mask = cv2.findHomography(img1_pts, img2_pts, cv2.RANSAC, 5.0)

        matches_mask = mask.ravel().tolist()

        h, w = origin_obj_img.shape
        pts = np.float32([[0, 0], [0, h], [w, h], [w, 0]]).reshape(-1, 1, 2)

        if(matrix is None):
            return current_img
        dst = cv2.perspectiveTransform(pts, matrix)

        imgWarp = cv2.warpPerspective(
            cover_obj_img, matrix, (current_img.shape[1], current_img.shape[0])
        )

        maskNew = np.zeros((current_img.shape[0], current_img.shape[1]), np.uint8)
        cv2.fillPoly(maskNew, [np.int32(dst)], (255, 255, 255))
        maskInv = cv2.bitwise_not(maskNew)
        imgAug = cv2.bitwise_and(imgAug, imgAug, mask=maskInv)
        imgAug = cv2.bitwise_or(imgWarp, imgAug)
        return imgAug
    else:
        return current_img

def sift(img_from_video, hash_name):

    # img replacement obj
    video_frame = cv2.imread("train_data/replace.png")

    # img rfom video with color
    # test2 = cv2.imread("train_data/test3.jpg")

    # origin img
    origin = cv2.imread("train_data/original.png", cv2.IMREAD_GRAYSCALE)

    height, width = origin.shape
    video_frame = cv2.resize(video_frame, (width, height))



    kp_img1, des_img1 = get_keypoint_and_descriptor_SIFT(origin)
    # img from video to gray color

    kp_img2, des_img2 = get_keypoint_and_descriptor_SIFT(cv2.cvtColor(img_from_video, cv2.COLOR_BGR2GRAY))

    if (des_img1 is None or des_img2 is None):
        # return img2 without change
        return img_from_video


    matches = get_matches_BF_SIFT(des_img1, des_img2)

    good_points = get_good_points(matches, 0.7)


    return replace_obj(good_points, kp_img1, kp_img2, origin, img_from_video, video_frame, 10)


cap = cv2.VideoCapture('video_1.mp4')

if (cap.isOpened() == False):
    print("Unable to read camera feed")

frame_width = int(cap.get(4))
frame_height = int(cap.get(3))

out = cv2.VideoWriter('sift_descriptor.avi', cv2.VideoWriter_fourcc('M','J','P','G'), 20, (frame_width,frame_height))

while(True):
    ret, frame = cap.read()

    if ret == True:
        frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)


        new_frame = sift(frame, 'jopa')


        out.write(new_frame)

    else:
        break

cap.release()
out.release()

cv2.destroyAllWindows()
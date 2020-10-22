import cv2
import numpy as np
import glob
import time as tm
import math

def get_keypoint_and_descriptor_SIFT(img):
    sift = cv2.SIFT_create()
    kp_img, des_img = sift.detectAndCompute(img, None)

    return kp_img, des_img

def get_keypoint_and_descriptor_ORB(img):
    orb = cv2.ORB_create()
    kp_img, des_img = orb.detectAndCompute(img, None)

    return kp_img, des_img

def get_matches_BF_SIFT(des_img1, des_img2):
    bf = cv2.BFMatcher()
    return bf.knnMatch(des_img1, des_img2, k=2)



def get_matches_BF_ORB(des_img1, des_img2):
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


def get_is_detected_and_object(good_points, kp_img1, kp_img2, img1, img2, MIN_MATCH_COUNT):
    if len(good_points) > MIN_MATCH_COUNT:
        img1_pts = np.float32([kp_img1[m.queryIdx].pt for m in good_points]).reshape(-1, 1, 2)
        img2_pts = np.float32([kp_img2[m.trainIdx].pt for m in good_points]).reshape(-1, 1, 2)
        matrix, mask = cv2.findHomography(img1_pts, img2_pts, cv2.RANSAC, 5.0)
        matches_mask = mask.ravel().tolist()

        h, w = img1.shape
        pts = np.float32([[0, 0], [0, h], [w, h], [w, 0]]).reshape(-1, 1, 2)
        if(matrix is None):
            return False, img2
        dst = cv2.perspectiveTransform(pts, matrix)

        img2 = cv2.cvtColor(img2, cv2.COLOR_GRAY2BGR)
        detected_obj = cv2.polylines(img2, [np.int32(dst)], True, (0, 255, 0), 3)
        return True, detected_obj
    else:
        return False, img2


def save_to_result(img1, kp_img1, img_detected_obj, kp_img2, good_points, is_detected, hash_name):

    match = cv2.drawMatches(img1, kp_img1, img_detected_obj, kp_img2, good_points, None, flags=2)
    cv2.imwrite("result/" + hash_name + "_" + ("good" if is_detected else "jopa") + ".jpg", match)



def orb(img1, img2, hash_name):

    time_start = tm.time()

    kp_img1, des_img1 = get_keypoint_and_descriptor_ORB(img1)
    kp_img2, des_img2 = get_keypoint_and_descriptor_ORB(img2)

    if (des_img1 is None or des_img2 is None):
        save_to_result(img1, [], img2, [], [], False, hash_name + "_undefined")
        return 0, 0, float('inf')

    matches = get_matches_BF_ORB(des_img1, des_img2)

    try:
        distances = [m.distance for m, n in matches]
        avr_dist = sum(distances) / len(matches)
    except ValueError:
        save_to_result(img1, [], img2, [], [], False, hash_name + "_undefined")
        return 0, 0, float('inf')

    time_proc = tm.time() - time_start

    #--------

    good_points = get_good_points(matches, 0.7)

    # ---------
    relative_num_of_correct_features = len(good_points) / len(matches)
    #--------

    # -------
    distances = [m.distance for m, n in matches]
    avr_dist = sum(distances) / len(matches)
    # -------

    is_detected, img_detected_obj = get_is_detected_and_object(good_points, kp_img1, kp_img2, img1, img2, 10)

    save_to_result(img1, kp_img1, img_detected_obj, kp_img2, good_points, is_detected, hash_name)

    return time_proc, relative_num_of_correct_features, avr_dist


def sift(img1, img2, hash_name):

    time_start = tm.time()

    kp_img1, des_img1 = get_keypoint_and_descriptor_SIFT(img1)
    kp_img2, des_img2 = get_keypoint_and_descriptor_SIFT(img2)

    if (des_img1 is None or des_img2 is None):
        save_to_result(img1, [], img2, [], [], False, hash_name + "_undefined")
        return 0, 0, float('inf')

    matches = get_matches_BF_SIFT(des_img1, des_img2)

    try:
        distances = [m.distance for m, n in matches]
        avr_dist = sum(distances) / len(matches)
    except ValueError:
        save_to_result(img1, [], img2, [], [], False, hash_name + "_undefined")
        return 0, 0, float('inf')

    #---------
    time_proc = tm.time() - time_start

    #--------

    good_points = get_good_points(matches, 0.7)

    # ---------
    relative_num_of_correct_features = len(good_points) / len(matches)
    #--------

    # -------
    distances = [m.distance for m, n in matches]
    avr_dist = sum(distances) / len(matches)
    # -------

    is_detected, img_detected_obj = get_is_detected_and_object(good_points, kp_img1, kp_img2, img1, img2, 10)

    save_to_result(img1, kp_img1, img_detected_obj, kp_img2, good_points, is_detected, hash_name)

    return time_proc, relative_num_of_correct_features, avr_dist


def compare_correct_data(origin, data_correct, hash = ''):
    file = open("compare_result/compare_correct_data.txt", "a+")
    idx = 0

    for img in data_correct:
        idx+=1

        img_path = f"/result_correct/{hash}{idx}"
        file.write(f"{img_path} \n")

        orb_time, orb_correct, orb_dist = orb(origin, img, f"ORB{img_path}")
        sift_time, sift_correct, sift_dist = sift(origin, img, f"SIFT{img_path}")

        file.write(f"        Time          Good_points/matches      Avarage_distance\n")
        file.write(f"ORB  {orb_time} {orb_correct} {orb_dist} \n")
        file.write(f"Sift {sift_time} {sift_correct} {sift_dist}")
        file.write('\n\n')

    file.close()


def compare_wrong_data(origin, data_wrong, hash = ''):
    file = open("compare_result/compare_wrong_data.txt", "a+")
    idx = 0
    for img in data_wrong:
        idx += 1

        img_path = f"/result_wrong/{hash}{idx}"
        file.write(f"{img_path} \n")

        orb_time, orb_correct, orb_dist = orb(origin, img, f"ORB{img_path}")
        sift_time, sift_correct, sift_dist = sift(origin, img, f"SIFT{img_path}")

        file.write(f"        Time          Good_points/matches      Avarage_distance\n")
        file.write(f"ORB  {orb_time} {orb_correct} {orb_dist} \n")
        file.write(f"Sift {sift_time} {sift_correct} {sift_dist}")
        file.write('\n\n')


    file.close()


origin = cv2.imread("data/train_data/original.png", cv2.IMREAD_GRAYSCALE)

data_correct_path = glob.glob("data/correct_data_test/*.jpg")
data_wrong_path = glob.glob("data/wrong_data_test/*.jpg")

data_correct = [cv2.imread(_, cv2.IMREAD_GRAYSCALE) for _ in data_correct_path]
data_wrong = [cv2.imread(_, cv2.IMREAD_GRAYSCALE) for _ in data_wrong_path]

data_correct_resize = [cv2.resize(cv2.imread(_, cv2.IMREAD_GRAYSCALE), (0,0), fx=0.5, fy=0.5) for _ in data_correct_path]
data_wrong_resize = [cv2.resize(cv2.imread(_, cv2.IMREAD_GRAYSCALE), (0,0), fx=0.5, fy=0.5) for _ in data_wrong_path]

compare_correct_data(origin, data_correct)
compare_wrong_data(origin, data_wrong)

compare_correct_data(origin, data_correct_resize, 'resize')
compare_wrong_data(origin, data_wrong_resize, 'resize')



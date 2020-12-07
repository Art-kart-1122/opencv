import cv2
import numpy as np
import glob
import random
import math
import time as tm
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn import ensemble

from sklearn import ensemble, model_selection
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import make_circles
from sklearn.naive_bayes import GaussianNB

PATH_DATASET_1 = "../lab_2/src/data"
PATH_DATASET_2 = "../lab_2/src/data1"

def compute_data(X, n=8):

    orb = cv2.ORB_create()

    m = len(X)
    X_res = np.zeros((m,n))
    i = 0
    for x in X:
        img = cv2.imread(x, 0)
        img = cv2.resize(img, (128, 128))

        kp, des = orb.detectAndCompute(img, None)

        pt = [i.pt for i in kp]
        pt_arr = np.array(pt)

        if pt_arr.shape[0] < n:
            if len(pt) != 0:
                for j in range(pt_arr.shape[0]):
                    X_res[i,j] = 1 / pt_arr.shape[0]
        else:
            kmeans = KMeans(n_clusters=n, random_state=0).fit(pt_arr)


            list_masks = [ kmeans.labels_ == i for i in range(n)]

            j = 0


            for mask in list_masks:
                pt_mask = np.round(pt_arr[mask,:], 0).astype(np.int)

                color = [random.randint(1, 255), random.randint(1, 255), random.randint(1, 255)]



                hist = np.sum(img[pt_mask[:,0], pt_mask[:,1]]) / np.sum(mask)
                if math.isnan(hist):
                    hist = 0
                    print(x)
                X_res[i,j] = hist
                j += 1

        i += 1

    return X_res

X_dataset_2_true = compute_data(glob.glob(PATH_DATASET_1 + "/correct_data_test/*.jpg"))
Y_dataset_2_true = np.array([1 for i in range(X_dataset_2_true.shape[0])])

X_dataset_2_wrong = compute_data(glob.glob(PATH_DATASET_2 + "/wrong_data_test/*.jpg"))
Y_dataset_2_wrong = np.array([0 for i in range(X_dataset_2_wrong.shape[0])])

X_dataset_1_wrong = compute_data(glob.glob(PATH_DATASET_1 + "/wrong_data_test/*.jpg"))
Y_dataset_1_wrong = np.array([0 for i in range(X_dataset_1_wrong.shape[0])])


X_dataset = np.concatenate((X_dataset_2_true, X_dataset_2_wrong), axis=0)

X_dataset_1 = np.concatenate((X_dataset, X_dataset_1_wrong), axis=0)

Y_dataset = np.concatenate((Y_dataset_2_true, Y_dataset_2_wrong), axis=0)
Y_dataset_1 = np.concatenate((Y_dataset, Y_dataset_1_wrong), axis=0)


X_dataset_1_train, X_dataset_1_test, Y_dataset_1_train, Y_dataset_1_test = train_test_split(X_dataset_1, Y_dataset_1, test_size=0.2)

clf = ensemble.RandomForestClassifier(random_state=0, max_depth=5, max_features='auto', min_samples_split=0.3 )
clf.fit(X_dataset_1_train, Y_dataset_1_train)


# model have learnt

def replace_obj(good_points, is_detected, kp_img1, kp_img2, origin_obj_img, current_img, cover_obj_img):
    imgAug = current_img.copy()
    if is_detected and len(good_points) > 0:
        img1_pts = np.float32([kp_img1[m.queryIdx].pt for m in good_points]).reshape(-1, 1, 2)
        img2_pts = np.float32([kp_img2[m.trainIdx].pt for m in good_points]).reshape(-1, 1, 2)
        print(img1_pts)
        print('---------------')
        print(img2_pts)

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


def compute_hist(frame, n=8):
    orb = cv2.ORB_create()
    m = 1
    X_res = np.zeros((m,n))

    img = cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), (128, 128))
    kp, des = orb.detectAndCompute(img, None)

    pt = [i.pt for i in kp]
    pt_arr = np.array(pt)

    if pt_arr.shape[0] < n:
        if len(pt) != 0:
            for j in range(pt_arr.shape[0]):
                X_res[0,j] = 1 / pt_arr.shape[0]
    else:

        kmeans = KMeans(n_clusters=n, random_state=0).fit(pt_arr)

        list_masks = [ kmeans.labels_ == i for i in range(n)]

        j = 0
        for mask in list_masks:
            pt_mask = np.round(pt_arr[mask,:], 0).astype(np.int)
            hist = np.sum(img[pt_mask[:,0], pt_mask[:,1]]) / np.sum(mask)
            X_res[0,j] = hist
            j += 1

    return X_res

def get_keypoint_and_descriptor_ORB(img):
    orb = cv2.ORB_create()
    kp_img, des_img = orb.detectAndCompute(img, None)

    return kp_img, des_img

def get_good_points(matches, threshold):
    good_points = []
    try:
        for m, n in matches:
            if m.distance < threshold * n.distance:
                good_points.append(m)
    except(ValueError):
        return good_points

    return good_points

def get_matches_BF_SIFT(des_img1, des_img2):
    bf = cv2.BFMatcher()
    return bf.knnMatch(des_img1, des_img2, k=2)


def orb(img_from_video, hash_name):

    # img replacement obj
    video_frame = cv2.imread("train_data/replace.png")


    # origin img
    origin = cv2.imread("train_data/original.png", cv2.IMREAD_GRAYSCALE)

    height, width = origin.shape
    video_frame = cv2.resize(video_frame, (width, height))



    kp_img1, des_img1 = get_keypoint_and_descriptor_ORB(origin)
    # img from video to gray color

    kp_img2, des_img2 = get_keypoint_and_descriptor_ORB(cv2.cvtColor(img_from_video, cv2.COLOR_BGR2GRAY))

    if (des_img1 is None or des_img2 is None):
        # return img2 without change
        return img_from_video
    matches = get_matches_BF_SIFT(des_img1, des_img2)

    good_points = get_good_points(matches, 0.7)

    X = compute_hist(img_from_video)
    is_detected = clf.predict(X)
    print(is_detected)

    if is_detected[0] == 1:
        photo = replace_obj(good_points, True, kp_img1, kp_img2, origin, img_from_video, video_frame)
    else:
        photo = replace_obj(good_points, False, kp_img1, kp_img2, origin, img_from_video, video_frame)
    return photo

cap = cv2.VideoCapture('video_1.mp4')

if (cap.isOpened() == False):
    print("Unable to read camera feed")

frame_width = int(cap.get(4))
frame_height = int(cap.get(3))

out = cv2.VideoWriter('classifier_orb3.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 10, (frame_width, frame_height))

while(True):
    ret, frame = cap.read()

    if ret == True:
        frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
        new_frame = orb(frame, 'jopa')

        out.write(new_frame)

    else:
        break

cap.release()
out.release()

cv2.destroyAllWindows()
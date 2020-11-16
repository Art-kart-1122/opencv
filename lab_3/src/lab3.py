import cv2
import numpy as np
import glob
import random
import math
import time as tm
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn import ensemble
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, roc_curve, roc_auc_score, classification_report, recall_score, precision_recall_curve

import matplotlib.pyplot as plt
from sklearn import ensemble, model_selection
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import make_circles
from sklearn.naive_bayes import GaussianNB

PATH_DATASET_1 = "../../lab_2/src/data"
PATH_DATASET_2 = "../../lab_2/src/data1"

def view_image(image, name_of_window):
    cv2.namedWindow(name_of_window, cv2.WINDOW_NORMAL)
    cv2.imshow(name_of_window, image)
    cv2.waitKey()

def compute_data(X, n=8):

    orb = cv2.ORB_create()

    m = len(X)
    X_res = np.zeros((m,n))
    i = 0
    for x in X:
        img = cv2.imread(x, 0)
        img = cv2.resize(img, (128, 128))

        kp, des = orb.detectAndCompute(img, None)
        #outImage = cv2.drawKeypoints(img, kp,np.array([]),  color=(255,0,0) )
        #view_image(outImage, 'Key points')
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
            #image = np.zeros((img.shape[1], img.shape[0], 3), np.uint8)

            for mask in list_masks:
                pt_mask = np.round(pt_arr[mask,:], 0).astype(np.int)

                color = [random.randint(1, 255), random.randint(1, 255), random.randint(1, 255)]
                #image[pt_mask[:,1], pt_mask[:,0]] = color


                hist = np.sum(img[pt_mask[:,0], pt_mask[:,1]]) / np.sum(mask)
                if math.isnan(hist):
                    hist = 0
                    print(x)
                X_res[i,j] = hist
                j += 1
            #view_image(image, 'Cluster')
        i += 1

    return X_res

X_dataset_2_true = compute_data(glob.glob(PATH_DATASET_2 + "/correct_data_test/*.jpg"))
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
#print(X_dataset_1_train)
clf.fit(X_dataset_1_train, Y_dataset_1_train)
#y_pred = clf.predict(X_dataset_1_test)

#
# def evaluate_predictions(y_true, y_pred, binary=False, probs=None):
#     print("Confusion Matrix:\n", confusion_matrix(y_pred, y_true))
#     if binary:
#         # ROC-AUC Curve
#         # keep probabilities for the positive outcome only
#         print(f"F1-score: {f1_score(y_true, y_pred)} | Precision score: {recall_score(y_true, y_pred)} | Recall score: {recall_score(y_true, y_pred)}")
#         probs = probs[:, 1]
#         fpr, tpr, thresholds = roc_curve(y_true, probs)
#         auc = roc_auc_score(y_true, probs)
#         print(f"AUC score: {auc}")
#         # plot the roc curve for the modelplt.plot(fpr, tpr)
#         plt.title("ROC Curve")
#         # axis labels
#         plt.xlabel('False Positive Rate')
#         plt.ylabel('True Positive Rate')
#         plt.show()
#         # PR Curve
#         precision, recall, _ = precision_recall_curve(y_true, probs)
#         plt.plot(recall, precision)
#         # axis labels
#         plt.title("PR Curve")
#         plt.xlabel('Recall')
#         plt.ylabel('Precision')
#         plt.show()
#     else:
#         print("\n", classification_report(y_true, y_pred))

#
# clf = ensemble.RandomForestClassifier(random_state=0, max_depth=9, max_features='auto', min_samples_split=0.4)
# clf.fit(X_dataset_1_train, Y_dataset_1_train)
# y_pred = clf.predict(X_dataset_1_test)
# evaluate_predictions(Y_dataset_1_test, y_pred)
#
#
#
#

#
# params_depth = [{'max_depth': i} for i in range(1, 20, 1)]
# params_features = [{'max_features': i} for i in ['auto', 'sqrt','log2']]
# params_split = [{'min_samples_split': float(i/10)} for i in range(1, 10, 1)]
# labels_depth = ["max_depth %s" % i for i in range(1, 20, 1)]
# labels_features = ["max_features %s" % i for i in ['auto', 'sqrt','log2']]
# labels_split = ["min_samples_split %s" % float(i/10) for i in range(1, 10, 1)]
#
# print('\nMax_depth\n')
# for param, label in zip(params_depth, labels_depth) :
#     clf = ensemble.RandomForestClassifier(random_state=0, **param)
#     clf.fit(X_dataset_1_train, Y_dataset_1_train)
#     y_pred = clf.predict(X_dataset_1_test)
#     print('Param '+ label +' \n %s' %confusion_matrix(y_pred, Y_dataset_1_test))
# print('\nMax_features\n')
#
# for param, label in zip(params_features, labels_features) :
#     clf = ensemble.RandomForestClassifier(random_state=0, **param)
#     clf.fit(X_dataset_1_train, Y_dataset_1_train)
#     y_pred = clf.predict(X_dataset_1_test)
#     print('Param '+ label +' \n %s' %confusion_matrix(y_pred, Y_dataset_1_test))
#
# print('\nMin_samples_split\n')
# for param, label in zip(params_split, labels_split) :
#     clf = ensemble.RandomForestClassifier(random_state=0, **param)
#     clf.fit(X_dataset_1_train, Y_dataset_1_train)
#     y_pred = clf.predict(X_dataset_1_test)
#     print('Param '+ label +' \n %s' %confusion_matrix(y_pred, Y_dataset_1_test))

#
#

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
        time_start = tm.time()
        kmeans = KMeans(n_clusters=n, random_state=0).fit(pt_arr)
        print(tm.time() - time_start)
        list_masks = [ kmeans.labels_ == i for i in range(n)]

        j = 0
        for mask in list_masks:
            pt_mask = np.round(pt_arr[mask,:], 0).astype(np.int)
            hist = np.sum(img[pt_mask[:,0], pt_mask[:,1]]) / np.sum(mask)
            X_res[0,j] = hist
            j += 1

    return X_res


cap = cv2.VideoCapture('dataset_2.mp4')

if (cap.isOpened() == False):
    print("Unable to read camera feed")

frame_width = int(cap.get(4))
frame_height = int(cap.get(3))

out = cv2.VideoWriter('orb_dataset_2_time_kmeans.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 10, (frame_width,frame_height))

while(True):
    ret, frame = cap.read()

    if ret == True:
        frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)


        X = compute_hist(frame)


        class_object = clf.predict(X)




        position = (25, 350)
        cv2.putText(frame, 'Class ' + str(class_object), position, cv2.FONT_HERSHEY_SIMPLEX, 2, (209, 80, 0, 255), 3)
        out.write(frame)

    else:
        break

cap.release()
out.release()

cv2.destroyAllWindows()
#import sys, os
#sys.path.append("./..")
#os.chdir("./..")

from utils import *
import cascade as cs
import cv2, math
import pandas as pd
import numpy as np
from timeit import default_timer as timer



class Point:
    def __init__(self, x, y):
        self.x = x
        self.y = y
    

def Draw_Boxes_on_face(image, cascade):
    sart1 = timer()
    positives = positive_rectangles(image, 19, 1.25, cascade)

    sart2 = timer()
    bins_old = make_partition(positives)
    boxes_old = get_all_boxes(bins_old)
    for _ in range(10):
        bins = make_partition(boxes_old)
        boxes = get_all_boxes(bins)
        if(len(boxes_old) == len(boxes)):
            break
        else:
            boxes_old = boxes

    sart3 = timer()
    boxed_image = draw_boxes(image, boxes)

    end = timer()
    time_take = end - sart1
    print("Time taken for positive rectangles: ", sart2 - sart1)
    print("Time taken for partitioning: ", sart3 - sart2)
    print("Time taken for drawing boxes: ", end - sart3)
    print("Total time taken: ", time_take)
    return boxed_image, time_take


def create_cascade():
    X_train_fe,y_train = Create_data(100)
    Strong_Classifiers,No_of_samples_per_layer,No_of_positive_samples_per_layer,No_of_negative_samples_per_layer = np.array(cs.Train_Cascade(X_train_fe, y_train))
    return Strong_Classifiers 



# getting all the possitive windows
def positive_rectangles(image, window_size_base, ratio, cascade):
    sizes, deltas  = window_sizes_deltas(image, window_size_base, ratio)
    print("No of windo sizes: ", len(sizes))
    print("No of windows:", count_total_subwindows(image, sizes, deltas))
    rectangles = []
    features = []
    for i in range(len(sizes)):
        rectangle, feature = positive_rectangles_size(image, sizes[i], deltas[i], window_size_base, cascade)
        rectangles += rectangle
        features += feature
    print("expected features to be evaluated: ", np.mean(features))
    print("number_of_positive_boxes:", len(rectangles))
    return rectangles


def window_sizes_deltas(img, min_wind, ratio):
    sizes = []
    deltas = []
    temp = min_wind * ratio
    dtemp = 1.5
    while(round(temp) < min(img.shape)):
        sizes.append(math.floor(round(temp)))
        deltas.append(math.floor(round(dtemp)))
        temp *= ratio
        dtemp *= ratio
    return sizes, deltas

def count_total_subwindows(img, sizes, deltas):
    total = 0
    for i in range(len(sizes)):
        total += (((img.shape[0] - sizes[i] + 1) * (img.shape[1] - sizes[i] + 1)) // (deltas[i] * deltas[i]))
    return total


def positive_rectangles_size(image, size, delta, window_size_base, cascade):
    #dim = (width, height)
    dim = (image.shape[1] - size + 19, image.shape[0] - size + 19)
    new_image = cv2.resize(image, dim)
    height = new_image.shape[0] - 19 + 1
    width = new_image.shape[1] - 19 + 1

    Ans = []
    features = []
    for i in range(0, height, delta):
        for j in range(0, width, delta):
            #print(size, i*width + j)
            window = new_image[i:i+window_size_base, j:j+window_size_base]
            #print(window.shape, window_size_base)
            pos, n_feature = cs.Cascade_predict_opt(window, [1], cascade)
            features.append(n_feature)
            if pos == True:
                rect = [Point(j, i), Point(j + size - 1, i + size - 1)]
                Ans.append(rect)
            else:
                continue
    return Ans, features




# Making Partions from the list of possitive windows
def make_partition(positives):
    bins = []
    for rect in positives:
        found_bin = False
        for bin in bins: 
            if found_bin == False:
                for rect_bin in bin:
                    if(do_overlap_api(rect, rect_bin)):
                        bin.append(rect)
                        found_bin = True
                        break
            else:
                break
        if found_bin == False:
            bins.append([rect])
    return bins


def do_overlap_api(rectangles_1,rectangles_2):
    return do_overlap(rectangles_1[0], rectangles_1[1], rectangles_2[0], rectangles_2[1])


def do_overlap(l1, r1, l2, r2):
    # if rectangle has area 0, no overlap
    if l1.x == r1.x or l1.y == r1.y or r2.x == l2.x or l2.y == r2.y:
        return False
    # If one rectangle is on left side of other
    if l1.x > r2.x or l2.x > r1.x:
        return False
    # If one rectangle is above other
    if r1.y < l2.y or r2.y < l1.y:
        return False
    return True





# merging all windows in a particular partition
def get_all_boxes(partions):
    boxes = []
    for partion in partions:
        boxes.append(mean_rect(partion))
    return boxes


def mean_rect(rect_array):
    ls = [i[0] for i in rect_array]
    rs = [i[1] for i in rect_array]
    ls_av = mean_points(ls)
    rs_av = mean_points(rs)
    return (ls_av, rs_av)


def mean_points(l1_array):
    x_new = 0
    y_new = 0
    for i in range(len(l1_array)):
        x_new += l1_array[i].x
        y_new += l1_array[i].y
    x_new /= len(l1_array)
    y_new /= len(l1_array)
    return Point(int(x_new), int(y_new))







# Drwaing the boxes on the image
def draw_boxes(imageo, boxes):
    image = imageo.copy() 
    for box in boxes:
        image = cv2.rectangle(image, (box[0].x, box[0].y), (box[1].x, box[1].y), (0, 255, 0), 2)
    return image








#creating cascade
cascade = pickle.load(open("cascade.pkl", "rb"))
print("Cascade loaded")


#image reading
img = cv2.imread("end-to-end-window/test2.jpeg")
print(img.shape)

scale_percent = 22 # percent of original size
width = int(img.shape[1] * scale_percent / 100)
height = int(img.shape[0] * scale_percent / 100)
dim = (width, height)
resized = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
print(resized.shape)

image = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
print(image.shape)
cv2.imwrite('end-to-end-window/test_resized.jpg', image)

boxed, time = Draw_Boxes_on_face(image, cascade)
cv2.imwrite('end-to-end-window/test_boxed.jpg', boxed)
print(time)
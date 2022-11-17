import numpy as np


##       API 
#  get_features()
#      - input: grey scale image 24*24, numpy 2d array 
#      - output: numpy 1d array of features 


def get_features(image):
    int_image = caliculate_intergral_image(image)    
    (length, bredth) = image.shape
    rect_type1 = combine_pairs(get_all_pairs(length), get_all_pairs_2x(bredth))
    fetures_type1 = [two_rect_feature_v(a, b, c, d, int_image) for a, b, c, d in rect_type1]
    rect_type2 = combine_pairs(get_all_pairs_2x(length), get_all_pairs(bredth))
    fetures_type2 = [two_rect_feature_h(a, b, c, d, int_image) for a, b, c, d in rect_type2]
    rect_type3 = combine_pairs(get_all_pairs(length), get_all_pairs_3x(bredth))
    fetures_type3 = [three_rect_feature_v(a, b, c, d, int_image) for a, b, c, d in rect_type3]
    rect_type4 = combine_pairs(get_all_pairs_3x(length), get_all_pairs(bredth))
    fetures_type4 = [three_rect_feature_h(a, b, c, d, int_image) for a, b, c, d in rect_type4]
    rect_type5 = combine_pairs(get_all_pairs_2x(length), get_all_pairs_2x(bredth))
    fetures_type5 = [four_rect_feature(a, b, c, d, int_image) for a, b, c, d in rect_type5]
    features = fetures_type1 + fetures_type2 + fetures_type3 + fetures_type4 + fetures_type5
    return np.array(features)


def caliculate_intergral_image(img):
    integral_img = np.zeros(img.shape)
    colum_sum = np.zeros(img.shape[1])
    for i in range(img.shape[0]):
        colum_sum += img[i]
        integral_img[i] = np.cumsum(colum_sum)
    return integral_img


def combine_pairs(X, Y):
    ans = []
    for x in X:
        for y in Y:
            a = (x[0] , y[0])
            b = (x[0] , y[1])
            c = (x[1] , y[1])
            d = (x[1] , y[0])
            ans.append((a, b, c, d))
    return ans


def get_all_pairs(no_of_items):
    ans = []
    for i in range(0, no_of_items):
        for j in range(i+1, no_of_items):
            ans.append((i, j))
    return ans


def get_all_pairs_2x(no_of_items):
    ans = []
    for i in range(0, no_of_items):
        for j in range(i+2, no_of_items, 2):
            ans.append((i, j))
    return ans


def get_all_pairs_3x(no_of_items):
    ans = []
    for i in range(0,no_of_items):
        for j in range(i+3, no_of_items, 3):
            ans.append((i, j))
    return ans


def two_rect_feature_v(a, b, c, d, int_image):
    ab = mid_point(a, b)
    cd = mid_point(c, d)
    grey = area_rectange(ab, b, c, cd, int_image)
    white = area_rectange(a, ab, cd, d, int_image)
    return grey - white 


def two_rect_feature_h(a, b, c, d, int_image):
    da = mid_point(d, a)
    bc = mid_point(b, c)
    grey = area_rectange(a, b, bc, da, int_image)
    white = area_rectange(da, bc, c, d, int_image)
    return grey - white 


def three_rect_feature_v(a, b, c, d, int_image):
    ab_1 = first_third(a, b)
    ab_2 = second_third(a, b)
    cd_1 = first_third(c, d)
    cd_2 = second_third(c, d)
    grey = area_rectange(ab_1, ab_2, cd_1, cd_2, int_image)
    white = area_rectange(a, ab_1, cd_2, d, int_image) + area_rectange(ab_2, b, c, cd_1, int_image)
    return grey - white 


def three_rect_feature_h(a, b, c, d, int_image):
    bc_1 = first_third(b, c)
    bc_2 = second_third(b, c)
    da_1 = first_third(d, a)
    da_2 = second_third(d, a)
    grey = area_rectange(da_2, bc_1, bc_2, da_1, int_image)
    white = area_rectange(a, b, bc_1, da_2, int_image) + area_rectange(da_1, bc_2, c, d, int_image)
    return grey - white 


def four_rect_feature(a, b, c, d, int_image):
    ab = mid_point(a, b)
    bc = mid_point(b, c)
    cd = mid_point(c, d)
    da = mid_point(d, a)
    o = mid_point(ab, cd)
    grey = area_rectange(ab, b, bc, o, int_image) + area_rectange(da, o, cd, d, int_image)
    white = area_rectange(a, ab, o, da, int_image) + area_rectange(o, bc, c, cd, int_image)
    return grey - white 


# Shape of the rectange.
#a--b
#d--c
def area_rectange(a, b, c, d, int_image):
    return (int_image[c] - int_image[d]) - (int_image[b] - int_image[a])


def mid_point(a, b):
    sum = (a[0] + b[0], a[1] + b[1])
    return (sum[0] // 2, sum[1] // 2)


def first_third(a, b):
    sum = (2 * a[0] + b[0], 2 * a[1] + b[1])
    return (sum[0] // 3, sum[1] // 3)


def second_third(a, b):
    sum = (a[0] + 2 * b[0], a[1] + 2 * b[1])
    return (sum[0] // 3, sum[1] // 3)
import numpy as np



def get_all_feature(image):
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
    return integral_img


def combine_pairs(X, Y):
    ans = []
    for x in X:
        for y in Y:
            a = (X[0] , Y[0])
            b = (X[0] , Y[1])
            c = (X[1] , Y[1])
            d = (X[1] , Y[0])
            ans.append((a, b, c, d))
    return ans


def get_all_pairs(no_of_items):
    ans = []
    for i in range(no_of_items):
        for j in range(i+1, no_of_items):
            ans.append((i, j))
    return ans


def get_all_pairs_2x(no_of_items):
    ans = []
    for i in range(no_of_items):
        for j in range(i+2, no_of_items, 2):
            ans.append((i, j))
    return ans


def get_all_pairs_3x(no_of_items):
    ans = []
    for i in range(no_of_items):
        for j in range(i+3, no_of_items, 3):
            ans.append((i, j))
    return ans


def two_rect_feature_v(a, b, c, d, int_image):
    e = (a + b) / 2
    f = (c + d) / 2
    grey = area_rectange(e, b, c, f, int_image)
    white = area_rectange(a, e, f, d, int_image)
    return white - grey


def two_rect_feature_h(a, b, c, d, int_image):
    e = (a + c) / 2
    f = (b + c) / 2
    grey = area_rectange(a, b, f, e, int_image)
    white = area_rectange(e, f, c, d, int_image)
    return white - grey


def three_rect_feature_v(a, b, c, d, int_image):
    ab_1 = (2*a + b) / 3
    ab_2 = (a + 2*b) / 3
    cd_1 = (2*c + d) / 3
    cd_2 = (c + 2*d) / 3
    grey = area_rectange(ab_1, ab_2, cd_1, cd_2, int_image)
    white = area_rectange(a, ab_1, cd_2, d, int_image) + area_rectange(ab_2, b, c, cd_1, int_image)
    return white - grey


def three_rect_feature_h(a, b, c, d, int_image):
    bc_1 = (2*b + c) / 3
    bc_2 = (b + 2*c) / 3
    da_1 = (2*d + a) / 3
    da_2 = (a + 2*d) / 3
    grey = area_rectange(da_2, bc_1, bc_2, da_1, int_image)
    white = area_rectange(a, b, bc_1, da_2, int_image) + area_rectange(da_1, bc_2, c, d, int_image)
    return white - grey


def four_rect_feature(a, b, c, d, int_image):
    ab = (a + b)/2
    bc = (b + c)/2
    cd = (c + d)/2
    da = (d + a)/2
    o = (a + b + c + d)/4
    grey = area_rectange(ab, b, bc, o, int_image) + area_rectange(da, o, cd, d, int_image)
    white = area_rectange(a, ab, o, da, int_image) + area_rectange(o, bc, c, cd, int_image)
    return white - grey


#a--b
#d--c
def area_rectange(a, b, c, d, int_image):
    return (int_image[c] - int_image[d]) - (int_image[b] - int_image[a])


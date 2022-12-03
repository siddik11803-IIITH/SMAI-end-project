import features
import numpy as np
import random



def test_integral_image(l):
    Input = np.ones((l, l))
    iiInput = features.caliculate_intergral_image(Input)
    for i in range(l):
        for j in range(l):
            assert(iiInput[i][j] == (i + 1) * (j + 1))
    return True

 
def test_combine_pairs():
    # Simple to veirfy correctness
    pass


def test_get_pairs(l):
    # Simple to veirfy correctness
    pass


def test_get_pairs_23x(l):
    pairs2 = features.get_all_pairs_2x(l)
    for pair in pairs2:
        assert((pair[1] + pair[0]) % 2 == 0)
    pairs3 = features.get_all_pairs_3x(l)
    for pair in pairs3:
        assert((2 * pair[1] + pair[0]) % 3 == 0)
        assert((pair[1] + 2 * pair[0]) % 3 == 0)
    return True 


def test_area_rectangle(a, b, c, d, int_image):
    delx = c[0] - a[0]
    dely = c[1] - a[1]
    assert(dely * delx == features.area_rectange(a, b, c, d, int_image))
    return True


def test_nth_feature(image):
    Iimage = features.caliculate_intergral_image(image)
    rectangles = features.get_rectanges(image.shape[0], image.shape[1])
    all_features = features.get_features(image, rectangles)
    for n in range(len(all_features)):
        feature = features.get_nth_feature(Iimage, rectangles, n)
        assert(abs(feature -  all_features[n]) < 1e-9)
    return True


def complete_test():
    ## Data setup
    l = random.randint(1, 25)
    int_image = features.caliculate_intergral_image(np.ones((l, l)))
    
    rect_type1 = features.combine_pairs(features.get_all_pairs(l), features.get_all_pairs_2x(l))
    fetures_type1 = [features.two_rect_feature_v(a, b, c, d, int_image) for a, b, c, d in rect_type1]
    rect_type2 = features.combine_pairs(features.get_all_pairs_2x(l), features.get_all_pairs(l))
    fetures_type2 = [features.two_rect_feature_h(a, b, c, d, int_image) for a, b, c, d in rect_type2]
    rect_type3 = features.combine_pairs(features.get_all_pairs(l), features.get_all_pairs_3x(l))
    fetures_type3 = [features.three_rect_feature_v(a, b, c, d, int_image) for a, b, c, d in rect_type3]
    rect_type4 = features.combine_pairs(features.get_all_pairs_3x(l), features.get_all_pairs(l))
    fetures_type4 = [features.three_rect_feature_h(a, b, c, d, int_image) for a, b, c, d in rect_type4]
    rect_type5 = features.combine_pairs(features.get_all_pairs_2x(l), features.get_all_pairs_2x(l))
    fetures_type5 = [features.four_rect_feature(a, b, c, d, int_image) for a, b, c, d in rect_type5]

    featuress_even = fetures_type1 + fetures_type2 + fetures_type5
    featuress_odd = fetures_type3 + fetures_type4
    rectangles_even = rect_type1 + rect_type2 + rect_type5
    rectangles_odd = rect_type3 + rect_type4


    ## Testing
    test_integral_image(l)
    test_combine_pairs()
    test_get_pairs(l)
    test_get_pairs_23x(l)

    for feature in featuress_even:
        assert(abs(feature - 0) < 1e-9)
    for i in range(len(rectangles_odd)):
        a = rectangles_odd[i][0]
        b = rectangles_odd[i][1]
        c = rectangles_odd[i][2]
        d = rectangles_odd[i][3]
        assert(abs(featuress_odd[i] + features.area_rectange(a, b, c, d, int_image)/3) < 1e-9)
        
    for a, b, c, d in (rectangles_even + rectangles_odd):
        test_area_rectangle(a, b, c, d, int_image)

    test_nth_feature(np.ones((19, 19)))
    return




if __name__ == "__main__":
    complete_test()
    print("All tests passed")
import numpy as np



class Image():
    def __init__(self, img):
        self.img = img
    def i(self, x ,y):
        return self.img[x][y]
    def s(self, x, y):
        if(y == -1):
            return 0
        else:
            return self.s(x, y-1) + self.i(x, y)
    def ii(self, x, y):
        if(x == -1):
            return 0
        else:
            return self.ii(x-1, y) + self.s(x, y)
            
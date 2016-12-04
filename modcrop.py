import numpy as np
def modcrop(img,modulo):
    if img.shape[2]==1:
        sz = np.array(img.shape)
        sz = sz - sz%modulo
        ix = range(sz[0])
        iy = range(sz[1])
        iz = range(1)
        img = img[np.ix_(ix,iy,iz)]
        return img
    else :
        sz = np.array(img.shape)
        sz = sz - sz%modulo
        ix = range(sz[0])
        iy = range(sz[1])
        iz = range(3)
        img = img[np.ix_(ix,iy,iz)]
        return img

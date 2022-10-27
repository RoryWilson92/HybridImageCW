import numpy as np
from PIL import Image

from MyConvolution import convolve
from MyHybridImages import makeGaussianKernel

image = Image.open('res/cat.bmp')
data = np.asarray(image)
k = makeGaussianKernel(2)
k0 = np.array([[1, 0, -1],
			   [2, 0, -2],
			   [1, 0, -1]])
k1 = k0.transpose()
k2 = np.matmul(k0, k1)
image2 = Image.fromarray(convolve(data, k))
image2.save("out.bmp")
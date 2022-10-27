import numpy as np
import PIL


def floor(x: float):
	return int(x) if x - int(x) >= 0 else int(x) - 1


def convolve(image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
	"""
	Convolve an image with a kernel assuming zero-padding of the image to handle the borders
	
	:param image: the image (either greyscale shape=(rows,cols) or colour shape=(rows,cols,channels))
	:type numpy.ndarray
	
	:param kernel: the kernel (shape=(kheight,kwidth); both dimensions odd)
	:type numpy.ndarray
	
	:returns the convolved image (of the same shape as the input image)
	:rtype numpy.ndarray
	"""
	# Your code here. You'll need to vectorise your implementation to ensure it runs
	# at a reasonable speed.

	imX = image.shape[0]
	imY = image.shape[1]
	channelNum = image.shape[2]

	tpXrad = floor(kernel.shape[0] / 2)
	tpYrad = floor(kernel.shape[1] / 2)

	kernelInverse = kernel.transpose()
	res = image.copy()

	for col in range(tpXrad, imX - tpXrad):
		for row in range(tpYrad, imY - tpYrad):
			for chn in range(channelNum):
				currentArray = image[col - tpXrad:col + tpXrad + 1, row - tpYrad:row + tpYrad + 1, chn]
				res[col, row, chn] = (currentArray * kernelInverse).sum()

	return res


from PIL import Image
from numpy import asarray

image = Image.open('desk.jpg')
data = asarray(image)
k = np.array([[-1 / 9, -1 / 9, -1 / 9],
			  [-1 / 9, -1 / 9, -1 / 9],
			  [-1 / 9, -1 / 9, -1 / 9]])
k0 = np.array([[1, 0, -1],
			   [2, 0, -2],
			   [1, 0, -1]])
k1 = k0.transpose()
k2 = np.matmul(k0, k1)
image2 = Image.fromarray(convolve(data, k))
image2.save("desk4.png")

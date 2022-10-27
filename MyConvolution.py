import numpy as np


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

	tpXrad = kernel.shape[0] // 2
	tpYrad = kernel.shape[1] // 2

	kernelInverse = kernel.transpose()
	res = image.copy()

	for col in range(tpXrad, imX - tpXrad):
		for row in range(tpYrad, imY - tpYrad):
			for chn in range(channelNum):
				currentArray = image[col - tpXrad:col + tpXrad + 1, row - tpYrad:row + tpYrad + 1, chn]
				res[col, row, chn] = (currentArray * kernelInverse).sum()

	return res

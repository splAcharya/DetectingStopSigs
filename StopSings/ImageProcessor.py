#Author: Swapnil Acharya
#Since: 5/16/2020

import numpy


def applyHistogramEqualization(imageArray, imageHeight, imageWidth):
	""" This function perfrom histogram eqalization on a 2D grey scale image array

		Instead of manupulating contrast and brightness of an image manually, Histogram equalization
		attempts to detemined the best value for both brightness and contrast. It overall improves the 
		appearence of the image. This function increases brightnes over most of the image and decreases 
		brightness over some of the image. Overall, histogram equalization improves appearance of an image.

		Args:
			imageArray:
				The 2D greyscale image array

			imageHeight:
				The height of the 2D image array

			imageWidth:
				The wdith of the 2D image array

		Returns:
			histogram euqalized 2D image array, the height and width of return array is the same as input imageArray
	"""

	print("Startting Histogram Equalization")

	#build histogram
	histogram = numpy.zeros([256],int) #declare a histogram container of size 256 since, a pixel's intensity in a greyscale image can go from 0-255

	for i in range(0,imageHeight):
		for j in range(0,imageWidth):
			intensity = imageArray[i,j]
			histogram[intensity] = histogram[intensity] + 1


	#build cumulative histogram
	for i in range(1,256):
		histogram[i] = histogram[i-1] + histogram[i]


	#Build normalized histogram
	histogram = histogram * (255/(imageHeight*imageWidth))

	#apply histogram equalization
	for i in range(0,imageHeight):
		for j in range(0,imageWidth):
			intensity = imageArray[i,j]
			newIntensity = histogram[intensity]
			if(newIntensity < 0):
				imageArray[i,j] = 0
			elif(newIntensity > 255):
				imageArray[i,j] =255
			else:
				imageArray[i,j] = newIntensity

	
	print("Completed Histogram Equalization")
	return imageArray



def applyGaussianBlur(imageArray, imageHeight, imageWidth):
	""" This function smooths(blurrs) an image by apply guassian blur.

		Args:
			imageArray: the 2D image array of an greyscale image
			
			imageHeight: the height of 2D image array

			imageWidth: the width of 2D image array

		Returns:
			blurred 2D image array whose height and width are the same as the input 2D image Array
	"""

	print("Started Applying Gaussian Blur")
   
	#Define Gausian Kernel
		
	#              1,  4,  6,  4, 1
	#              4, 16, 26, 16, 4
	#   1/256 *    6, 24, 36, 24, 6
	#              4, 16, 24, 16, 4
	#              1, 4,   6,  4, 1


	kernel = numpy.array([[1,  4,  6,  4, 1],
						 [4, 16, 26, 16, 4],
						 [6, 24, 36, 24, 6],
						 [4, 16, 24, 16, 4],
						 [1, 4,   6,  4, 1]],float)

	kernel = kernel / 256 #scale kernel

	paddedImageArray = numpy.zeros([imageHeight+4+4, imageWidth+4+4],float) #pad 4 zeroes in all direction since, gaussian kernel is 5x5
	paddedImageArray[4:imageHeight+4,4:imageWidth+4] = numpy.array(imageArray,float).copy() #insert image array in the zero array to create zero padded array
	convolvedImageArray = numpy.zeros(paddedImageArray.shape,float)

	#start convolution
	for i in range(4,paddedImageArray.shape[0]-4):
		for j in range(4,paddedImageArray.shape[1]-4):
			outerSum = 0
			for m in range(0,kernel.shape[0]):
					innerSum = 0
					for n in range(0,kernel.shape[1]):
						innerSum += paddedImageArray[i-m,j-n] * kernel[m,n] 
					outerSum += innerSum
			convolvedImageArray[i,j] = outerSum
			if( 255 < convolvedImageArray[i,j] < 257):
				convolvedImageArray[i,j]=255
		
	#remove zero padding
	imageArray = convolvedImageArray[4:imageHeight+4, 4:imageWidth+4].copy() #remove zero padding
	print("Completed Applying Gaussian Blur")
	return imageArray




#def discreetFourierTransfrom2D(inputArr, imageHeight, imageWidth):
	""" This function perfrom discreet fourier transfrom on an 2D greyscale image.

		Args:
			imageArray: the 2D image array of grey scale image

			imageHeight: the height of 2D image array

			imageWidth: the width of 2D image array

		Returns:
			the Frequency Domain equivalent of image array
	"""


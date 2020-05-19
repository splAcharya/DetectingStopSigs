#Author: Swapnil Acharya
#Since: 5/16/2020

import numpy

def adjustBrightness(imageArray, adjustmentConstant):
	""" This function adjusts the brightness of the 2D array by a contant value

	Args:
		imageArray: the 2D greyscale image array whos brightness is to be adjusted
		
		adjustmentConstant: The constant that is to be used for brightness adjustment

	Returns:
		None
	"""

	print("Starting Brightness adjustment")

	imageHeight, imageWidth = imageArray.shape

	for i in range(0,imageHeight):
		for j in range(0,imageWidth):
			tempC =  imageArray[i,j] + adjustmentConstant
			if(tempC > 255):
				imageArray[i,j] = 255
			else:
				imageArray[i,j] = tempC
	print("Brightness adjustment complete")

	return imageArray #return brightness adjusted image array
	



def adjustContrastInAllPixels(imageArray, centerConstant, factor):
	""" This function adjusts the contrast of the 2D array given a correction factor 

	Args:
		imageArray: the 2D image array whose brightness is to be adjusted

		centerContant: the offset or centervalue from and to which pixel's intesity is to be moved or pull towards

		factor: The constant that is used to adjust brightness
				increase contrast, 1.05 < factor < 1.2
				decrease constrast 0.1 < factor < 1.0

	Returns:
		None
	"""

	print("Starting contrast adjustment")

	imageHeight, imageWidth = imageArray.shape
	for i in range(0,imageHeight):
		for j in range(0,imageWidth):
			tempC = int(imageArray[i,j])
			tempD = int(factor * (tempC - centerConstant) + centerConstant)
			if(tempD < 0):
				imageArray[i,j] = 0
			elif(tempD > 255):
				imageArray[i,j] = 255
			else:
				imageArray[i,j] = tempD
	print("Contrast Adjustment complete")

	return imageArray #return contrast adjusted imageArray




def applyHistogramEqualization(imageArray):
	""" This function perfrom histogram eqalization on a 2D grey scale image array

		Instead of manupulating contrast and brightness of an image manually, Histogram equalization
		attempts to detemined the best value for both brightness and contrast. It overall improves the 
		appearence of the image. This function increases brightnes over most of the image and decreases 
		brightness over some of the image. Overall, histogram equalization improves appearance of an image.

		Args:
			imageArray:
				The 2D greyscale image array

		Returns:
			histogram euqalized 2D image array, the height and width of return array is the same as input imageArray
	"""

	print("Startting Histogram Equalization")


	imageHeight, imageWidth = imageArray.shape #get height and widhth of 2D image array

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





def convolve(imageArray,kernel):
	""" This function perfrom convolution between image array and given kernel.

		Zeros padding is done around the image array based on the dimenstion of kernel
		after that convolution is perfromed.

		Args:
			imageArray: the 2D greyscale image array

			kernel: the kernel/filter/mask to apply to the 2D image array

		Returns:
			convolved imageArray, the hieght and width of the convolved array is the same as
			the input imageArray
	"""

	imageHeight, imageWidth = imageArray.shape
	kernelHeight, kernelWidth = kernel.shape

	paddedImageArray = numpy.zeros([imageHeight+((kernelHeight-1)*2), imageWidth+((kernelWidth-1)*2)],float) #pad 4 zeroes in all direction since, gaussian kernel is 5x5
	paddedImageArray[(kernelHeight-1):imageHeight+(kernelHeight-1),(kernelWidth-1):imageWidth+(kernelHeight-1)] = numpy.array(imageArray,float).copy() #insert image array in the zero array to create zero padded array
	convolvedImageArray = numpy.zeros(paddedImageArray.shape,float)

	#start convolution
	for i in range(4,paddedImageArray.shape[0]-(kernelHeight-1)):
		for j in range(4,paddedImageArray.shape[1]-(kernelWidth-1)):
			outerSum = 0
			for m in range(0,kernel.shape[0]):
					innerSum = 0
					for n in range(0,kernel.shape[1]):
						innerSum += paddedImageArray[i-m,j-n] * kernel[m,n] 
					outerSum += innerSum
			convolvedImageArray[i,j] = outerSum
		
	#remove zero padding
	convolvedImageArray = convolvedImageArray[4:imageHeight+4, 4:imageWidth+4].copy() #remove zero padding

	#return convolved array
	return convolvedImageArray





def applyGaussianBlur(imageArray):
	""" This function smooths(blurrs) an image by apply guassian blur.

		Args:
			imageArray: the 2D image array of an greyscale image

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

	
	blurredArray = convolve(imageArray,kernel) #apply blurr via convolution

	print("Completed Applying Gaussian Blur")

	return blurredArray #return blurred array




def detectEdges(imageArray):
	""" This function perfrom edge detection in an greyscale 2D imageArray

		this function applyies vertical and horizational sobel masks to image array,
		before useing this function smotthing masks such as gaussian blur is recomended

		Args:
			imageArray: the 2D grey scale image array

		Returns:
			edge edetected image array
	"""

	print("Started Edge detection")

	# sobel vertical mask
	#	-1, 0,	1
	#	-2,	0,	2
	#	-1,	0,	1
	sobelVerticalMask = numpy.array([[-1,0,1],[-5,0,5],[-1,0,1]],float)


	#sobel horizontal mask
	#	-1,	-2, -1
	#	0,	0,	0
	#	1,	2,	1
	sobelHorizontalMask = numpy.array([[-1,-5,-1],[0,0,0],[1,5,1]],float)


	#apply vertical mask to 2D image array
	verticalEdges = numpy.array(convolve(imageArray,sobelVerticalMask),float)

	#apply horizontal mask to 2D image array
	horizontalEdges = numpy.array(convolve(imageArray,sobelHorizontalMask),float)


	#get gradient magnitude
	gradientMadnitude = numpy.hypot(horizontalEdges,verticalEdges)

	#get gradient direction
	gradientDirection = numpy.arctan2(verticalEdges,horizontalEdges)

	print("completed Edge detection")

	return gradientMadnitude, gradientDirection #return imageArray and imageAray direction





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

	
	blurredArray = convolve(imageArray.copy(),kernel.copy()) #apply blurr via convolution

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
	sobelVerticalMask = numpy.array([[-1,0,1],[-2,0,2],[-1,0,1]],float)


	#sobel horizontal mask
	#	-1,	-2, -1
	#	0,	0,	0
	#	1,	2,	1
	sobelHorizontalMask = numpy.array([[-1,-2,-1],[0,0,0],[1,2,1]],float)


	#apply vertical mask to 2D image array
	verticalEdges = numpy.array(convolve(imageArray.copy(),sobelVerticalMask.copy()),float)

	#apply horizontal mask to 2D image array
	horizontalEdges = numpy.array(convolve(imageArray.copy(),sobelHorizontalMask.copy()),float)

	#get gradient magnitude
	gradientMadnitude = numpy.hypot(horizontalEdges,verticalEdges)

	#get gradient direction
	gradientDirection = numpy.arctan2(verticalEdges,horizontalEdges)

	print("completed Edge detection")

	return gradientMadnitude, gradientDirection #return imageArray and imageAray direction




def applyNonMaximSupression(imageArray,imageGradient):
	""" THis function perfrom gradient based thinning of the 2D pixel data

		THe result of this operation is reflected on the 2D array.

		Args:
			imageArray: the 2D edge detected image array

			imageGradient: the gradient of 2D edge detect image

		Returns:
			2D imageArray with edges supressed
	"""


	print("Thinning Edges via non maxium supression started")
		
	#convert directions from radian to degrees
	gradientDirection = imageGradient.copy()
	pixelData = imageArray.copy()

	gradientDirection = gradientDirection * (180/numpy.pi) #convert from radian to degrees

	suppressCount = 0
	negcount = 0

	imageHeight, imageWidth = pixelData.shape

	for i in range(1,imageHeight-1):
		for j in range(1,imageWidth-1):

			#get the current edge's direction
			direction = gradientDirection[i,j]
			if(direction < 0):
				negcount += 1

			#define container for pixel at opposite edges
			opEdge1 = 0 
			opEdge2 = 0

			#if gradient direction is between  either to left or right of the current pixel
			if(( 0 <= direction < 22.5 ) or (157.5 <= direction <= 180)):
				opEdge1 = pixelData[i,j+1]
				opEdge2 = pixelData[i,j-1]

			#if gradient direction is at the 45 degree angle or 225 degree angle
			elif( 22.5 <= direction < 67.5):
				opEdge1 = pixelData[i+1,j-1]
				opEdge2 = pixelData[i-1,j+1]

			#if gradient direction is at the 90 degree angle or 270 degree angle
			elif( 67.5 <= direction < 122.5 ):
				opEdge1 = pixelData[i+1,j]
				opEdge2 = pixelData[i-1,j]

			#if gradient direction is at the 135 degree angle or 315 degree angle
			elif( 112.5 <= direction < 157.5 ):
				opEdge1 = pixelData[i-1,j-1]
				opEdge2 = pixelData[i+1,j+1]
				
			#check
			if((pixelData[i,j] >= opEdge1) and (pixelData[i,j] >= opEdge2)):
				pass
			else:
				pixelData[i,j] = 0
				suppressCount += 1

	print("Thinning Edges via non maxium supression completed. SupressCount = %d Negcount = %d"%(suppressCount,negcount))

	return pixelData




def doubleThresholdingandEdgeTracking(imageArray,lowerThreshold,upperThreshold, neighborCount):
	"""This function perfrom double thresholding and edge tracking to remove noise and thing edges from the 2D pixel data

		The result of this operation is reflected on 2D array

		Args:
			imageArray: the 2D greyscale image array
			lowerThreshold: pixel intensity below this threshold value will be set to zero
			upperThreshold: pixel intensoty above this threshold value will be set to 255
			Neighborcount: the number of surrounding pixels to be check for edge tracking

		Returns:
			thresholded  adn edge tracked imageArray
	"""

	print("Started Removing unrelated Edges via Double Thresholding and tracking related edge via edge tracking")
		
	pixelData = imageArray.copy()
	imageHeight, imageWidth = pixelData.shape

	for i in range(neighborCount,imageHeight-neighborCount):
		for j in range(neighborCount,imageWidth-neighborCount):

			#if this pixel's intensity is above upper threshoold then its an edge
			if(pixelData[i,j] > upperThreshold):
				pixelData[i,j] = 255

			#if the pixel's  intensity is below lower threshold the its not and edge
			elif(pixelData[i,j] < lowerThreshold):
				pixelData[i,j] = 0

			#if the pixel's intenshity is below uperthreshold and above lower threshold, check surrounding pixels
			else:
					
				#get horizontally right neigbors
				hrNeighbors = list(pixelData[i,j+1:neighborCount+1])
				hrCount = len([k for k in hrNeighbors if k > upperThreshold])

						
				#get horizontally left neighbors
				hlNeighbors = list(pixelData[i,j-1:j-neighborCount-1:-1])
				hlCount = len([k for k in hlNeighbors if k > upperThreshold])



				#get vertically down neighbors
				vdNeighbors = list(pixelData[i+1:neighborCount+1,j])
				vdCount = len([k for k in vdNeighbors if k > upperThreshold])


						   
				#get vertically up neighbors
				vuNeighbors = list(pixelData[i-1:i-neighborCount-1,j])
				vuCount = len([k for k in vuNeighbors if k > upperThreshold])

				#get primary diagonal right
				pdrNeighbors = []
				mthIndex = i + 1
				nthIndex= j + 1
				indexCount = 0
				while(indexCount < neighborCount):
					pdrNeighbors.append(pixelData[mthIndex,nthIndex])
					mthIndex += 1
					nthIndex += 1
					indexCount +=1
				pdrCount = len([k for k in pdrNeighbors if k > upperThreshold])



				#get primary diagonal left
				pdlNeighbors = []
				mthIndex = i - neighborCount
				nthIndex= j  - neighborCount
				indexCount = 0
				while(indexCount < neighborCount):
					pdlNeighbors.append(pixelData[mthIndex,nthIndex])
					mthIndex += 1
					nthIndex += 1
					indexCount += 1
				pdlCount = len([k for k in pdlNeighbors if k > upperThreshold])


				#get seconday diagonal left
				sdlNeighbors=[]
				mthIndex = i + 1
				nthIndex = j - 1
				indexCount = 0
				while(indexCount < neighborCount):
					sdlNeighbors.append(pixelData[mthIndex,nthIndex])
					mthIndex += 1
					nthIndex -= 1
					indexCount += 1
				sdlCount = len([k for k in sdlNeighbors if k > upperThreshold])


				#get secondary diagonal right
				sdrNeighbors=[]
				mthIndex = i - 1
				nthIndex = j + 1
				indexCount = 0
				while(indexCount < neighborCount):
					sdrNeighbors.append(pixelData[mthIndex,nthIndex])
					mthIndex -= 1
					nthIndex += 1
					indexCount += 1
				sdrCount = len([k for k in sdrNeighbors if k > upperThreshold])


				#descision to keep the pixel or not
				if(( hlCount >= neighborCount) or (hrCount >= neighborCount) or (vuCount >= neighborCount) or (vdCount >= neighborCount) or (pdrCount >= neighborCount) or (pdlCount >= neighborCount) or (sdrCount >= neighborCount) or (sdlCount >= neighborCount)):
					pixelData[i,j] = 255
				else:
					pixelData[i,j] = 0
		
	print("Completed Removing unrelated Edges via Double Thresholding and tracking related edge via edge tracking")

	return pixelData #return imageArray 



def applyLineDetector(imageArray):
	"""This function applies line detector to 2D image array

		Args:
			imageArray: the 2D greyscale image array
	"""

	imageHeight, imageWidth  = imageArray.shape


	#define line kernels
	horizontalKernel = numpy.array([[-1,-1,-1],
									[2,2,2],
									[-1,-1,-1]],float)


	verticalKernel = numpy.array([[-1,2,-1],
								  [-1,2,-1],
								  [-1,2,-1]],float)

	pos45Kernel = numpy.array([[-1,-1,2],
								[-1,2,-1],
								[2,-1,-1]],float)
	
	neg45Kernel = numpy.array([[2,-1,-1],
							   [-1,2,-1],
							   [-1,-1,2]],float)


	#convolved with different kernels
	horizontalEdges = convolve(imageArray,horizontalKernel) #hotizontal edges
	verticalEdges = convolve(imageArray,verticalKernel) #vertical edges
	pos45Edges = convolve(imageArray,pos45Kernel) #edges along rimary diagonal
	neg45Edges = convolve(imageArray,neg45Kernel) #edges alson seconday diagonal


	
	strongestEdge = numpy.zeros(imageArray.shape,float) #define container

	greatest = lambda a,b: a if a > b else b #lamba expression to find greater element in two numbers

	#find the strongest edge
	for i in range(0,imageHeight):
		for j in range(0,imageWidth):

			max = greatest(horizontalEdges[i,j],verticalEdges[i,j])
			max = greatest(max,pos45Edges[i,j])
			max = greatest(max,neg45Edges[i,j])

			strongestEdge[i,j] = max

	return strongestEdge
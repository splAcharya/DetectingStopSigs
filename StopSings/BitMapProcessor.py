#Author: Swapnil Acharya
#Since: 5/16/2020

import numpy #for array and basic math functions

def readBitMapImage(ImagePath):
	""" Default constructor Reads a bit map image file and extract header file information and raw pixel data

	Header file of bitmap images contain information such as widht and height of image,
	the offset from where pixel data can be read. This function also initialized required
	private, protected and public feilds.

	Args:
		Imagepath: THe physical path of the image file to read


	Returns:
		None

	"""

	print("Image Scanning Started")
	fileName = open(ImagePath,"rb") #open file to read bytes
	#BITMAP FILE HEADER
	fileType = int.from_bytes(fileName.read(2),"little") # 2 characte string value in ASCII. It must be 'BM' or '0x42 0x4D'     
	fileSize = int.from_bytes(fileName.read(4),"little") #An unsigned int representing entire file size in bytes
	reserved1 = int.from_bytes(fileName.read(2),"little")
	reserved2 = int.from_bytes(fileName.read(2),"little")
	pixelDataOffset = int.from_bytes(fileName.read(4),"little")# Unsinged int representing the offset of actual pixel data in bytes
	fileName.seek(0) #reset file read index to zero to save complete FILE header for backup
	fileheader = fileName.read(14)
		
	#BITMAP INFO HEADER
	headerSize= int.from_bytes(fileName.read(4),"little") #Unsigned Integer representing the size of the headerin bytes
	imageWidth = int.from_bytes(fileName.read(4),"little") #extract width information( [19-22] bytes)
	imageHeight = int.from_bytes(fileName.read(4),"little") #extract height information( [23-26] bytes)
	planes = int.from_bytes(fileName.read(2),"little")# Unsigned integer representing the numer of color planes.
	bitsPerPixel = int.from_bytes(fileName.read(2),"little")#extract bitcount to see if ColorTable section exists
	compression = int.from_bytes(fileName.read(4),"little") #Unsinged integer representing the value of compression to use
	imageSize = int.from_bytes(fileName.read(4),"little") #unsigned integer representing the final size of the compressed image.
	xPixelPerMeter = int.from_bytes(fileName.read(4),"little") #unsigned integer 
	yPixelPerMeter = int.from_bytes(fileName.read(4),"little") #unsigned integer
	totalColors = int.from_bytes(fileName.read(4),"little") #unsinged int representing the numbers of colors in the color palate
	importantColors = int.from_bytes(fileName.read(4),"little") #unsinged integer representing the number of important colors
	fileName.seek(14) #reset file read index to zero to save complete FILE INFO HEADER for backup
	infoheader = fileName.read(pixelDataOffset - 14)
		

	#COLOR TABLE, ONLY EXISTS IF BITS PER PIXEL <= 8
	red = 0
	green = 0
	blue = 0
	reserved3 = 0
	if(bitsPerPixel <= 8):
		red = int.from_bytes(fileName.read(1),"little")
		green = int.from_bytes(fileName.read(1),"little")
		blue = int.from_bytes(fileName.read(1),"little")
		reserved3 = int.from_bytes(fileName.read(1),"little")

		

	#pirnt image height
	print("Height: %d, Width: %d"%(imageHeight,imageWidth))

	#COMPLETE HEADER
	fileName.seek(0) #reset file read cursor position to begining of file
	completeHeader = fileName.read(pixelDataOffset) #get complete header


	#READ ACTUAL PIXEL DATA AS  1 bytes EACH
	bytesPerPixel = bitsPerPixel//8
	pixelData = [] #list to hold raw pixel data whihc will late be a numpy array
	for i in range(0,imageHeight):
		tempL = [] #temporary list to hold each row
		for j in range(0,imageWidth):
			curPixel = int.from_bytes(fileName.read(1),"little")
			temp = fileName.read(2)
			tempL.append(curPixel)
		pixelData.append(tempL)

	fileName.close() #close file 
	pixelData = numpy.array(pixelData) #convert to numpy array,
	print("Image Scanning Completed")
	return completeHeader, pixelData, imageHeight, imageWidth



def writeBitMapImage(header,imageArray,imageHeight,imageWidth,imageName,pathToSave):
	"""This function writes the 2D pixel array into a Bitmap image file

	Args:
		ImagePath: The Physical path in drive where the image file in to be output
		ImageName: The name for the image file

	Returns:
		None
	"""

	print("Image Writting Started")
	fileToSave = open((pathToSave + imageName + ".bmp"),"wb+") #create a file towrite
	fileToSave.write(header) #write header
		
	#write data
	for i in range(0,imageHeight):
		for j in range(0,imageWidth):
			temp = int(imageArray[i,j])
			temp = temp.to_bytes(1,"little")
			fileToSave.write(temp) #write the piel intensity for R
			fileToSave.write(temp) #write the piel intensity for G
			fileToSave.write(temp) #write the piel intensity for B
	fileToSave.close() #save and close file
	print("Image Writting Complete")
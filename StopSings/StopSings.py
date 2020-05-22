import BitMapProcessor as BMP
import ImageProcessor as IMP 
import numpy
import time
import os
def main():

    startTime = time.time()
    path = "D:\Repo\Personal\DetectingStopSigns\StopSings\inputImages\image_1.bmp"
    pathToSave = "D:\Repo\Personal\DetectingStopSigns\StopSings\outputImages\oi_"
    header, imgAr, imgH, imgW = BMP.readBitMapImage(path)
    imgAr = IMP.applyHistogramEqualization(imgAr) #improve appearence of the image
    imgAr = IMP.applyGaussianBlur(imgAr) #smooth the image with low pass filter
    imgAr, imgArDr = IMP.detectEdges(imgAr) #apply sobel edge detection
    imgAr = IMP.applyNonMaximSupression(imgAr,imgArDr) #thin edges
    imgAr = IMP.doubleThresholdingandEdgeTracking(imgAr,70,150,5)
    
    #im = numpy.eye(imgAr.shape[0],imgAr.shape[1])
    #for i in range(0,imgH):
    #    for j in range(0,imgW):
    #        if(im[i,j] == 1):
    #            im[i,j] = 255

    #imgAr = IMP.blendTwoImages(imgAr,im,0.6)
    #TODO: change blend setting to see if the second image has and edge then emphasize those edge elements and let other elemtns be same from first image

    houghAcc = IMP.houghTransfrom(imgAr,thetaStep = 5)
    houghPoints = IMP.detectHoughPoints(houghAcc,50,imgH,imgW)
    imgAr = IMP.createHoughLineImage(houghPoints,imgH,imgW)


    #TODO: to see hough lines, maybe try plotting all the points from one point to another and adding them together using nump.eye
    BMP.writeBitMapImage(header,imgAr,"image_1blended",pathToSave)
    #TODO: learn about FFT and write your won FFT algorithm
    #TODO: use only one threshold(lower threshold) and track anything above the lower threshold
    #TODO: it is not possibe to draw hough lines on original image, the only way to do it is to 
    #       superimpose one image over the other

    endTime = time.time()
    print("Elapsed Time: %0.3f seconds"%(endTime-startTime))

if __name__ == "__main__":
    main()

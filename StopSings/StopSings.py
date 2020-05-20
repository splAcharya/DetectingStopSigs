import BitMapProcessor as BMP
import ImageProcessor as IMP 




def main():
    path = "D:\Repo\Personal\DetectingStopSigs\StopSings\inputImages\image_1.bmp"
    pathToSave = "D:\Repo\Personal\DetectingStopSigs\StopSings\outputImages\oi_"
    header, imgAr, imgH, imgW = BMP.readBitMapImage(path)
    a = imgAr.copy()
    imgAr = IMP.applyHistogramEqualization(imgAr) #improve appearence of the image
    imgAr = IMP.applyGaussianBlur(imgAr) #smooth the image with low pass filter
    imgAr, imgArDr = IMP.detectEdges(imgAr) #apply sobel edge detection
    imgAr = IMP.applyNonMaximSupression(imgAr,imgArDr) #thin edges
    imgAr = IMP.doubleThresholdingandEdgeTracking(imgAr,70,150,5)
    imgAr = IMP.blendTwoImages(imgAr,a,0.5)    
    #houghAcc = IMP.houghTransfrom(imgAr,thetaStep = 5)
    #TODO: to see hough lines, maybe try plotting all the points from one point to another and adding them together using nump.eye
    BMP.writeBitMapImage(header,imgAr,"image_1blended",pathToSave)
    #TODO: learn about FFT and write your won FFT algorithm
    #TODO: use only one threshold(lower threshold) and track anything above the lower threshold
    #TODO: it is not possibe to draw hough lines on original image, the only way to do it is to 
    #       superimpose one image over the other

if __name__ == "__main__":
    main()

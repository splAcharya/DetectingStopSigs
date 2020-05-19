import BitMapProcessor as BMP
import ImageProcessor as IMP 




def main():
    path = "D:\Repo\Personal\DetectingStopSigs\StopSings\inputImages\image_1.bmp"
    pathToSave = "D:\Repo\Personal\DetectingStopSigs\StopSings\outputImages\oi_"
    header, imgAr, imgH, imgW = BMP.readBitMapImage(path)
    imgAr = IMP.applyHistogramEqualization(imgAr) #improve appearence of the image
    imgAr = IMP.applyGaussianBlur(imgAr) #smooth the image with low pass filter
    imgAr, imgArDr = IMP.detectEdges(imgAr) #apply sobel edge detection
    imgAr = IMP.applyNonMaximSupression(imgAr,imgArDr) #thin edges
    imgAr = IMP.doubleThresholdingandEdgeTracking(imgAr,70,150,5)
    BMP.writeBitMapImage(header,imgAr,"image_1ab",pathToSave)
    #TODO: learn about FFT and write your won FFT algorithm
    #TODO: use only one threshold(lower threshold) and track anything above the lower threshold
if __name__ == "__main__":
    main()

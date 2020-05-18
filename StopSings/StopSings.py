import BitMapProcessor as BMP
import ImageProcessor as IMP




def main():
    path = "D:\Repo\Personal\DetectingStopSigs\StopSings\inputImages\image_1.bmp"
    pathToSave = "D:\Repo\Personal\DetectingStopSigs\StopSings\outputImages\oi_"
    header, imgAr, imgH, imgW = BMP.readBitMapImage(path)
    imgAr = IMP.applyHistogramEqualization(imgAr,imgH,imgW) #improve appearence of the image
    imgAr = IMP.applyGaussianBlur(imgAr,imgH,imgW) #smooth the image with low pass filter
    BMP.writeBitMapImage(header,imgAr,imgH,imgW,"image_1",pathToSave)
    #TODO: learn about FFT and write your won FFT algorithm

if __name__ == "__main__":
    main()

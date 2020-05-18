import BitMapProcessor as BMP
import ImageProcessor as IMP




def main():
    path = "D:\OneDrive\Onedurive\Ms_Computer_Science_Stuff\School_Stuff\Spring2020\CSCI_540_Introduction_To_Artifical_Intelligence\Vision_AI_Project\Code\StopSigns\images\image_1.bmp"
    pathToSave = "D:\OneDrive\Onedurive\Ms_Computer_Science_Stuff\School_Stuff\Spring2020\CSCI_540_Introduction_To_Artifical_Intelligence\Vision_AI_Project\Code\StopSigns\SS"
    header, imgAr, imgH, imgW = BMP.readBitMapImage(path)
    imgAr = IMP.applyHistogramEqualization(imgAr,imgH,imgW) #improve appearence of the image
    imgAr = IMP.applyGaussianBlur(imgAr,imgH,imgW)
    BMP.writeBitMapImage(header,imgAr,imgH,imgW,"image_1",pathToSave)
    #TODO: learn about FFT and write your won FFT algorithm

if __name__ == "__main__":
    main()

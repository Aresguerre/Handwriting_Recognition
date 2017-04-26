# TrainAndTest.py

import cv2
import numpy as np
import operator
import os

MIN_CONTOUR_AREA = 1000

RESIZED_IMAGE_WIDTH = 20
RESIZED_IMAGE_HEIGHT = 30

class Contour():

    npTabContour = None
    bordRect = None          
    intRectX = 0                
    intRectY = 0                 
    intRectWidth = 0             
    intRectHeight = 0            
    aire = 0.0                

    def setTopLeftWidthHeight(self):                
        [intX, intY, intWidth, intHeight] = self.bordRect
        self.intRectX = intX
        self.intRectY = intY
        self.intRectWidth = intWidth
        self.intRectHeight = intHeight

    def valid(self):                             
        if self.fltArea < MIN_CONTOUR_AREA: return False         
        return True
 
def main():
    contours = []                 
    contoursValid = []               

    try:
        npTabClassifications = np.loadtxt("classifications.txt", np.float32)                   
    except:
        print "error, unable to open classifications.txt, exiting program\n"
        os.system("pause")
        return
    # end try

    try:
        npTabApplatis = np.loadtxt("flattened_images.txt", np.float32)                  
    except:
        print "error, unable to open flattened_images.txt, exiting program\n"
        os.system("pause")
        return
    # end try

    npTabClassifications = npTabClassifications.reshape((npTabClassifications.size, 1))       

    kNearest = cv2.ml.KNearest_create()                   

    kNearest.train(npTabApplatis, cv2.ml.ROW_SAMPLE, npTabClassifications)

    srcTest = cv2.imread("test1.png")          

    if srcTest is None:                            
        print "error: image not read from file \n\n"         
        os.system("pause")                                   
        return                                               
    # end if

    srcTestGray = cv2.cvtColor(srcTest, cv2.COLOR_BGR2GRAY)        
    srcTestBlur = cv2.GaussianBlur(srcTestGray, (5,5), 0)                     

                                                         
    srcTestInv = cv2.adaptiveThreshold(srcTestBlur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)                                    

    cpyTestInv = srcTestInv.copy()        

    cpyContours, npTabContours, npTabHierarchy = cv2.findContours(cpyTestInv, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)   

    for npTabContour in npTabContours:                             
        contour = Contour()                                             
        contour.npTabContour = npTabContour                                          
        contour.bordRect = cv2.boundingRect(contour.npTabContour)     
        contour.setTopLeftWidthHeight()                    
        contour.aire = cv2.contourArea(contour.npTabContour)           
        contours.append(contour)                                     
    # end for

    for contour in contours:                 
        if contour.valid():             
            contoursValid.append(contour)       
        # end if
    # end for

    contoursValid.sort(key = operator.attrgetter("intRectX"))         
    strFinalString = ""         

    for contour in contoursValid:         
                                               
        cv2.rectangle(srcTest, (contour.intRectX, contour.intRectY), (contour.intRectX + contour.intRectWidth, contour.intRectY + contour.intRectHeight), (0, 255, 0),2)                        

        imgCropped = srcTestInv[contour.intRectY : contour.intRectY + contour.intRectHeight, contour.intRectX : contour.intRectX + contour.intRectWidth]

        imgCroppedResized = cv2.resize(imgCropped, (RESIZED_IMAGE_WIDTH, RESIZED_IMAGE_HEIGHT))             

        npTabApplati = imgCroppedResized.reshape((1, RESIZED_IMAGE_WIDTH * RESIZED_IMAGE_HEIGHT))      

        npTabApplati = np.float32(npTabApplati)       

        retval, npTabResults, neigh_resp, dists = kNearest.findNearest(npTabApplati, k = 1)     

        print npTabResults
        strCurrentChar = str(chr(int(npTabResults[0][0])))                                            

        strFinalString = strFinalString + strCurrentChar            
    # end for

    print "\n" + strFinalString + "\n"                 

    cv2.imshow("srcTest", srcTest)      
    cv2.waitKey(0)                                          

    cv2.destroyAllWindows()             

    return

if __name__ == "__main__":
    main()
# end if
import sys
import numpy as np
import cv2
import os
 
# MIN_CONTOUR_AREA = 100

# RESIZED_IMAGE_WIDTH = 20
# RESIZED_IMAGE_HEIGHT = 30
 
def main():
    srcLearn = cv2.imread("digits.png")             

    if srcLearn is None:                           
        print "error: image not read from file \n\n"         
        os.system("pause")                                  
        return                                               
    # end if
    # cv2.imshow("original", srcLearn)
    srcGray = cv2.cvtColor(srcLearn, cv2.COLOR_BGR2GRAY)           
    # cv2.imshow("gray", srcGray)
    srcBlur = cv2.GaussianBlur(srcGray, (5,5), 0)                         
    # cv2.imshow("blur", srcBlur)
    srcInv = cv2.adaptiveThreshold(srcBlur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)                                     
    cv2.imshow("srcInv", srcInv)       
    cpyInv = srcInv.copy()         
    cpyContours, npTabContours, npaHierarchy = cv2.findContours(cpyInv, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)            
    npTabApplatis =  np.empty((0, 20 * 30))
    intClassifications = []          
    intValidChars = [ord('0'), ord('1'), ord('2'), ord('3'), ord('4'), ord('5'), ord('6'), ord('7'), ord('8'), ord('9'),
                     ord('A'), ord('B'), ord('C'), ord('D'), ord('E'), ord('F'), ord('G'), ord('H'), ord('I'), ord('J'),
                     ord('K'), ord('L'), ord('M'), ord('N'), ord('O'), ord('P'), ord('Q'), ord('R'), ord('S'), ord('T'),
                     ord('U'), ord('V'), ord('W'), ord('X'), ord('Y'), ord('Z'), 
                     ord('a'), ord('b'), ord('c'), ord('d'), ord('e'), ord('f'), ord('g'), ord('h'), ord('i'), ord('j'), 
                     ord('k'), ord('l'), ord('m'), ord('n'), ord('o'), ord('p'), ord('q'), ord('r'), ord('s'), ord('t'), 
                     ord('u'), ord('v'), ord('w'), ord('x'), ord('y'), ord('z'),
                     ord('+'), ord('-'), ord('*'), ord(',')]

    for npTabContour in npTabContours:                           
        if cv2.contourArea(npTabContour) > 100:          
            [intX, intY, intW, intH] = cv2.boundingRect(npTabContour)         
            cv2.rectangle(srcLearn, (intX, intY), (intX+intW,intY+intH), (0, 0, 255), 2)                            
            imgCropped = srcInv[intY:intY+intH, intX:intX+intW]                                  
            imgCroppedResized = cv2.resize(imgCropped, (20, 30))     
            cv2.imshow("imgCropped", imgCropped)                    
            cv2.imshow("imgCroppedResized", imgCroppedResized)      
            cv2.imshow("srcLearn", srcLearn)      
            intChar = cv2.waitKey(0)                    
            if intChar == 27:                  
                sys.exit()                     
            elif intChar in intValidChars:      
                intClassifications.append(intChar)                                               
                npTabApplati = imgCroppedResized.reshape((1, 20 * 30))  
                npTabApplatis = np.append(npTabApplatis, npTabApplati, 0)                    
            # end if
        # end if
    # end for
    floatClassifications = np.array(intClassifications, np.float32)                   
    npTabClassification = floatClassifications.reshape((floatClassifications.size, 1))   
    print "\n\nDONE !!\n"

    np.savetxt("classifications.txt", npTabClassification)           
    np.savetxt("flattened_images.txt", npTabApplatis)          

    cv2.destroyAllWindows()

    return

if __name__ == "__main__":
    main()
# end if





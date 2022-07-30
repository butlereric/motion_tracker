# cython code for motion detector - actual detector

import numpy as np
import cv2,datetime
import csv,time

cpdef run(int blurnum,double accum,int sens,int setup,name,int runtime):
    cap = cv2.VideoCapture(0)
    ret,f = cap.read()
    avg = np.float32(f)
    shape = f.shape
    totpix = shape[0]*shape[1]
    fourcc = cv2.cv.CV_FOURCC(*'XVID')
    out = cv2.VideoWriter('videos/'+name+'.avi',fourcc,20.0,(640,480))
    font = cv2.FONT_HERSHEY_SIMPLEX
    kernel = np.ones((blurnum,blurnum),np.uint8)
    writeinter = open('videos/'+name+'.csv','wb')
    writefile = csv.writer(writeinter)
    starttime = time.clock()
                
    cdef int step = 0
    cdef int vidset = 0
    cdef int counter = 0

    run = True
    while run:
        
        ret,f = cap.read()
        fblur = cv2.morphologyEx(f,cv2.MORPH_OPEN,kernel)
        
        cv2.accumulateWeighted(fblur,avg,accum)
        
        bkg = cv2.convertScaleAbs(avg)
        diff = cv2.absdiff(fblur,bkg)

        if setup == 1:
            if np.sum(diff)/totpix > sens:
                cv2.imshow('frame',diff)
            else:
                cv2.imshow('frame',f)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            if (time.clock()-starttime) > runtime:
                run = False
            if np.sum(diff)/totpix > sens:
                dt = str(datetime.datetime.now().time())
                cv2.putText(f,dt,(10,470),font,1,(0,0,255))
                out.write(f)
                vidset = 1
                counter = 0
                step+=1
                row = (step,dt)
                writefile.writerow(row)

    # this always ends things
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    writeinter.close()

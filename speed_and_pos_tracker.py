'''Tracks moving object in video, compares to CSV file for timestamps'''

# dt needs to read in time

import numpy as np
import cv2
import csv
import os

path = 'Test Files'

font = cv2.FONT_HERSHEY_SIMPLEX
writeOut = []

def review(aviName,(starts,stops),csvfile,write):
    cap = cv2.VideoCapture(os.path.join(path,aviName))
    ret,frame = cap.read()
    
    avg = np.float32(frame)
    kernel = np.ones((7,7),np.uint8)
    totdist = 0
    tottime = 0
    
    avgstarts = []
    for s in starts:
        avgstarts.append(s-200)
    
    framenum = 0
    run = False
    runavgs = False
    framecount = 0
    previouspos = None
    previoustime = None
    
    print 'First roach at:',starts[0]

    for x in range(0,int(cap.get(cv2.cv.CV_CAP_PROP_FRAME_COUNT)-1)):
        if framenum in starts:
            run = True
        elif framenum in stops:
            run = False
            runavgs = False
        if framenum in avgstarts:
            runavgs = True
        if framecount == 1000:
            print framenum
            framecount = 0
        ret, frame = cap.read()
        framenum+=1
        framecount+=1
        
        if runavgs:
            timestamp = csvfile[framenum][1]
            dt = timeConvert(timestamp)
            cv2.putText(frame,timestamp,(10,470),font,1,(211,213,205))
            fblur = cv2.morphologyEx(frame,cv2.MORPH_OPEN,kernel) #cv2.blur(frame,(1,1))
            cv2.accumulateWeighted(fblur,avg,.01)
            res = cv2.convertScaleAbs(avg)
            
            if run:
                diff = cv2.absdiff(fblur,res)
                diff = cv2.cvtColor(diff,cv2.COLOR_BGR2GRAY)
                diff = cv2.threshold(diff,80,255,cv2.THRESH_BINARY)[1]
                cnts = cv2.findContours(diff,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)[0]
                if len(cnts) > 0:
                    cnt = sorted(cnts,key=cv2.contourArea,reverse=True)[0]
                    if len(cnt) > 5:
                        ellipse = cv2.fitEllipse(cnt)
                        if (ellipse[1][0]*ellipse[1][1]) > 625:
                            cv2.ellipse(frame,ellipse,(0,255,0),2)
                            center = (int(ellipse[0][0]),int(ellipse[0][1]))
                            cv2.circle(frame,center,2,(0,0,255))
                            if previouspos == None:
                                previouspos = center
                                previoustime = dt
                            else:
                                d = Euclid(previouspos,center)
                                previouspos = center
                                tottime+=(dt-previoustime)
                                totdist+=d
                                previoustime = dt

                cv2.imshow('frame',frame)
                #cv2.imshow('avg',res)
                #cv2.imshow('diff',diff)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

    cap.release()
    cv2.destroyAllWindows()
    
    v = totdist/tottime
    write.append(v)
    write.append(totdist)
    return write
    
def Euclid(pos1,pos2):
    cc = float(((pos1[0]-pos2[0])**2)+((pos1[1]-pos2[1])**2))
    c = cc**0.5
    return c
    
def timeConvert(t):
    ts = t.split(':')
    seconds = (float(ts[0])*360)+(float(ts[1])*60)+float(ts[2])
    return seconds
    
def timeReader(name):
    name = name[:-4]
    name+='.csv'
    timeread = csv.reader(open(os.path.join(path,name),'rb'))
    csvfile = []
    for row in timeread:
        csvfile.append(row)
    return csvfile

def findStarts(name):
    name = name[:-4]
    name+='marks.csv'
    read = csv.reader(open(os.path.join(path,name),'rb'))
    starts = []
    stops = []
    for row in read:
        starts.append(int(row[0]))
        try:
            stops.append(int(row[1]))
        except IndexError:
            pass
    return (starts,stops)

def writeIt():
    writer = csv.writer(open('roachAnalysis.csv', 'w'))
    for row in writeObj:
        writer.writerow(row)
        
def timeIt((starts,stops),csvfile,write):
    timestarts = []
    timestops = []
    for n in starts:
        timestarts.append(timeConvert(csvfile[n][1]))
    for n in stops:
        timestops.append(timeConvert(csvfile[n][1]))
    timestops.append(timeConvert(csvfile[-1][1]))
    tottime = 0
    for x in range(0,len(starts)):
        start = timestarts[x]
        stop = timestops[x]
        tottime+=float(stop-start)
    write.append(tottime)
    return write

writeObj = [['Name','TimeMoving','AvgSpeed','Distance']]
for root,dirs,files in os.walk(path):
    for f in files:
        if f[-4:] == '.avi':
            csvfile = timeReader(f)
            (starts,stops) = findStarts(f)
            write = [f[:-4]]
            w = timeIt((starts,stops),csvfile,write)
            w = review(f,(starts,stops),csvfile,write)
            writeObj.append(w)

writeIt()

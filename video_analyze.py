import cv2
from tkinter import *
from tkinter import filedialog
import PIL.Image, PIL.ImageTk
import numpy as np
import colorsys
import matplotlib
import csv

# turn these two global variables into settings in the GUI
# penalize movers who stop in the open
# use mask for something
# track speed

TRACKS = True

class CapturedVideo:
    def __init__(self, video=0):
        self.vid = cv2.VideoCapture(video)
        if not self.vid.isOpened():
            raise ValueError("Unable to open video source", video)
        width = self.vid.get(cv2.CAP_PROP_FRAME_WIDTH)
        height = self.vid.get(cv2.CAP_PROP_FRAME_HEIGHT)
        delta1 = 900. / width
        delta2 = 600. / height
        self.delta = min(delta1, delta2)
        self.width = int(width * self.delta)
        self.height = int(height * self.delta)
        # im = cv2.resize(im, (0, 0), fx=delta, fy=delta)

    def count_frames(self):
        return(self.vid.get(cv2.CAP_PROP_FRAME_COUNT))

    def get_frame(self, framenum):
        if self.vid.isOpened():
            self.vid.set(1, framenum)
            ret, frame = self.vid.read()
            if ret:
                # Return a boolean success flag and the current frame converted to BGR
                frame = cv2.resize(frame, (0, 0), fx = self.delta, fy = self.delta)
                return (ret, cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            else:
                return (ret, None)
        else:
            return (False, None)

    def __del__(self):
        if self.vid.isOpened():
            self.vid.release()

class DisplayApp:
    def __init__(self, root):
        self.AppParent = root
        self.AppParent.title('Video Analyzer')
        self.MASKMODE = StringVar()
        self.MASKMODE.set('N')
        self.MainFrame = Frame(self.AppParent)
        self.MainFrame.pack()
        self.makeframes()

    def makeframes(self):  # sets up grid for window
        self.ButtonFrame = Frame(self.MainFrame, padx=10, pady=10)
        self.ButtonFrame.grid(row=0, column=0)
        self.VideoFrame = Frame(self.MainFrame, padx=10, pady=10)
        self.VideoFrame.grid(row=0, column=1)
        self.NotificationFrame = Frame(self.MainFrame, padx=10, pady=10)
        self.NotificationFrame.grid(row=1, column=0)
        self.Notifications = Text(self.NotificationFrame, height=5, width=40)
        self.Notifications.grid(row=0, column=0)
        self.framenum = 0
        self.makebuttons()

    def makebuttons(self):  # makes buttons in ButtonFrame
        row = 0
        column = 0
        Button(self.ButtonFrame, text="Open Video File", command=self.openvideofile).grid(row=row, column=column)
        row += 1
        Button(self.ButtonFrame, text="Frame forward", command=self.frameforward).grid(row=row, column=column)
        row += 1
        Button(self.ButtonFrame, text="Frame backward", command=self.framebackward).grid(row=row, column=column)
        row += 1
        self.frameScaler = Scale(self.ButtonFrame, from_=0, to_=100, orient = HORIZONTAL, command = self.changeframe)
        self.frameScaler.grid(row=row, column=column)
        row += 1
        Radiobutton(self.ButtonFrame, text="Add to Mask", variable=self.MASKMODE, value='A', indicatoron=False).grid(row=row, column=column)
        row += 1
        Radiobutton(self.ButtonFrame, text="Remove from Mask", variable=self.MASKMODE, value='R', indicatoron=False).grid(row=row,
                                                                                                  column=column)
        row += 1
        Radiobutton(self.ButtonFrame, text="Don\'t Alter Mask", variable=self.MASKMODE, value='N', indicatoron=False).grid(row=row,
                                                                                                  column=column)
        row += 1
        self.toleranceScaler = Scale(self.ButtonFrame, from_=0, to_=100, orient=HORIZONTAL)
        self.toleranceScaler.grid(row=row, column=column)
        self.toleranceScaler.set(3)
        row += 1
        Button(self.ButtonFrame, text="Show Mask", command=self.showmask).grid(row=row, column=column)
        row += 1
        Button(self.ButtonFrame, text="Clear Mask", command=self.clearmask).grid(row=row, column=column)
        row += 1
        self.contourCounterEntry = Entry(self.ButtonFrame)
        self.contourCounterEntry.grid(row=row, column=column)
        self.contourCounterEntry.insert(0, '0')
        row +=1
        Button(self.ButtonFrame, text="Select Largest Masks", command=self.selectlargestmasks).grid(row=row, column=column)
        row += 1
        self.smoothScaler = Scale(self.ButtonFrame, from_=0, to_=100, orient=HORIZONTAL)
        self.smoothScaler.grid(row=row, column=column)
        self.smoothScaler.set(5)
        row += 1
        Button(self.ButtonFrame, text="Smooth Mask", command=self.smoothmask).grid(row=row, column=column)
        row += 1
        Button(self.ButtonFrame, text="Convex Smooth Mask", command=self.convexmask).grid(row=row, column=column)
        row += 1
        Label(self.ButtonFrame, text="Variables relating to analysis").grid(row=row,column=column)
        row += 1
        self.histEntry = Entry(self.ButtonFrame)
        self.histEntry.grid(row=row, column=column)
        self.histEntry.insert(0, '50')
        row += 1
        self.varEntry = Entry(self.ButtonFrame)
        self.varEntry.grid(row=row, column=column)
        self.varEntry.insert(0, '20')
        row += 1
        #Button(self.ButtonFrame, text="Make Stabilized Video", command=self.makestabilizedvideo).grid(row=row, column=column)
        #row += 1
        #Button(self.ButtonFrame, text="Analyze DOF", command=self.findmoversDOF).grid(row=row, column=column)
        #column += 1
        Button(self.ButtonFrame, text="Analyze MOG", command=self.findmoversMOG).grid(row=row, column=column)
        #column -= 1
        row += 1

    def openvideofile(self):
        path = filedialog.askopenfilename()
        self.vid = CapturedVideo(path)
        self.canvas = Canvas(self.VideoFrame, width = self.vid.width, height = self.vid.height)
        self.canvas.grid(row=0, column = 0)
        self.canvas.bind("<Button-1>", self.clickhandler)  # attach clickhandler method to clicks on canvas
        self.mask = np.zeros((self.vid.height, self.vid.width), dtype=np.uint8)
        length = self.vid.count_frames()
        self.frameScaler.config(to_=length)
        self.framenum = 0
        self.showframe()

    def showframe(self):
        ret, frame = self.vid.get_frame(self.framenum)
        if ret:  # ret is a True/False for frame existence
            self.photo = PIL.ImageTk.PhotoImage(image=PIL.Image.fromarray(frame))
            self.canvas.create_image(0, 0, image=self.photo, anchor=NW)
        self.Notifications.insert((1.0), 'Showing frame ' + str(self.framenum) + '\n')

    def frameforward(self):
        self.framenum += 1
        self.showframe()

    def framebackward(self):
        self.framenum -= 1
        self.showframe()

    def changeframe(self, x):
        # x is the number in the scale widget calling this
        self.framenum = int(x)
        self.showframe()

    def clickhandler(self, event):
        mode = self.MASKMODE.get()
        if mode == 'N':
            pass
        else:
            x = int(self.canvas.canvasx(event.x))
            y = int(self.canvas.canvasy(event.y))
            ret, frame = self.vid.get_frame(self.framenum)
            colorarray = np.asarray(frame)
            R = colorarray[y][x][0]
            G = colorarray[y][x][1]
            B = colorarray[y][x][2]
            if mode == 'A':
                self.addtomask(R,G,B)
            elif mode == 'R':
                self.removefrommask(R, G, B)

    def addtomask(self, R, G, B):
        # convert RGB to HSV
        R = float(R/255.)
        G = float(G / 255.)
        B = float(B / 255.)
        HSV = colorsys.rgb_to_hsv(R, G, B)
        H = HSV[0]
        # get frame, get tolerance
        ret, frame = self.vid.get_frame(self.framenum)
        frame = np.asarray(frame)
        tolerance = float(self.toleranceScaler.get())/100.
        # make an HSV frame, slice off H part
        frame = frame / 255.  # converts scale for color conversion
        newframe = matplotlib.colors.rgb_to_hsv(frame)[:,:,0]
        # get array of places where frame DOES NOT meet criteria - too difficult to get a match on two categories
        arr = np.where(newframe > H + tolerance, 255, 0)
        arr2 = np.where(newframe < H - tolerance, 255, 0)
        # np,maximum combines arrays and puts in the maximum number - e.g., all 255s that are present
        bigarr = np.maximum(arr, arr2)
        # flip opposite mask
        reversearr = np.where(bigarr > 250, 0, 255)
        # combine with old mask
        self.mask = np.maximum(reversearr, self.mask)
        self.showmask()
        self.Notifications.insert((1.0), 'Added to mask' + '\n')

    def removefrommask(self, R, G, B):
        # convert RGB to HSV
        R = float(R / 255.)
        G = float(G / 255.)
        B = float(B / 255.)
        HSV = colorsys.rgb_to_hsv(R, G, B)
        H = HSV[0]
        # get frame, get tolerance
        ret, frame = self.vid.get_frame(self.framenum)
        frame = np.asarray(frame)
        tolerance = float(self.toleranceScaler.get()) / 100.
        # make an HSV frame, slice off H part
        frame = frame / 255.  # converts scale for color conversion
        newframe = matplotlib.colors.rgb_to_hsv(frame)[:, :, 0]
        # get array of places where frame DOES NOT meet criteria - too difficult to get a match on two categories
        arr = np.where(newframe > H + tolerance, 255, 0)
        arr2 = np.where(newframe < H - tolerance, 255, 0)
        # np,maximum combines arrays and puts in the maximum number - e.g., all 255s that are present
        bigarr = np.maximum(arr, arr2)
        self.mask = np.minimum(bigarr, self.mask)
        self.showmask()
        self.Notifications.insert((1.0), 'Removed from mask' + '\n')

    def showmask(self):
        reversemask = np.where(self.mask > 250, 0, 255)
        ret, frame = self.vid.get_frame(self.framenum)
        frame = np.asarray(frame)
        #redchannel = frame[:,:,2]
        #redchannel = redchannel + redmask
        frame[:,:,2] = np.minimum(reversemask,frame[:,:,2])
        frame[:, :, 1] = np.minimum(reversemask, frame[:, :, 1])
        frame[:, :, 0] = np.minimum(reversemask, frame[:, :, 0])
        self.photo = PIL.ImageTk.PhotoImage(image=PIL.Image.fromarray(frame))
        self.canvas.create_image(0, 0, image=self.photo, anchor=NW)

    def clearmask(self):
        self.mask = np.zeros(self.mask.shape, dtype=np.uint8)
        self.Notifications.insert((1.0), 'Cleared' + '\n')

    def selectlargestmasks(self):
        # get number of masks to keep
        n = int(self.contourCounterEntry.get())
        # get contours, order by size
        ret, thresh = cv2.threshold(self.mask.astype(np.uint8), 127, 255, 0)
        contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        sortcontours = sorted(contours, key=cv2.contourArea, reverse=True)
        # make a blank mask
        tempmask = np.zeros(self.mask.shape, dtype=np.uint8)
        # get largest contours, draw them on tempmask
        try:
            for x in range(0, n):
                c = sortcontours[x]
                tempmask = cv2.drawContours(tempmask, [c], 0, (255), -1)
        except IndexError:  # if there aren't that many contours
            pass
        self.mask = tempmask
        self.showmask()
        self.Notifications.insert((1.0), 'Selected ' + str(n) + ' largest objects\n')

    def smoothmask(self):
        # get contours
        ret, thresh = cv2.threshold(self.mask.astype(np.uint8), 127, 255, 0)
        contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        # make a blank mask
        tempmask = np.zeros(self.mask.shape, dtype=np.uint8)
        # write smoothed contours onto tempmask
        smoother = float(self.smoothScaler.get()/1000)
        for c in contours:
            epsilon = 0.005 * cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, epsilon, True)
            tempmask = cv2.drawContours(tempmask, [approx], 0, (255), -1)
        self.mask = tempmask
        self.showmask()
        self.Notifications.insert((1.0), 'Smoothed\n')

    def convexmask(self):
        # get contours
        ret, thresh = cv2.threshold(self.mask.astype(np.uint8), 127, 255, 0)
        contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        # make a blank mask
        tempmask = np.zeros(self.mask.shape, dtype=np.uint8)
        # write smoothed contours onto tempmask
        for c in contours:
            approx = cv2.convexHull(c)
            tempmask = cv2.drawContours(tempmask, [approx], 0, (255), -1)
        self.mask = tempmask
        self.showmask()
        self.Notifications.insert((1.0), 'Convex smoothed\n')

    def findmoversDOF(self):
        ret, frame1 = self.vid.get_frame(0)
        prvs = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
        hsv = np.zeros_like(frame1)
        hsv[..., 1] = 255
        # iterate through frames
        for x in range(0, int(self.vid.count_frames())):
            ret, f = self.vid.get_frame(x)
            next = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
            flow = cv2.calcOpticalFlowFarneback(prvs, next, None, 0.5, 3, 15, 3, 5, 1.2, 0)
            mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
            hsv[..., 0] = ang * 180 / np.pi / 2
            hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
            rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
            # display
            self.photo = PIL.ImageTk.PhotoImage(image=PIL.Image.fromarray(rgb.astype(np.uint8)))
            self.canvas.create_image(0, 0, image=self.photo, anchor=NW)
            self.canvas.update_idletasks()
            self.Notifications.insert((1.0), 'Frame ' + str(x) + '\n')

    def findmoversMOG(self):
        # grab MOG weights
        hist = int(self.histEntry.get())
        var = int(self.varEntry.get())
        fgbg = cv2.createBackgroundSubtractorMOG2(history=hist, varThreshold=var, detectShadows=0)
        # set up list to take positions of movers
        moverslist = []
        # iterate through frames
        for x in range(0, 8700): # int(self.vid.count_frames())):
            ret, f = self.vid.get_frame(x)
            fgmask = fgbg.apply(f)
            # analyze
            contours, hierarchy = cv2.findContours(fgmask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            bigconts = []
            templist = [] # holds mover points for this iteration
            for cont in contours:
                if cv2.contourArea(cont) > 70:
                    bigconts.append(cont)
                    M = cv2.moments(cont)
                    cX = int(M["m10"] / M["m00"])
                    cY = int(M["m01"] / M["m00"])
                    templist.append([cX, cY])
            moverslist.append(templist)
            # make a blank mask
            tempmask = np.zeros(self.mask.shape, dtype=np.uint8)
            # write big contours onto tempmask
            for c in bigconts:
                tempmask = cv2.drawContours(tempmask, [c], 0, (255), -1)
            tempmask = np.dstack((tempmask, tempmask, tempmask))
            for m in moverslist[-1]: # most recent movers
                cv2.circle(tempmask, (m[0], m[1]), 5, (255,0,0), -1)
            # display
            self.photo = PIL.ImageTk.PhotoImage(image=PIL.Image.fromarray(tempmask.astype(np.uint8)))
            self.canvas.create_image(0, 0, image=self.photo, anchor=NW)
            self.canvas.update_idletasks()
            self.Notifications.insert((1.0), 'Frame ' + str(x) + '\n')
        tracks = self.makeTracks(moverslist)
        if TRACKS:
            for t in tracks:
                for x in range(1, len(t)-1):
                    (xt, yt) = t[x-1]
                    (xt1, yt1) = t[x]
                    cv2.line(tempmask, (xt,yt),(xt1, yt1), (0, 0, 255), 1)
            # display
            self.photo = PIL.ImageTk.PhotoImage(image=PIL.Image.fromarray(tempmask.astype(np.uint8)))
            self.canvas.create_image(0, 0, image=self.photo, anchor=NW)
            self.canvas.update_idletasks()
            cv2.imwrite('tracks.png', tempmask)
            cv2.imwrite('screenshot.png', f)
        self.writeout(tracks)

    def writeout(self, tracks):
        with open('Tracks.csv', 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            for track in tracks:
                writer.writerow([len(track)])

    def makeTracks(self, moverslist):
        tracks = []
        for m in moverslist: # go through frames
            for mover in m:
                [cX,cY] = mover
                d = 200
                track = None
                if len(tracks) <= 0:
                    tracks.append([[cX, cY]])
                else:
                    for x in range(0, len(tracks)-1):
                        t = tracks[x][-1]
                        dist = ((cX - t[0])**2)+((cY-t[1])**2)
                        if dist < d:
                            track = x
                            d = dist
                    if track != None:
                        tracks[track].append([cX, cY])
                    else:
                        tracks.append([[cX, cY]])
        largetracks = []
        for t in tracks:
            if len(t) > 10:
                largetracks.append(t)
        return largetracks


    def makestabilizedvideo(self):
        # define output codec
        fourcc = cv2.VideoWriter_fourcc(*'MJPG')
        # Set up output video
        out = cv2.VideoWriter('stabilized_video.mp4', fourcc, 29, (self.vid.width, self.vid.height))
        # grab first frame
        ret, f = self.vid.get_frame(0)
        gray_previous = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
        # transformation array
        transforms = np.zeros((int(self.vid.count_frames()) - 1, 3), np.float32)
        for x in range(0, int(self.vid.count_frames()-2)):
            prev_pts = cv2.goodFeaturesToTrack(gray_previous,
                                     maxCorners=200,
                                     qualityLevel=0.01,
                                     minDistance=30,
                                     blockSize=3)

            # next frame
            ret, f = self.vid.get_frame(x)
            gray_current = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
            # optical flow
            curr_pts, status, err = cv2.calcOpticalFlowPyrLK(gray_previous, gray_current, prev_pts, None)
            # Sanity check
            assert prev_pts.shape == curr_pts.shape
            # Filter only valid points
            idx = np.where(status == 1)[0]
            prev_pts = prev_pts[idx]
            curr_pts = curr_pts[idx]

            # Find transformation matrix
            m = cv2.estimateAffinePartial2D(prev_pts.astype(np.float32), curr_pts.astype(np.float32))

            # Extract translation
            dx = m[0, 2]
            dy = m[1, 2]

            # Extract rotation angle
            da = np.arctan2(m[1, 0], m[0, 0])

            # Store transformation
            transforms[x] = [dx, dy, da]

            # Move to next frame
            gray_previous = gray_current

        # Compute trajectory using cumulative sum of transformations
        trajectory = np.cumsum(transforms, axis=0)
        smoothed_trajectory = self.smooth(trajectory)

        # Calculate difference in smoothed_trajectory and trajectory
        difference = smoothed_trajectory - trajectory

        # Calculate newer transformation array
        transforms_smooth = transforms + difference

        # make new, stabilized video
        for x in range(0, int(self.vid.count_frames() - 2)):
            ret, f = self.vid.get_frame(x)

            # Extract transformations from the new transformation array
            dx = transforms_smooth[x, 0]
            dy = transforms_smooth[x, 1]
            da = transforms_smooth[x, 2]

            # Reconstruct transformation matrix accordingly to new values
            m = np.zeros((2, 3), np.float32)
            m[0, 0] = np.cos(da)
            m[0, 1] = -np.sin(da)
            m[1, 0] = np.sin(da)
            m[1, 1] = np.cos(da)
            m[0, 2] = dx
            m[1, 2] = dy

            # Apply affine wrapping to the given frame
            frame_stabilized = cv2.warpAffine(f, m, (self.vid.width, self.vid.height))

            # Fix border artifacts
            frame_stabilized = self.fixBorder(frame_stabilized)

            # Write the frame to the file
            frame_out = cv2.hconcat([f, frame_stabilized])
            out.write(frame_out)

    def movingAverage(self, curve, radius):
        window_size = 2 * radius + 1
        # Define the filter
        f = np.ones(window_size) / window_size
        # Add padding to the boundaries
        curve_pad = np.lib.pad(curve, (radius, radius), 'edge')
        # Apply convolution
        curve_smoothed = np.convolve(curve_pad, f, mode='same')
        # Remove padding
        curve_smoothed = curve_smoothed[radius:-radius]
        # return smoothed curve
        return curve_smoothed

    def smooth(self, trajectory):
        smoothed_trajectory = np.copy(trajectory)
        # Filter the x, y and angle curves
        for i in range(3):
            smoothed_trajectory[:, i] = self.movingAverage(trajectory[:, i], radius=2)

        return smoothed_trajectory

    def fixBorder(self, frame):
        s = frame.shape
        # Scale the image 4% without moving the center
        T = cv2.getRotationMatrix2D((s[1] / 2, s[0] / 2), 0, 1.04)
        frame = cv2.warpAffine(frame, T, (s[1], s[0]))
        return frame



root = Tk()
displayapp = DisplayApp(root)
root.mainloop()

import cv2
from tkinter import *
from tkinter import filedialog
import PIL.Image, PIL.ImageTk
import numpy as np
import colorsys
import matplotlib
import csv

tracks_enabled = True


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
        self.show_mode = StringVar()
        self.show_mode.set('M')
        self.background_subtractor = StringVar()
        self.background_subtractor.set('MOG')
        self.shadows = 0
        self.start_after_hist = 1
        self.MainFrame = Frame(self.AppParent)
        self.MainFrame.pack()
        self.make_frames()

    def make_frames(self):  # sets up grid for window
        self.ButtonFrame = Frame(self.MainFrame, padx=10, pady=10)
        self.ButtonFrame.grid(row=0, column=0)
        self.VideoFrame = Frame(self.MainFrame, padx=10, pady=10)
        self.VideoFrame.grid(row=0, column=1)
        self.NotificationFrame = Frame(self.MainFrame, padx=10, pady=10)
        self.NotificationFrame.grid(row=1, column=0)
        self.Notifications = Text(self.NotificationFrame, height=5, width=40)
        self.Notifications.grid(row=0, column=0)
        self.framenum = 0
        self.make_buttons()

    def make_buttons(self):  # makes buttons in ButtonFrame
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

        Radiobutton(self.ButtonFrame, text="Show all motion", variable=self.show_mode, value='M').grid(
            row=row, column=column)
        row += 1
        Radiobutton(self.ButtonFrame, text="Show size-filtered motion", variable=self.show_mode, value='F').grid(row=row,
                                            column=column)
        row += 1
        Radiobutton(self.ButtonFrame, text="Show unprocessed frame", variable=self.show_mode, value='R').grid(row=row,
                                            column=column)
        row += 1
        Radiobutton(self.ButtonFrame, text="Show normal frame with size-filtered motion", variable=self.show_mode, value='Q').grid(row=row,
                                                                                                              column=column)
        row += 1
        Label(self.ButtonFrame, text='').grid(row=row, column=column)
        row += 1

        Checkbutton(self.ButtonFrame, text='Mark shadows', variable=self.shadows, onvalue=1, offvalue=0).grid(row=row,
                                                                                                              column=column)
        row += 1

        Checkbutton(self.ButtonFrame, text='Track motion after [history] number of frames', variable=self.start_after_hist, onvalue=1, offvalue=0).grid(row=row,
                                                                                                              column=column)
        row += 1
        Label(self.ButtonFrame, text='').grid(row=row, column=column)
        row += 1

        Radiobutton(self.ButtonFrame, text="Use MOG background subtraction", variable=self.background_subtractor, value='MOG').grid(row=row,
                                                                                                              column=column)
        row += 1
        Radiobutton(self.ButtonFrame, text="Use KNN background subtraction", variable=self.background_subtractor,
                    value='KNN').grid(row=row,
                                    column=column)
        row += 1
        Label(self.ButtonFrame, text='').grid(row=row,column=column)
        row += 1

        Label(self.ButtonFrame, text="Variables relating to analysis").grid(row=row,column=column)
        row += 1
        Label(self.ButtonFrame, text="History length").grid(row=row, column=column)
        row += 1
        self.histEntry = Entry(self.ButtonFrame)
        self.histEntry.grid(row=row, column=column)
        self.histEntry.insert(0, '50')
        row += 1
        Label(self.ButtonFrame, text="Threshold").grid(row=row, column=column)
        row += 1
        self.varEntry = Entry(self.ButtonFrame)
        self.varEntry.grid(row=row, column=column)
        self.varEntry.insert(0, '20')
        row += 1
        Label(self.ButtonFrame, text="Size filter").grid(row=row, column=column)
        row += 1
        self.size_filter = Entry(self.ButtonFrame)
        self.size_filter.grid(row=row, column=column)
        self.size_filter.insert(0, '70')
        row += 1
        Button(self.ButtonFrame, text="Analyze", command=self.movers_background_subtract).grid(row=row, column=column)
        row += 1
        Button(self.ButtonFrame, text="Test", command=self.test).grid(row=row, column=column)
        row += 1

    def openvideofile(self):
        path = filedialog.askopenfilename()
        self.vid = CapturedVideo(path)
        self.canvas = Canvas(self.VideoFrame, width = self.vid.width, height = self.vid.height)
        self.canvas.grid(row=0, column = 0)
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

    def movers_background_subtract(self):
        # grab weights
        hist = int(self.histEntry.get())
        var = int(self.varEntry.get())
        size_filter = int(self.size_filter.get())
        if self.background_subtractor.get() == 'MOG':
            fgbg = cv2.createBackgroundSubtractorMOG2(history=hist, varThreshold=var, detectShadows=self.shadows)
        else:
            fgbg = cv2.createBackgroundSubtractorKNN(history=hist, dist2Threshold=var, detectShadows=self.shadows)
        # set up list to take positions of movers
        moverslist = []
        # iterate through frames
        start = 0
        if self.start_after_hist == 1:
            start = hist
        for x in range(start, int(self.vid.count_frames())):
            ret, f = self.vid.get_frame(x)
            fgmask = fgbg.apply(f)
            # analyze
            contours, hierarchy = cv2.findContours(fgmask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            bigconts = []
            templist = []  # holds mover points for this iteration
            for cont in contours:
                if cv2.contourArea(cont) > size_filter:
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
            for m in moverslist[-1]:  # most recent movers
                cv2.circle(tempmask, (m[0], m[1]), 5, (255, 0, 0), -1)
            # display
            if self.show_mode.get() == 'R':
                self.photo = PIL.ImageTk.PhotoImage(image=PIL.Image.fromarray(f))
            elif self.show_mode.get() == 'M':
                self.photo = PIL.ImageTk.PhotoImage(image=PIL.Image.fromarray(fgmask))
            elif self.show_mode.get() == 'F':
                self.photo = PIL.ImageTk.PhotoImage(image=PIL.Image.fromarray(tempmask.astype(np.uint8)))
            else:  # must be normal frame with size filtered motion
                for m in moverslist[-1]:
                    cv2.circle(f, (m[0], m[1]), size_filter + 5, (255, 0, 0))
                self.photo = PIL.ImageTk.PhotoImage(image=PIL.Image.fromarray(f))
            self.canvas.create_image(0, 0, image=self.photo, anchor=NW)
            self.canvas.update_idletasks()
            self.Notifications.insert((1.0), 'Frame ' + str(x) + '\n')
        tracks = self.makeTracks(moverslist)
        if tracks_enabled:
            for t in tracks:
                for x in range(1, len(t)-1):
                    (xt, yt) = t[x-1]
                    (xt1, yt1) = t[x]
                    cv2.line(tempmask, (xt,yt),(xt1, yt1), (0, 0, 255), 1)
            # write out
            cv2.imwrite('tracks.png', tempmask)
            try:
                cv2.imwrite('screenshot.png', f)
            except cv2.error:
                print('Could not save screenshot')
        self.writeout(tracks)

    def test(self):
        hsv = np.zeros_like(self.vid.get_frame(0)[1])
        hsv[..., 1] = 255
        start = 1
        for x in range(start, int(self.vid.count_frames())):
            ret_old, f_old = self.vid.get_frame(x - 1)
            ret_new, f_new = self.vid.get_frame(x)
            f_old = cv2.cvtColor(f_old, cv2.COLOR_BGR2GRAY)
            f_new = cv2.cvtColor(f_new, cv2.COLOR_BGR2GRAY)
            flow = cv2.calcOpticalFlowFarneback(f_old, f_new, None, 0.5, 3, 15, 3, 5, 1.2, 0)
            mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
            hsv[..., 0] = ang * 180 / np.pi / 2
            hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
            bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
            self.photo = PIL.ImageTk.PhotoImage(image=PIL.Image.fromarray(bgr))
            self.canvas.create_image(0, 0, image=self.photo, anchor=NW)
            self.canvas.update_idletasks()
            self.Notifications.insert((1.0), 'Frame ' + str(x) + '\n')

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



root = Tk()
displayapp = DisplayApp(root)
root.mainloop()

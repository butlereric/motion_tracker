import cv2
from tkinter import *
from tkinter import filedialog
from tkinter import ttk
import PIL.Image, PIL.ImageTk
import numpy as np
import csv
import os
import pandas as pd

tracks_enabled = True

try:
    os.mkdir('Output')
except FileExistsError:
    pass

try:
    os.mkdir('Masks')
except FileExistsError:
    pass


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
        self.menu = Menu(root)
        self.show_mode = StringVar()
        self.show_mode.set('M')
        self.background_subtractor = StringVar()
        self.background_subtractor.set('MOG')
        self.shadows = IntVar()
        self.shadows.set(0)
        self.start_after_hist = IntVar()
        self.start_after_hist.set(1)
        self.stopped = IntVar()
        self.stopped.set(0)
        self.hist_text = StringVar()
        self.hist_text.set('History Length')
        self.thresh_text = StringVar()
        self.thresh_text.set('Threshold')
        self.mask_type = StringVar()
        self.mask_type.set('Hole')
        self.show_mask = IntVar()
        self.show_mask.set(0)
        self.mask_intensity_level = 0.7
        self.write_to_one_file = IntVar()
        self.write_to_one_file.set(0)
        self.folder_name = ''
        self.MainFrame = Frame(self.AppParent)
        self.MainFrame.pack()
        self.click_locations = []
        self.make_frames()
        self.make_menus()
        self.make_or_reset_persistent_data()

    def make_or_reset_persistent_data(self):
        self.persistent_data = pd.DataFrame({'Time_of_track': [], 'Avg_area_of_mover': [], 'Distance_of_track': []})

    def make_frames(self):  # sets up grid for window
        self.tabbed_frame = ttk.Notebook(self.MainFrame)
        self.tabbed_frame.grid(row=0, column=0)
        self.video_view_frame = ttk.Frame(self.tabbed_frame)
        self.mask_frame = ttk.Frame(self.tabbed_frame)
        self.morph_filter_frame = ttk.Frame(self.tabbed_frame)
        self.motion_detect_options_frame = ttk.Frame(self.tabbed_frame)
        for tab, label in [[self.video_view_frame, 'Video View'], [self.mask_frame, 'Mask Editing'],
                           [self.morph_filter_frame, 'Filter'], [self.motion_detect_options_frame, 'Detection Options']]:
            self.tabbed_frame.add(tab, text=label)
        self.tabbed_frame.grid(row=0, column=0)
        self.VideoFrame = Frame(self.MainFrame, padx=10, pady=10)
        self.VideoFrame.grid(row=0, column=1)
        self.VideoFrame.bind_all('<Button-1>', self.canvas_click_event)
        self.NotificationFrame = Frame(self.MainFrame, padx=10, pady=10)
        self.NotificationFrame.grid(row=1, column=0)
        self.Notifications = Text(self.NotificationFrame, height=5, width=40)
        self.Notifications.grid(row=0, column=0)
        self.framenum = 0
        self.mask = np.array([])
        self.make_buttons()

    def make_buttons(self):  # makes buttons in ButtonFrame
        # video view frame
        row = 0
        column = 0
        Button(self.video_view_frame, text="Frame forward", command=self.frameforward).grid(row=row, column=column)
        row += 1
        Button(self.video_view_frame, text="Frame backward", command=self.framebackward).grid(row=row, column=column)
        row += 1
        self.frameScaler = Scale(self.video_view_frame, from_=0, to_=100, orient = HORIZONTAL, command = self.changeframe)
        self.frameScaler.grid(row=row, column=column)
        row += 1

        Radiobutton(self.video_view_frame, text="Show all motion", variable=self.show_mode, value='M').grid(
            row=row, column=column)
        row += 1
        Radiobutton(self.video_view_frame, text="Show size-filtered motion", variable=self.show_mode, value='F').grid(row=row,
                                            column=column)
        row += 1
        Radiobutton(self.video_view_frame, text="Show unprocessed frame", variable=self.show_mode, value='R').grid(row=row,
                                            column=column)
        row += 1
        Radiobutton(self.video_view_frame, text="Show normal frame with size-filtered motion", variable=self.show_mode, value='Q').grid(row=row,
                                                                                                              column=column)
        row += 1
        Label(self.video_view_frame, text='').grid(row=row, column=column)
        row += 1

        Checkbutton(self.video_view_frame, text='Mark shadows', variable=self.shadows, onvalue=1, offvalue=0).grid(row=row,
                                                                                                              column=column)
        row += 1

        Checkbutton(self.video_view_frame, text='Track motion after [history] number of frames', variable=self.start_after_hist, onvalue=1, offvalue=0).grid(row=row,
                                                                                                              column=column)
        row += 1

        Label(self.video_view_frame, text='Process this many frames (-1 means "all")').grid(row=row, column=column)
        row += 1
        self.framesEntry = Entry(self.video_view_frame)
        self.framesEntry.grid(row=row, column=column)
        self.framesEntry.insert(0, '-1')
        row += 1
        Label(self.video_view_frame, text='').grid(row=row, column=column)
        row += 1

        Radiobutton(self.video_view_frame, text="Use MOG background subtraction", variable=self.background_subtractor,
                    value='MOG').grid(row=row,
                                                                                                              column=column)
        row += 1
        Radiobutton(self.video_view_frame, text="Use KNN background subtraction", variable=self.background_subtractor,
                    value='KNN').grid(row=row,
                                    column=column)
        row += 1
        Radiobutton(self.video_view_frame, text="Use optical flow", variable=self.background_subtractor,
                    value='Flow').grid(row=row,
                                      column=column)
        row += 1
        Label(self.video_view_frame, text='').grid(row=row,column=column)
        row += 1

        Button(self.video_view_frame, text="Analyze", command=self.analyze).grid(row=row, column=column)
        row += 1
        self.stopped_button = Checkbutton(self.video_view_frame, text='Stop running analysis', variable=self.stopped,
                                          onvalue=1, offvalue=0, indicatoron=False)
        self.stopped_button.grid(row=row, column=column)
        row += 1
        Checkbutton(self.video_view_frame, text='Write everything to one file', variable=self.write_to_one_file, onvalue=1, offvalue=0).grid(
            row=row,
            column=column)
        row += 1

        # motion detect options frame
        Label(self.motion_detect_options_frame, text="Variables relating to analysis").grid(row=row,column=column)
        row += 1
        self.hist_label = Label(self.motion_detect_options_frame, textvariable=self.hist_text)
        self.hist_label.grid(row=row, column=column)
        row += 1
        self.histEntry = Entry(self.motion_detect_options_frame)
        self.histEntry.grid(row=row, column=column)
        self.histEntry.insert(0, '500')
        row += 1
        self.thresh_label = Label(self.motion_detect_options_frame, textvariable=self.thresh_text)
        self.thresh_label.grid(row=row, column=column)
        row += 1
        self.varEntry = Entry(self.motion_detect_options_frame)
        self.varEntry.grid(row=row, column=column)
        self.varEntry.insert(0, '20')
        row += 1

        # morph filter frame
        Label(self.morph_filter_frame, text="Minimum size filter").grid(row=row, column=column)
        row += 1
        self.size_filter_min = Entry(self.morph_filter_frame)
        self.size_filter_min.grid(row=row, column=column)
        self.size_filter_min.insert(0, '3')
        row += 1
        Label(self.morph_filter_frame, text="Maximum size filter").grid(row=row, column=column)
        row += 1
        self.size_filter_max = Entry(self.morph_filter_frame)
        self.size_filter_max.grid(row=row, column=column)
        self.size_filter_max.insert(0, '200')
        row += 1

        # mask filter frame
        Button(self.mask_frame, text="Fill mask with black", command=self.fill_mask_black).grid(row=row, column=column)
        row += 1
        Button(self.mask_frame, text="Fill mask with white", command=self.fill_mask_white).grid(row=row, column=column)
        row += 1
        self.mask_intensity = Scale(self.mask_frame, from_=0, to_=100, orient=HORIZONTAL, command=self.change_mask_intensity)
        self.mask_intensity.grid(row=row, column=column)
        self.mask_intensity.set(self.mask_intensity_level * 100)
        row += 1
        Radiobutton(self.mask_frame, text="Cut a hole", variable=self.mask_type,
                    value='Hole', command=self.clear_click_locations).grid(row=row,
                                       column=column)
        row += 1
        Radiobutton(self.mask_frame, text="Draw a blob", variable=self.mask_type,
                    value='Blob', command=self.clear_click_locations).grid(row=row,
                                       column=column)
        row += 1
        Button(self.mask_frame, text="New mask object", command=self.clear_click_locations).grid(row=row, column=column)
        row += 1
        Button(self.mask_frame, text="Output Separate Mask File/s", command=self.output_separate_masks).grid(row=row, column=column)
        row += 1

    def make_menus(self):
        file = Menu(self.menu, tearoff=0)
        self.menu.add_cascade(label='File', menu=file)
        file.add_command(label='Open Video', command=self.openvideofile)
        file.add_command(label='Make Masks from Video Folder', command=self.make_masks_for_folder)
        file.add_command(label='Analyze Video Folder', command=self.open_video_folder)
        file.add_command(label='Quit', command=root.destroy)

        root.config(menu=self.menu)

    def clear_frames(self):
        self.show_mask.set(0)
        self.click_locations = []
        for frame in [self.morph_filter_frame, self.video_view_frame, self.motion_detect_options_frame, self.mask_frame]:
            frame.grid_forget()

    def show_morph_filters(self):
        self.clear_frames()
        self.morph_filter_frame.grid(row=0, column=0)

    def show_video_frame(self):
        self.clear_frames()
        self.video_view_frame.grid(row=0, column=0)

    def show_mask_frame(self):
        self.clear_frames()
        self.show_mask.set(1)
        self.mask_frame.grid(row=0, column=0)
        if self.show_mask.get() == 0:
            self.make_mask()
        self.showframe()

    def show_motion_detect_options(self):
        self.clear_frames()
        self.motion_detect_options_frame.grid(row=0, column=0)

    def find_movie_files_in_folder(self, path):
        files_list = []
        for root, dirs, files in os.walk(path):
            for f in files:
                if f.split('.')[1] in ['mp4', 'MP4', 'mov', 'MOV']:
                    files_list.append([os.path.join(root, f), f])
        return files_list

    def open_video_folder(self):
        folder = filedialog.askdirectory()
        self.folder_name = folder.split('/')[-1]
        videos = self.find_movie_files_in_folder(folder)
        print(videos)

    def make_masks_for_folder(self):
        folder = filedialog.askdirectory()
        videos = self.find_movie_files_in_folder(folder)
        for vid_path, vid_name in videos:
            vid = CapturedVideo(vid_path)
            mask = vid.get_frame(0)
            name = vid_name.split('.')[0]
            cv2.imwrite('Masks/' + name + '.png', mask)


    def openvideofile(self):
        path = filedialog.askopenfilename()
        self.vid_name = os.path.basename(path).split('.')[0]
        self.vid = CapturedVideo(path)
        self.canvas = Canvas(self.VideoFrame, width = self.vid.width, height = self.vid.height)
        self.canvas.grid(row=0, column=0)
        self.mask = np.multiply(np.ones((self.vid.height, self.vid.width), dtype=np.uint8), 255)
        length = self.vid.count_frames()
        self.frameScaler.config(to_=length)
        self.framenum = 0
        self.read_in_vars()
        self.read_in_mask_files()
        self.showframe()

    def showframe(self):
        ret, frame = self.vid.get_frame(self.framenum)
        if ret:  # ret is a True/False for frame existence
            if self.show_mask.get() == 1:  # showing mask
                dark_frame = np.multiply(frame, self.mask_intensity_level)
                mask = cv2.bitwise_and(dark_frame, dark_frame, mask=cv2.bitwise_not(self.mask))
                frame = cv2.bitwise_and(frame, frame, mask=self.mask)
                frame = frame + mask
            self.photo = PIL.ImageTk.PhotoImage(image=PIL.Image.fromarray(frame.astype(np.uint8)))
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

    def analyze(self):
        self.write_vars_to_file()
        if self.background_subtractor.get() == 'Flow':  # use optical flow
            self.optical_flow()
        else:
            self.movers_background_subtract()

    def make_mask(self):
        if self.mask.shape == self.vid.get_frame(0)[1].shape:
            pass
        else:
            self.mask = np.multiply(np.ones((self.vid.height, self.vid.width), dtype=np.uint8), 255)
            self.showframe()

    def fill_mask_black(self):
        self.mask = np.zeros((self.vid.height, self.vid.width), dtype=np.uint8)
        self.show_mask.set(1)
        self.showframe()

    def fill_mask_white(self):
        self.mask = np.multiply(np.ones((self.vid.height, self.vid.width), dtype=np.uint8), 255)
        self.show_mask.set(1)
        self.showframe()

    def change_mask_intensity(self, x):
        self.mask_intensity_level = (100 - int(x)) / 100
        self.showframe()

    def clear_click_locations(self):
        self.click_locations = []

    def canvas_click_event(self, event):
        if str(type(event.widget)) == "<class 'tkinter.Canvas'>":  # only trigger when click is on canvas
            if self.show_mask.get() == 1:  # modifying mask
                self.click_locations.append((event.x, event.y))
                if len(self.click_locations) > 1:
                    pts = np.array(self.click_locations, np.int32)
                    pts = pts.reshape((-1, 1, 2))
                    if self.mask_type.get() == 'Hole':
                        self.mask = cv2.fillPoly(self.mask, [pts], 255)
                    else:
                        self.mask = cv2.fillPoly(self.mask, [pts], 0)
                    self.showframe()

    def output_separate_masks(self):
        ret, frame = self.vid.get_frame(self.framenum)
        if ret:  # ret is a True/False for frame existence
            frame = np.clip(frame, a_min=15, a_max=255)  # prevents values in frame below 15
            if self.show_mask.get() == 1:  # showing mask
                frame = cv2.bitwise_and(frame, np.dstack((self.mask, self.mask, self.mask)))
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            cv2.imwrite('Masks/' + self.vid_name + '.png', rgb_frame)
        self.Notifications.insert((1.0), 'Making mask file\n')

    def read_in_mask_files(self):
        try:
            mask = cv2.imread('Masks/' + self.vid_name + '.png')
            #self.mask = mask[:, :, 0]
            ret, self.mask = cv2.threshold(mask[:, :, 0], 10, 255, cv2.THRESH_BINARY)
            self.Notifications.insert((1.0), 'Found mask file\n')
            self.show_mask.set(1)
        except (FileExistsError, TypeError):
            pass

    def size_filter_func(self, contours):
        min_size = int(self.size_filter_min.get())
        max_size = int(self.size_filter_max.get())
        bigconts = []
        templist = []  # holds mover points for this iteration
        for cont in contours:
            area = cv2.contourArea(cont)
            if max_size >= area >= min_size:
                bigconts.append(cont)
                M = cv2.moments(cont)
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])
                (x, y), radius = cv2.minEnclosingCircle(cont)
                templist.append([cX, cY, area, x, y, radius])
        return bigconts, templist

    def movers_background_subtract(self):
        self.stopped_button.deselect()  # un-stop us when you click this
        # grab weights
        hist = int(self.histEntry.get())
        var = int(self.varEntry.get())
        frames = int(self.framesEntry.get())
        detect_shadows = False
        if self.shadows.get() == 1:
            detect_shadows = True
        if self.background_subtractor.get() == 'MOG':
            fgbg = cv2.createBackgroundSubtractorMOG2(history=hist, varThreshold=var, detectShadows=detect_shadows)
        else:
            fgbg = cv2.createBackgroundSubtractorKNN(history=hist, dist2Threshold=var, detectShadows=detect_shadows)
        # set up list to take positions of movers
        moverslist = []
        # iterate through frames
        start = 0
        if self.start_after_hist.get() == 1:
            start = hist
        end = int(self.vid.count_frames())
        if frames > 0:
            end = start + frames
        can_run = True
        if start > end:
            self.Notifications.insert((1.0), 'Starting frame is past end of clip\n')
            can_run = False
        for x in range(start, end):
            ret, f = self.vid.get_frame(x)
            masked_f = cv2.bitwise_and(f, f, mask=self.mask)
            fgmask = fgbg.apply(masked_f)
            # analyze
            min_size = int(self.size_filter_min.get())
            kernel = np.ones((min_size, min_size), np.uint8)
            fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel)
            fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_CLOSE, kernel)
            contours, hierarchy = cv2.findContours(fgmask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            bigconts, templist = self.size_filter_func(contours)
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
                    cv2.circle(f, (int(m[3]), int(m[4])), int(m[5]), (255, 0, 0))
                self.photo = PIL.ImageTk.PhotoImage(image=PIL.Image.fromarray(f))
            self.canvas.create_image(0, 0, image=self.photo, anchor=NW)
            self.canvas.update_idletasks()
            self.Notifications.insert((1.0), 'Frame ' + str(x) + '\n')
            self.AppParent.update()
            if self.stopped.get() == 1:
                break

        tracks, track_summaries = self.makeTracks(moverslist)
        if tracks_enabled and can_run:
            for t in tracks:
                for x in range(1, len(t)-1):
                    (xt, yt, area) = t[x-1]
                    (xt1, yt1, area2) = t[x]
                    cv2.line(tempmask, (xt, yt), (xt1, yt1), (0, 0, 255), 1)
            # write out
            try:
                cv2.imwrite('Output/' + self.vid_name + ' tracks.png', tempmask)
            except UnboundLocalError:
                print('Could not save tracks')
            try:
                cv2.imwrite('Output/' + self.vid_name + ' screenshot.png', f)
            except cv2.error:
                print('Could not save screenshot')
        self.writeout(track_summaries)

    def optical_flow(self):
        self.stopped_button.deselect()
        intensity_thresh = 50
        winsize = 5
        frames = 50
        hsv = np.zeros_like(self.vid.get_frame(0)[1])
        hsv[..., 1] = 255
        # set up list to take positions of movers
        moverslist = []
        start = 1
        end = int(self.vid.count_frames())
        if frames > 0:
            end = start + frames
        for x in range(start, end):
            ret_old, f_old = self.vid.get_frame(x - 1)
            ret_new, f_new = self.vid.get_frame(x)
            f_old_gr = cv2.cvtColor(f_old, cv2.COLOR_BGR2GRAY)
            f_new_gr = cv2.cvtColor(f_new, cv2.COLOR_BGR2GRAY)
            flow = cv2.calcOpticalFlowFarneback(f_old_gr, f_new_gr, None, 0.5, 3, winsize, 3, 5, 1.2, 0)
            mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
            #hsv[..., 2] = ang * 180 / np.pi / 2
            hsv[..., 2] = 100
            hsv[..., 0] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
            hsv[..., 0][hsv[..., 0] < intensity_thresh] = 0
            bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
            t, thresholded = cv2.threshold(hsv[..., 0], intensity_thresh, 255, cv2.THRESH_BINARY)

            # analyze
            contours, hierarchy = cv2.findContours(thresholded, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            bigconts, templist = self.size_filter_func(contours)
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
                self.photo = PIL.ImageTk.PhotoImage(image=PIL.Image.fromarray(f_new))
            elif self.show_mode.get() == 'M':
                self.photo = PIL.ImageTk.PhotoImage(image=PIL.Image.fromarray(bgr))
            elif self.show_mode.get() == 'F':
                self.photo = PIL.ImageTk.PhotoImage(image=PIL.Image.fromarray(tempmask.astype(np.uint8)))
            else:  # must be normal frame with size filtered motion
                for m in moverslist[-1]:
                    cv2.circle(f_new, (int(m[3]), int(m[4])), int(m[5]), (255, 0, 0))
                self.photo = PIL.ImageTk.PhotoImage(image=PIL.Image.fromarray(f_new))
            self.canvas.create_image(0, 0, image=self.photo, anchor=NW)
            self.canvas.update_idletasks()
            self.Notifications.insert((1.0), 'Frame ' + str(x) + '\n')
            self.AppParent.update()
            if self.stopped.get() == 1:
                break

        tracks, track_summaries = self.makeTracks(moverslist)
        if tracks_enabled:
            for t in tracks:
                for x in range(1, len(t) - 1):
                    (xt, yt, area) = t[x - 1]
                    (xt1, yt1, area2) = t[x]
                    cv2.line(tempmask, (xt, yt), (xt1, yt1), (0, 0, 255), 1)
            # write out
            cv2.imwrite('Output/' + self.vid_name + ' tracks.png', tempmask)
            try:
                cv2.imwrite('Output/' + self.vid_name + ' screenshot.png', f_new)
            except cv2.error:
                print('Could not save screenshot')
        self.writeout(track_summaries)

    def test(self):
        pass

    def write_vars_to_file(self):
        hist = int(self.histEntry.get())
        var = int(self.varEntry.get())
        frames = int(self.framesEntry.get())
        min_size = int(self.size_filter_min.get())
        max_size = int(self.size_filter_max.get())
        vars = [
            ['hist', hist],
            ['var', var],
            ['frames', frames],
            ['filter_1', min_size],
            ['filter_2', max_size]
        ]
        with open('saved_program_state.csv', 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            for x in vars:
                writer.writerow(x)

    def read_in_vars(self):
        try:
            with open('saved_program_state.csv', newline='') as csvfile:
                reader = csv.reader(csvfile)
                vars = [row for row in reader]
                widgets_list = [self.histEntry, self.varEntry, self.framesEntry, self.size_filter_min,
                                self.size_filter_max]
                step = 0
                for w in widgets_list:
                    w.delete(0, END)
                    w.insert(0, vars[step][1])
                    step += 1
        except FileNotFoundError:
            pass

    def writeout(self, track_data):
        df = pd.DataFrame(track_data)
        if self.write_to_one_file.get() == 0:
            self.make_or_reset_persistent_data()  # clear persistent data since we're not using it
            out_file_name = os.path.join('Output', self.vid_name + ' Tracks.csv')
            df.to_csv(out_file_name, index=False)
        else:  # write to one file
            out_file_name = os.path.join('Output', self.folder_name + ' Tracks.csv')
            self.persistent_data = pd.merge(self.persistent_data, df)
            self.persistent_data.to_csv(out_file_name, index=False)

    def makeTracks(self, moverslist):
        # takes a list of movers positions and calculates all track data
        # tracks is opencv data that is then used to draw lines
        # track_summaries is the dataframe that gets written out
        tracks = []
        for m in moverslist: # go through frames
            for mover in m:
                [cX, cY, area, x1, y1, radius] = mover
                d = 200
                track = None
                if len(tracks) <= 0:
                    tracks.append([[cX, cY, area]])
                else:
                    for x in range(0, len(tracks)-1):
                        t = tracks[x][-1]
                        dist = ((cX - t[0]) ** 2) + ((cY - t[1]) ** 2)
                        if dist < d:
                            track = x
                            d = dist
                    if track != None:
                        tracks[track].append([cX, cY, area])
                    else:
                        tracks.append([[cX, cY, area]])
        track_summaries = {'Time_of_track': [], 'Avg_area_of_mover': [], 'Distance_of_track': []}
        for track in tracks:
            sum_area = 0
            sum_distance = 0
            for x in range(0, len(track) - 1):
                moment = track[x]
                sum_area += moment[2]
                if x > 0:  # wait until second item
                    old_moment = track[x - 1]
                    d = (((moment[0] - old_moment[0]) ** 2) + ((moment[1] - old_moment[1]) ** 2)) ** 0.5
                    sum_distance += d
            track_summaries['Time_of_track'].append(len(track))
            track_summaries['Avg_area_of_mover'].append(sum_area / len(track))
            track_summaries['Distance_of_track'].append(sum_distance)

        return tracks, track_summaries



root = Tk()
displayapp = DisplayApp(root)
root.mainloop()

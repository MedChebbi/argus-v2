from enum import IntEnum

import cv2
import numpy as np


def nothing(x):
    pass

class Mode(IntEnum):
    """
    Type of detection to build
    """
    LINE = 0,
    BLOB = 1

class ColorSpace(IntEnum):
    """
    """
    BGR = 0,
    HSV = 1,
    LAB = 2

class ColorDetector:

    def __init__(self, logger, params):
        self._logger = logger
        self.color_space = params['color_space']
        self.min_values = np.array(params['color_min_range'])
        self.max_values = np.array(params['color_max_range'])
        self.max_area_thre = params["MAX_AREA"]
        self.min_area_thre = params["MIN_AREA"]
        self.detection_mode = params['detection_mode']

        self.blob_detected = False
        self.error = 0
        self.ang = 0


    def __initialize_trackbars(self):
        #initialize trackbar gui
        cv2.namedWindow("Trackbars")
        cv2.resizeWindow("Trackbars", 400, 400)

        maxValue = 255
        maxValueHue = 180
        #Black approx range in hsv: (0, 0, 0) ~ (180, 255, 30)
        #White approx range in hsv: (0, 0, 180) ~ (180, 0, 255)
        #Green approx range in hsv: (40, 40,40) ~ (70, 255,255)
        #Red approx range in hsv: (0, 120, 30) ~ (25, 255, 255)
        #Blue approx range in hsv: (110,150,50) ~ (120,255,255)
        cv2.createTrackbar("min_Hue", "Trackbars", self.min_values[0],maxValueHue , nothing)
        cv2.createTrackbar("min_Saturation", "Trackbars", self.min_values[1] ,maxValue, nothing)
        cv2.createTrackbar("min_Value", "Trackbars", self.min_values[2] ,maxValue, nothing)
        cv2.createTrackbar("max_Hue", "Trackbars", self.max_values[0],maxValueHue , nothing)
        cv2.createTrackbar("max_Saturation", "Trackbars",self.max_values[1] , maxValue, nothing)
        cv2.createTrackbar("max_Value", "Trackbars", self.max_values[2], maxValue, nothing)


    def __extract_values(self):
        min_H = cv2.getTrackbarPos("min_Hue", "Trackbars")
        max_H = cv2.getTrackbarPos("max_Hue", "Trackbars")
        min_S = cv2.getTrackbarPos("min_Saturation", "Trackbars")
        max_S = cv2.getTrackbarPos("max_Saturation", "Trackbars")
        min_V = cv2.getTrackbarPos("min_Value", "Trackbars")
        max_V = cv2.getTrackbarPos("max_Value", "Trackbars")
        return np.array([min_H,min_S,min_V]), np.array([max_H,max_S,max_V])
    

    def fine_tune(self):
        self.min_values, self.max_values = self.__extract_values()


    def detect(self, frame, debug_frame, params=None, draw = False):
        
        needed_info = []
        self.WIDTH = frame.shape[1]
        self.HEIGHT = frame.shape[0]
        setpoint = self.WIDTH//2

        self.x_last = self.WIDTH//2
        self.y_last = self.HEIGHT//2

        roi = frame.copy()
        if self.detection_mode == Mode.LINE:
            self.max_area_thre = params["MAX_AREA"]
            self.min_area_thre = params["MIN_AREA"]
            thres = params["THRES"]
            self.max_values = np.array([thres,thres,thres])
            self.roi_mask = int(self.HEIGHT/2)
            roi[0:self.roi_mask,:] = (255,255,255)      #(B, G, R)

        blurred_frame = cv2.GaussianBlur(roi,(5,5),1)
        if self.color_space == ColorSpace.HSV:
            processed_frame = cv2.cvtColor(blurred_frame, cv2.COLOR_BGR2HSV)
        elif self.color_space == ColorSpace.BGR:
            processed_frame = blurred_frame
        elif self.color_space == ColorSpace.LAB:
            processed_frame = cv2.cvtColor(blurred_frame, cv2.COLOR_BGR2LAB)

        mask = cv2.inRange(processed_frame, self.min_values, self.max_values)
        kernel = np.ones((3,3), np.uint8)
        mask = cv2.erode(mask, kernel, iterations=2)
        mask = cv2.dilate(mask, kernel, iterations=2) #we used those filters to smooth the mask for a better detection

        contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        contours_len = len(contours)
        areas = [0]
        for i in contours:
            area = cv2.contourArea(i)
            areas.append(area)
        max_area = max(areas)
        
        if contours_len > 0 and (self.min_area_thre < max_area < self.max_area_thre):
            self.blob_detected = True
            

            if self.detection_mode == Mode.LINE:
                box = self.__detect_black(contours, setpoint)
                centertext = "Offset: " + str(self.error)
                if draw:
                    cv2.drawContours(debug_frame,[box],0,(0,0,255),3)
                    cv2.putText(debug_frame,"Angle: "+str(self.ang),(10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, self.RED, 2)
                    cv2.drawContours(debug_frame, contours, -1, self.GREEN, 3)
                    cv2.putText(debug_frame, centertext, (200,340), cv2.FONT_HERSHEY_SIMPLEX, 1, self.RED,2)
                    cv2.circle(debug_frame, (self.WIDTH//2, self.HEIGHT//2),5, self.BLUE,cv2.FILLED)
                    cv2.line(debug_frame, (int(self.x_last), 200), (int(self.x_last), 250),self.BLUE,3)
            
            else:
                new_areas = []
                #print("[INFO] Color mode")
                
                if contours_len == 1:
                    x,y,w,h = cv2.boundingRect(contours[0])
                else:
                    y_max = 0
                    for j, cont in enumerate(contours):
                        area = cv2.contourArea(cont)
                        new_areas.append(area)
                        new_max_area = max(new_areas)
                        x,y,w,h = cv2.boundingRect(cont)
                        if y > y_max:
                            y_max = y
                            j_max = j
                        
                    # print("y_max: ",y_max)
                    # print(areas.index(j_max))
                    lowest_contour = contours[j_max]
                    biggest_contour = contours[new_areas.index(new_max_area)]
                    x,y,w,h = cv2.boundingRect(lowest_contour)
                
                self.x_last = x + (w//2)
                self.y_last = y + (h//2)
                self.error = int(self.x_last - setpoint)
                centertext = "Offset: " + str(self.error)
                if draw:
                    #cv2.drawContours(debug_frame,lowest_contour,-1, self.GREEN,3)
                    cv2.rectangle(debug_frame, (x,y), (x+w,y+h), self.GREEN,2)
                    cv2.putText(debug_frame, centertext, (200,340), cv2.FONT_HERSHEY_SIMPLEX, 1, self.RED,2)
                    cv2.circle(debug_frame, (self.WIDTH//2, self.HEIGHT//2),5, self.BLUE,cv2.FILLED)
                    cv2.line(debug_frame, (int(self.x_last), 200), (int(self.x_last), 250),self.BLUE,3)
                
        else :
            self.blob_detected = False
            
        needed_info.append(self.blob_detected)
        needed_info.append(self.error)
        needed_info.append(self.ang)

        return debug_frame, mask, needed_info


    def __detect_black(self, contours, setpoint):
        coff = 10
        contours_len = len(contours)
        if contours_len == 1:
            x,y,w,h = cv2.boundingRect(contours[0])
            blackbox = cv2.minAreaRect(contours[0])

        else:
            dets = []
            off_bottom = 0
            for cont_num in range (contours_len):
                blackbox = cv2.minAreaRect(contours[cont_num])
                (x_min, y_min), (w_min, h_min), ang = blackbox
                box = cv2.boxPoints(blackbox)
                (x_box, y_box) = box[0]
                if y_box > self.HEIGHT-self.roi_mask-1 :
                    off_bottom += 1
                dets.append((y_box,cont_num,x_min,y_min))
            dets = sorted(dets)
            if off_bottom > 1:
                canditates_off_bottom=[]
                for con_num in range ((contours_len - off_bottom), contours_len):
                    (y_highest,con_highest,x_min, y_min) = dets[con_num]
                    total_distance = (abs(x_min - self.x_last)**2 + abs(y_min - self.y_last)**2)**0.5
                    canditates_off_bottom.append((total_distance,con_highest))
                canditates_off_bottom = sorted(canditates_off_bottom)
                (total_distance,con_highest) = canditates_off_bottom[0]
                blackbox = cv2.minAreaRect(contours[con_highest])
            else:
                (y_highest,con_highest,x_min, y_min) = dets[contours_len-1]
                blackbox = cv2.minAreaRect(contours[con_highest])

        (x_min, y_min), (w_min, h_min), ang = blackbox
        self.x_last = x_min
        self.y_last = y_min

        if (ang < -45) or (w_min > h_min and ang < 0): ang += 90
        if (w_min < h_min and ang > 0): ang = (90-ang)*-1

        ang = int(ang)
        self.ang = ang

        box = cv2.boxPoints(blackbox)
        box = np.int0(box)
        error = int(self.x_last - setpoint)
            
        self.error = error//coff
        return box

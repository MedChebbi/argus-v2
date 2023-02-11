from enum import IntEnum

import cv2
import numpy as np

from argus_computer_vision.utils import Color


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
        """
        """
        self._logger = logger
        self._color_space = params['color_space']
        self._detection_mode = Mode(params['detection_mode'])

        self._min_values = np.array(params['color_min_range'])
        self._max_values = np.array(params['color_max_range'])
        self._max_area_thre = params["max_area"]
        self._min_area_thre = params["min_area"]

        self._x_last = None
        self._y_last = None

        self.SETPOINT_RATIO = 0.5 # Between ]0, 1[ ratio of frame width
        self.ROI_MASK_RATIO = 0.5 # Between ]0, 1[ ratio of frame height
        self._box = []

        self._blob_detected = False
        self._error = 0 # Should be between [-0.5 ; 0.5]
        self._ang = 0 # In angles

    @property
    def bbox(self):
        return self._box

    @property
    def x_last(self):
        return self._x_last

    @property
    def contours(self):
        return self._contours

    def __preprocess_mask(self, roi):
        """
        It gives a mask using color segmentation given a 3 channel frame
        """
        roi = cv2.GaussianBlur(roi, (5, 5), 1)
        if self._color_space == ColorSpace.HSV:
            t_frame = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        elif self._color_space == ColorSpace.LAB:
            t_frame = cv2.cvtColor(roi, cv2.COLOR_BGR2LAB)
        else:
            t_frame = roi

        mask = cv2.inRange(t_frame, self._min_values, self._max_values)
        kernel = np.ones((3,3), np.uint8)
        mask = cv2.erode(mask, kernel, iterations=2)
        mask = cv2.dilate(mask, kernel, iterations=2) # we used those filters to smooth the mask for a better detection
        return mask


    def detect(self, frame, params=None, debug=False):
        """
        """
        height, width = frame.shape[:2]
        frame_area = height * width
        print(f"frame area: {frame_area}")
        setpoint = width * self.SETPOINT_RATIO
        roi = frame.copy()
        if self._x_last is None or self._y_last is None:
            self._x_last = width // 2
            self._y_last = height // 2

        if params is not None:
            self._max_area_thre = params["max_area"]
            self._min_area_thre = params["min_area"]
            self._min_values = np.array(params['color_min_range'])
            self._max_values = np.array(params['color_max_range'])
            
        if self._detection_mode == Mode.LINE:
            self._roi_mask = int(height * self.ROI_MASK_RATIO)
            roi[:self._roi_mask, :] = Color.WHITE # (B, G, R)
        
        mask = self.__preprocess_mask(roi)
        # cv2.imshow("mask", mask)
        # cv2.waitKey(0)

        contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        self._contours = contours
        contours_len = len(contours)
        areas = [0]
        for i in contours:
            area = cv2.contourArea(i)
            areas.append(area)
        max_area = max(areas) / frame_area
        print(f"Max area: {max_area}")

        if contours_len > 0 and (self._min_area_thre < max_area < self._max_area_thre):
            self._logger.info(f"[{self._detection_mode.name}]: Detected")
            if self._detection_mode == Mode.LINE:
                self._box, error, ang = self.__detect_line(contours, setpoint, width, height)
            else:
                self._box, error, ang = self.__detect_blob(contours, setpoint, width)

            self._blob_detected = True
            self._error = error
            self._ang = ang

        else :
            self._blob_detected = False
            self._error = 0
            self._ang = 0
            self._box = []
        
        if debug:
            self._logger.info(f"[{self._detection_mode.name}]: {self._blob_detected}, error: {self._error}, angle: {self._ang}")
        
        return self._blob_detected, self._error, self._ang

    def __detect_line(self, contours, setpoint, width, height):
        """
        """
        contours_len = len(contours)
        if contours_len == 1:
            # x, y, w, h = cv2.boundingRect(contours[0])
            blackbox = cv2.minAreaRect(contours[0])

        else:
            dets = []
            off_bottom = 0
            for cont_num in range (contours_len):
                blackbox = cv2.minAreaRect(contours[cont_num])
                (x_min, y_min), (w_min, h_min), ang = blackbox
                box = cv2.boxPoints(blackbox)
                (x_box, y_box) = box[0]
                if y_box > height - self._roi_mask - 1:
                    off_bottom += 1
                dets.append((y_box,cont_num,x_min,y_min))
            dets = sorted(dets)
            if off_bottom > 1:
                canditates_off_bottom=[]
                for con_num in range ((contours_len - off_bottom), contours_len):
                    (y_highest,con_highest,x_min, y_min) = dets[con_num]
                    total_distance = (abs(x_min - self._x_last)**2 + abs(y_min - self._y_last)**2)**0.5
                    canditates_off_bottom.append((total_distance, con_highest))
                canditates_off_bottom = sorted(canditates_off_bottom)
                (total_distance, con_highest) = canditates_off_bottom[0]
                blackbox = cv2.minAreaRect(contours[con_highest])
            else:
                (y_highest, con_highest, x_min, y_min) = dets[contours_len-1]
                blackbox = cv2.minAreaRect(contours[con_highest])

        (x_min, y_min), (w_min, h_min), ang = blackbox
        self._x_last = x_min
        self._y_last = y_min

        if (ang < -45) or (w_min > h_min and ang < 0): ang += 90
        if (w_min < h_min and ang > 0): ang = (90-ang)*-1

        ang = int(ang)
        box = cv2.boxPoints(blackbox)
        box = np.int0(box)
        error = (self._x_last - setpoint) / width
        return box, error, ang

    def __detect_blob(self, contours, setpoint, width):
        """
        """
        new_areas = []
        contours_len = len(contours)
        # print("[INFO] Color mode")
        if contours_len == 1:
            x, y, w, h = cv2.boundingRect(contours[0])
        else:
            y_max = 0
            for j, cont in enumerate(contours):
                area = cv2.contourArea(cont)
                new_areas.append(area)
                new_max_area = max(new_areas)
                x, y, w, h = cv2.boundingRect(cont)
                if y > y_max:
                    y_max = y
                    j_max = j
                
            # print("y_max: ",y_max)
            # print(areas.index(j_max))
            lowest_contour = contours[j_max]
            biggest_contour = contours[new_areas.index(new_max_area)]
            x, y, w, h = cv2.boundingRect(lowest_contour)
        
        self._x_last = x + (w//2)
        self._y_last = y + (h//2)
        ang = 0
        error = (self._x_last - setpoint) / width
        
        return [x, y, w, h], error, ang





    # def detect(self, frame, debug_frame, params=None, draw = False):
        
    #     needed_info = []
    #     self.WIDTH = frame.shape[1]
    #     self.HEIGHT = frame.shape[0]
    #     setpoint = self.WIDTH//2

    #     self._x_last = self.WIDTH//2
    #     self._y_last = self.HEIGHT//2

    #     roi = frame.copy()
    #     if self._detection_mode == Mode.LINE:
    #         self._max_area_thre = params["MAX_AREA"]
    #         self._min_area_thre = params["MIN_AREA"]
    #         thres = params["THRES"]
    #         self._max_values = np.array([thres,thres,thres])
    #         self._roi_mask = int(self.HEIGHT/2)
    #         roi[:self._roi_mask,:] = Color.WHITE      # (B, G, R)

    #     blurred_frame = cv2.GaussianBlur(roi,(5,5),1)
    #     if self._color_space == ColorSpace.HSV:
    #         processed_frame = cv2.cvtColor(blurred_frame, cv2.COLOR_BGR2HSV)
    #     elif self._color_space == ColorSpace.BGR:
    #         processed_frame = blurred_frame
    #     elif self._color_space == ColorSpace.LAB:
    #         processed_frame = cv2.cvtColor(blurred_frame, cv2.COLOR_BGR2LAB)

    #     mask = cv2.inRange(processed_frame, self._min_values, self._max_values)
    #     kernel = np.ones((3,3), np.uint8)
    #     mask = cv2.erode(mask, kernel, iterations=2)
    #     mask = cv2.dilate(mask, kernel, iterations=2) #we used those filters to smooth the mask for a better detection

    #     contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    #     contours_len = len(contours)
    #     areas = [0]
    #     for i in contours:
    #         area = cv2.contourArea(i)
    #         areas.append(area)
    #     max_area = max(areas)
        
    #     if contours_len > 0 and (self._min_area_thre < max_area < self._max_area_thre):
    #         self._blob_detected = True
            

    #         if self._detection_mode == Mode.LINE:
    #             box = self.__detect_black(contours, setpoint)
    #             centertext = "Offset: " + str(self._error)
    #             if draw:
    #                 cv2.drawContours(debug_frame,[box],0,(0,0,255),3)
    #                 cv2.putText(debug_frame,"Angle: "+str(self._ang),(10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, self.RED, 2)
    #                 cv2.drawContours(debug_frame, contours, -1, self.GREEN, 3)
    #                 cv2.putText(debug_frame, centertext, (200,340), cv2.FONT_HERSHEY_SIMPLEX, 1, self.RED,2)
    #                 cv2.circle(debug_frame, (self.WIDTH//2, self.HEIGHT//2),5, self.BLUE,cv2.FILLED)
    #                 cv2.line(debug_frame, (int(self._x_last), 200), (int(self._x_last), 250),self.BLUE,3)
            
    #         else:
    #             new_areas = []
    #             #print("[INFO] Color mode")
                
    #             if contours_len == 1:
    #                 x,y,w,h = cv2.boundingRect(contours[0])
    #             else:
    #                 y_max = 0
    #                 for j, cont in enumerate(contours):
    #                     area = cv2.contourArea(cont)
    #                     new_areas.append(area)
    #                     new_max_area = max(new_areas)
    #                     x,y,w,h = cv2.boundingRect(cont)
    #                     if y > y_max:
    #                         y_max = y
    #                         j_max = j
                        
    #                 # print("y_max: ",y_max)
    #                 # print(areas.index(j_max))
    #                 lowest_contour = contours[j_max]
    #                 biggest_contour = contours[new_areas.index(new_max_area)]
    #                 x,y,w,h = cv2.boundingRect(lowest_contour)
                
    #             self._x_last = x + (w//2)
    #             self._y_last = y + (h//2)
    #             self._error = int(self._x_last - setpoint)
    #             centertext = "Offset: " + str(self._error)
    #             if draw:
    #                 #cv2.drawContours(debug_frame,lowest_contour,-1, self.GREEN,3)
    #                 cv2.rectangle(debug_frame, (x,y), (x+w,y+h), self.GREEN,2)
    #                 cv2.putText(debug_frame, centertext, (200,340), cv2.FONT_HERSHEY_SIMPLEX, 1, self.RED,2)
    #                 cv2.circle(debug_frame, (self.WIDTH//2, self.HEIGHT//2),5, self.BLUE,cv2.FILLED)
    #                 cv2.line(debug_frame, (int(self._x_last), 200), (int(self._x_last), 250),self.BLUE,3)
                
    #     else :
    #         self._blob_detected = False
            
    #     needed_info.append(self._blob_detected)
    #     needed_info.append(self._error)
    #     needed_info.append(self._ang)

    #     return debug_frame, mask, needed_info

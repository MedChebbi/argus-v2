from enum import IntEnum

import cv2
import numpy as np
import math


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
        self.ROI_MASK_RATIO = 0.2 # Between ]0, 1[ ratio of frame height
        self._box = []
        self._mask = None

        self._blob_detected = False
        self._error = 0 # Should be between [-0.5 ; 0.5]
        self._ang = 0 # In angles

    @property
    def mask(self):
        return self._mask

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
        
        # if self._detection_mode == Mode.LINE:
        #     roi = cv2.cvtColor(roi, cv2.COLOR_BGR2LAB)
        #     mask = cv2.inRange(t_frame, self._min_values, np.array([self._thr, self._thr, self._thr]))

        #     # roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        #     # # _, mask = cv2.threshold(roi, self._thr, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        #     # _, mask = cv2.threshold(roi, self._thr, 255, cv2.THRESH_BINARY_INV)
        mask = cv2.inRange(t_frame, self._min_values, self._max_values)
        kernel = np.ones((3,3), np.uint8)
        mask = cv2.erode(mask, kernel, iterations=2)
        mask = cv2.dilate(mask, kernel, iterations=2) # we used those filters to smooth the mask for a better detection
        # cv2.imshow("mask", mask)
        # cv2.waitKey(10)
        return mask


    def detect(self, frame, params=None, debug=False):
        """
        """
        height, width = frame.shape[:2]
        frame_area = height * width
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
                self._max_values = np.array([params['thr'] * 3])
        
        self._mask = self.__preprocess_mask(roi)
        mask = self._mask.copy()

        if self._detection_mode == Mode.LINE:
            self._roi_mask = int(height * self.ROI_MASK_RATIO)
            mask[:self._roi_mask, :] = 0

        contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        self._contours = contours
        contours_len = len(contours)
        areas = [0]
        for i in contours:
            area = cv2.contourArea(i)
            areas.append(area)
        max_area = max(areas) / frame_area

        if contours_len > 0 and (self._min_area_thre < max_area < self._max_area_thre):
            if self._detection_mode == Mode.LINE:
                self._box, error, ang = self.__detect_line(contours, setpoint, width, height)
            else:
                self._box, error, ang = self.__detect_blob(contours, setpoint, width)

            self._blob_detected = True
            self._error = error
            self._ang = ang

        else :
            self._blob_detected = False
            self._error = 0.0
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
                (x_min, y_min), _, _ = blackbox
                box = cv2.boxPoints(blackbox)
                (x_box, y_box) = box[0]
                if y_box > height - self._roi_mask - 1:
                    off_bottom += 1
                dets.append((y_box, cont_num, x_min, y_min))
            dets = sorted(dets)
            if off_bottom > 1:
                canditates_off_bottom=[]
                for con_num in range ((contours_len - off_bottom), contours_len):
                    (_ ,con_highest,x_min, y_min) = dets[con_num]
                    total_distance = (abs(x_min - self._x_last)**2 + abs(y_min - self._y_last)**2)**0.5
                    canditates_off_bottom.append((total_distance, con_highest))
                canditates_off_bottom = sorted(canditates_off_bottom)
                (total_distance, con_highest) = canditates_off_bottom[0]
                blackbox = cv2.minAreaRect(contours[con_highest])
            else:
                (_, con_highest, x_min, y_min) = dets[contours_len-1]
                blackbox = cv2.minAreaRect(contours[con_highest])

        (x_min, y_min), _ , _ = blackbox

        self._x_last = x_min
        self._y_last = y_min

        box = cv2.boxPoints(blackbox)
        box = np.int32(box)

        # Sort bbox by desending y
        ind = np.argsort(box[:, 1])
        b = box[ind]

        # Width is the x difference of the top 2 points
        w = abs(b[0][0] - b[1][0])

        # P1 is the highest center point of the rotated rect
        x1, y1 = int(np.min(b[:2, 0]) + w / 2), b[0][1]

        # P2 is the lowest center point of the rotated rect
        x2, y2 = int(np.min(b[2:, 0]) + w / 2), b[-1][1]

        # Calculating the angle between two vectors
        ang = math.atan2(y2 - y1, x2 - x1)
        ang = int(math.degrees(ang) - 90)
        
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

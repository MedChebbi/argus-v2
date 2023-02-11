import cv2

from argus_computer_vision.utils import Color


class Display:

    @staticmethod
    def display_line(debug_frame, line_detector, error, ang):
        height, width = debug_frame.shape[:2]
        x_last = line_detector.x_last
        box = line_detector.bbox
        contours = line_detector.contours
        centertext = f"Offset: {error:.3f}"
        if len(box):
            cv2.drawContours(debug_frame, [box],0, Color.RED,3)
            cv2.putText(debug_frame,"Angle: "+str(ang),(10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, Color.RED, 2)
            cv2.drawContours(debug_frame, contours, -1, Color.GREEN, 3)
            cv2.putText(debug_frame, centertext, (200,340), cv2.FONT_HERSHEY_SIMPLEX, 1, Color.RED,2)
            cv2.circle(debug_frame, (width // 2, height // 2), 5, Color.BLUE, cv2.FILLED)
            cv2.line(debug_frame, (int(x_last), 200), (int(x_last), 250),Color.BLUE, 3)
        return debug_frame

    @staticmethod
    def display_blob(image, bolb_detector):
        return image

    @staticmethod
    def display_line_state(image, state):
        return image

    @staticmethod
    def display_aruco_markers(image, corners, ids):
        if len(corners) > 0:
            ids = ids.flatten()
            for (marker_corner, marker_id) in zip(corners, ids):
        
                corners = marker_corner.reshape((4, 2))
                (top_left, top_right, bottom_right, bottom_left) = corners
        
                top_right = (int(top_right[0]), int(top_right[1]))
                bottom_right = (int(bottom_right[0]), int(bottom_right[1]))
                bottom_left = (int(bottom_left[0]), int(bottom_left[1]))
                top_left = (int(top_left[0]), int(top_left[1]))

                cv2.line(image, top_left, top_right, (0, 255, 0), 2)
                cv2.line(image, top_right, bottom_right, (0, 255, 0), 2)
                cv2.line(image, bottom_right, bottom_left, (0, 255, 0), 2)
                cv2.line(image, bottom_left, top_left, (0, 255, 0), 2)
        
                cX = int((top_left[0] + bottom_right[0]) / 2.0)
                cY = int((top_left[1] + bottom_right[1]) / 2.0)
                cv2.circle(image, (cX, cY), 4, (0, 0, 255), -1)
            
                cv2.putText(image, str(marker_id),(top_left[0], top_left[1] - 10), cv2.FONT_HERSHEY_SIMPLEX,
                            0.5, (0, 255, 0), 2)
        
        return image

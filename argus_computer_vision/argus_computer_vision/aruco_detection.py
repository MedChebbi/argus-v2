import cv2
import numpy as np


class ArUcoDetection:
    ARUCO_DICT = {
                "DICT_4X4_50": cv2.aruco.DICT_4X4_50,
                "DICT_4X4_100": cv2.aruco.DICT_4X4_100,
                "DICT_4X4_250": cv2.aruco.DICT_4X4_250,
                "DICT_4X4_1000": cv2.aruco.DICT_4X4_1000,
                "DICT_5X5_50": cv2.aruco.DICT_5X5_50,
                "DICT_5X5_100": cv2.aruco.DICT_5X5_100,
                "DICT_5X5_250": cv2.aruco.DICT_5X5_250,
                "DICT_5X5_1000": cv2.aruco.DICT_5X5_1000,
                "DICT_6X6_50": cv2.aruco.DICT_6X6_50,
                "DICT_6X6_100": cv2.aruco.DICT_6X6_100,
                "DICT_6X6_250": cv2.aruco.DICT_6X6_250,
                "DICT_6X6_1000": cv2.aruco.DICT_6X6_1000,
                "DICT_7X7_50": cv2.aruco.DICT_7X7_50,
                "DICT_7X7_100": cv2.aruco.DICT_7X7_100,
                "DICT_7X7_250": cv2.aruco.DICT_7X7_250,
                "DICT_7X7_1000": cv2.aruco.DICT_7X7_1000,
                "DICT_ARUCO_ORIGINAL": cv2.aruco.DICT_ARUCO_ORIGINAL,
                "DICT_APRILTAG_16h5": cv2.aruco.DICT_APRILTAG_16h5,
                "DICT_APRILTAG_25h9": cv2.aruco.DICT_APRILTAG_25h9,
                "DICT_APRILTAG_36h10": cv2.aruco.DICT_APRILTAG_36h10,
                "DICT_APRILTAG_36h11": cv2.aruco.DICT_APRILTAG_36h11
            }

    def __init__(self, logger=None, aruco_type='DICT_4X4_1000') -> None:
        """
        """
        self._logger = logger
        self._aruco_dict = cv2.aruco.Dictionary_get(self.ARUCO_DICT[aruco_type])
        self._aruco_params = cv2.aruco.DetectorParameters_create()


    @staticmethod
    def display(image, corners, ids):
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
                print("[Inference] ArUco marker ID: {}".format(marker_id))
        
        return image

    def detect(self, frame):
        corners, ids, _ = cv2.aruco.detectMarkers(frame, self._aruco_dict, parameters=self._aruco_params)
        return corners, ids

    def generate_aruco(self, id, aruco_type='DICT_4X4_1000'):
        tag_size = int(aruco_type.split('_')[-1])
        tag = np.zeros((tag_size, tag_size, 1), dtype="uint8")
        cv2.aruco.drawMarker(self._aruco_dict, id, tag_size, tag, 1)

        # Save the tag generated
        tag_name = aruco_type + "_" + str(id) + ".png"
        cv2.imwrite(tag_name, tag)


if __name__ == '__main__':
    detector = ArUcoDetection()
    img = cv2.imread('/home/mohamed/test.png')
    
    corners, ids = detector.detect(img)
    img = ArUcoDetection.display(img, corners, ids)
    cv2.imshow("frame", img)
    cv2.waitKey(0)
    # detector.generate_aruco(88)

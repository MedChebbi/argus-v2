import numpy as np
import cv2


class Color:
    RED = (0,0,255)
    BLUE = (255,0,0)
    GREEN = (0,255,0)
    BLACK = (0,0,0)
    WHITE = (255,255,255)


def drawPoints(img,points):
    for x in range(4):
        cv2.circle(img,(int(points[x][0]),int(points[x][1])),5,(0,0,255),cv2.FILLED)
    return img


def reorder(myPoints):
	myPoints = myPoints.reshape((4, 2))
	myPointsNew = np.zeros((4, 1, 2), dtype=np.int32)
	add = myPoints.sum(1)
	myPointsNew[0] = myPoints[np.argmin(add)]
	myPointsNew[3] =myPoints[np.argmax(add)]
	diff = np.diff(myPoints, axis=1)
	myPointsNew[1] =myPoints[np.argmin(diff)]
	myPointsNew[2] = myPoints[np.argmax(diff)]
	return myPointsNew
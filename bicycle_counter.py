from include.centroidtracker import CentroidTracker
from include.trackableobject import TrackableObject
import numpy as np
import imutils
import dlib
import cv2

INPUT = 'videos/video_1.mp4'
SKIP_FRAMES = 30

vs = cv2.VideoCapture(INPUT)

W = None
H = None

ct = CentroidTracker(maxDisappeared=40, maxDistance=50)
trackers = []
trackableObjects = {}

totalFrames = 0
counter = 0
firstFrame = None
kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
# przechodzimy po kazdej klatce filmu
while True:
    frame = vs.read()
    frame = frame[1]

    if frame is None:
        break

    frame = imutils.resize(frame, width=500)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    if W is None or H is None:
        (H, W) = frame.shape[:2]

    rects = []

    if totalFrames % SKIP_FRAMES == 0:
        trackers = []

        cv2.imwrite("output/frame/frame_" + str(totalFrames) + ".jpg", frame)
        cv2.imwrite("output/gray/gray_" + str(totalFrames) + ".jpg", gray)

        if firstFrame is None:
            firstFrame = gray
            continue

        frameDelta = cv2.absdiff(firstFrame, gray)
        thresh = cv2.threshold(frameDelta, 5, 255, cv2.THRESH_BINARY)[1]
        cnts = cv2.findContours(
            thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)

        cv2.imwrite("output/frameDelta/frameDelta_" + str(totalFrames) + ".jpg", frameDelta)
        cv2.imwrite("output/thresh/thresh_" + str(totalFrames) + ".jpg", thresh)

        for c in cnts:
            if cv2.contourArea(c) < 500:
                continue

            peri = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, 0.04 * peri, True)

            if len(approx) != 3:
                continue

            (x, y, w, h) = cv2.boundingRect(c)
            detected_circles = cv2.HoughCircles(
                thresh, cv2.HOUGH_GRADIENT, 1, 20, param1=50, param2=30, minRadius=1, maxRadius=40)

            check = 0

            if detected_circles is not None:
                detected_circles = np.uint16(np.around(detected_circles))

                for pt in detected_circles[0, :]:
                    a, b, r = pt[0], pt[1], pt[2]

                    if (a > x and a < x + w) and (b > y and b < y + h):
                        check += 1

            if check < 2:
                continue

            startX = int(x)
            startY = int(y)
            endX = int((x + w))
            endY = int((y + h))

            cv2.rectangle(frame, (startX, startY),
                          (endX, endY), (0, 255, 0), 2)

            tracker = dlib.correlation_tracker()
            rect = dlib.rectangle(startX, startY, endX, endY)
            tracker.start_track(frame, rect)

            trackers.append(tracker)

    else:

        for tracker in trackers:

            tracker.update(frame)
            pos = tracker.get_position()

            startX = int(pos.left())
            startY = int(pos.top())
            endX = int(pos.right())
            endY = int(pos.bottom())

            cv2.rectangle(frame, (startX, startY),
                          (endX, endY), (0, 255, 0), 2)
            rects.append((startX, startY, endX, endY))

    # uzyj "centroid tracker" zeby powiazac stare "object centroids"
    # z nowym "object centroids"
    objects = ct.update(rects)

    # przejdz po sledzonych objektach
    for (objectID, centroid) in objects.items():

        to = trackableObjects.get(objectID, None)

        if to is None:
            to = TrackableObject(objectID, centroid)
        else:
            to.centroids.append(centroid)

            if not to.counted:
                counter += 1
                to.counted = True

        trackableObjects[objectID] = to

    firstFrame = gray

    text = "Counter: {}".format(counter)
    cv2.putText(frame, text, (10, H - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):
        break

    totalFrames += 1

vs.release()
cv2.destroyAllWindows()

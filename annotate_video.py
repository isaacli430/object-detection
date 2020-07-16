import cv2, numpy as np, time

config = "yolov3.cfg"
weights = "yolov3.weights"


with open("coco.names") as f:
    labels = f.read().strip().split("\n")

colors = np.random.randint(0, 255, size=(len(labels), 3), dtype="uint8")

CONFIDENCE = 0.5
SCORE_THRESHOLD = 0.5
IOU_THRESHOLD = 0.5

#setup net
net = cv2.dnn.readNetFromDarknet(config, weights)
ln = net.getLayerNames()
ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]


#setup video
video = cv2.VideoCapture("rice_camera2.mp4")
writer = None
(W, H) = (None, None)

prop = cv2.CAP_PROP_FRAME_COUNT
total = int(video.get(prop))
print(total)

#process video frame by frame
while True:
    (grabbed, frame) = video.read()
    if not grabbed:
        break
    if W is None or H is None:
        (H, W) = frame.shape[:2]

    #feed frame into net
    blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)
    start = time.time()
    layerOutputs = net.forward(ln)
    end = time.time()

    boxes = []
    confidences = []
    classIDs = []

    for output in layerOutputs:
        for detection in output:
            scores = detection[5:]
            classID = np.argmax(scores)
            confidence = scores[classID]

            #assign boxes
            if confidence > CONFIDENCE:
                box = detection[0:4] * np.array([W, H, W, H])
                (centerX, centerY, width, height) = box.astype("int")

                x = int(centerX - (width / 2))
                y = int(centerY - (height / 2))

                boxes.append([x, y, int(width), int(height)])
                confidences.append(float(confidence))
                classIDs.append(classID)
    idxs = cv2.dnn.NMSBoxes(boxes, confidences, CONFIDENCE, SCORE_THRESHOLD)

    #draw boxes
    if len((idxs)) > 0:
        for i in idxs.flatten():
            (x, y) = (boxes[i][0], boxes[i][1])
            (w, h) = (boxes[i][2], boxes[i][3])

            color = [int(c) for c in colors[classIDs[i]]]
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            text = "{}: {:.4f}".format(labels[classIDs[i]], confidences[i])
            cv2.putText(frame, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    #Setup writer
    if writer is None:
        fourcc = cv2.VideoWriter_fourcc(*"MJPG")
        writer = cv2.VideoWriter("new_video.avi", fourcc, 30, (frame.shape[1], frame.shape[0]), True)

        if total > 0:
            elap = (end - start)
            print("[INFO] single frame took {:.4f} seconds".format(elap))
            print("[INFO] estimated total time to finish: {:.4f}".format(elap * total))
    writer.write(frame)

writer.release()
video.release()



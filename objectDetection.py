import cv2
from ultralytics import YOLO

def getColours(cls_num):
    base_colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]
    color_index = cls_num % len(base_colors)
    increments = [(1, -2, 1), (-2, 1, -1), (1, -1, 2)]
    color = [base_colors[color_index][i] + increments[color_index][i] * 
    (cls_num // len(base_colors)) % 256 for i in range(3)]
    return tuple(color)

yolo = YOLO('yolov8s.pt')
videoCap = cv2.VideoCapture(0)

while True:
    ret, frame = videoCap.read()
    if not ret:
        continue
    results = yolo.track(frame, stream=True)

    for result in results:
        ## get the class names
        classes_names = result.names

        ## iterate over each box that is found
        for box in result.boxes:

            ## if the confidence is larger than 40 percent
            if box.conf[0] > 0.4:
                ## get the coordinates of the bounding box
                [x1, y1, x2, y2] = box.xyxy[0]
                ## convert coordinates to ints
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

                # get the specific class and class name
                cls = int(box.cls[0])
                class_name = classes_names[cls]
                colour = getColours(cls)

                ## draw rectangle and put text
                cv2.rectangle(frame, (x1, y1), (x2, y2), colour, 2)
                cv2.putText(frame, f'{classes_names[int(box.cls[0])]}', (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 1, colour, 2)

    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

videoCap.release()
cv2.destroyAllWindows
        



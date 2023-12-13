import cv2 as cv
import numpy as np
import time 
import mediapipe as mp
import controller as cnt 

KNOWN_DISTANCE = 11.43  
VEHICLE_WIDTH = 70     

CONFIDENCE_THRESHOLD = 0.2
NMS_THRESHOLD = 0.1

HIGHLIGHT_COLOR = (0, 255, 0)
DISTANCE_BOX_COLOR = (0, 0, 0)
ALERT_COLOR = (0, 0, 255)

yoloNet = cv.dnn.readNet('yolov4-tiny.weights', 'yolov4-tiny.cfg')

yoloNet.setPreferableBackend(cv.dnn.DNN_BACKEND_CUDA)
yoloNet.setPreferableTarget(cv.dnn.DNN_TARGET_CUDA_FP16)

model = cv.dnn_DetectionModel(yoloNet)
model.setInputParams(size=(416, 416), scale=1/255, swapRB=True)

cap = cv.VideoCapture(0)  # Use the default webcam (usually index 0)

while True:
    ret, frame = cap.read()
    
    if not ret:
        break

    # Detect objects on the road
    classes, scores, boxes = model.detect(frame, CONFIDENCE_THRESHOLD, NMS_THRESHOLD)

    if classes is not None:
        for i in range(len(classes)):
            class_id = int(classes[i])
            score = float(scores[i])
            box = boxes[i]
            
            
            if class_id in [0, 1, 2]:  
                distance = (VEHICLE_WIDTH * KNOWN_DISTANCE) / box[2]

                object_color = HIGHLIGHT_COLOR
                text_color = (0, 0, 0)  # Black
                if distance < 1:  
                    object_color = ALERT_COLOR
                    text_color = (0, 0, 255)  # Red

                cv.rectangle(frame, box, object_color, 1)

                distance_text = f"{round(distance, 2)} m"
                text_size, _ = cv.getTextSize(distance_text, cv.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                text_x = box[0] + int((box[2] - text_size[0]) / 2)
                text_y = box[1] - 10
                cv.rectangle(frame, (text_x - 2, text_y - text_size[1] - 2),
                             (text_x + text_size[0] + 2, text_y + 2), DISTANCE_BOX_COLOR, -1)
                cv.putText(frame, distance_text, (text_x, text_y),
                           cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                cnt.control_servo_based_on_distance(distance)
                if distance < 1:
                    alert_text = "Break"
                    alert_size, _ = cv.getTextSize(alert_text, cv.FONT_HERSHEY_SIMPLEX, 0.5, 2)
                    alert_x = box[0] + int((box[2] - alert_size[0]) / 2)
                    alert_y = box[1] + box[3] + 20
                    cv.rectangle(frame, (alert_x - 2, alert_y - alert_size[1] - 2),
                                 (alert_x + alert_size[0] + 2, alert_y + 2), DISTANCE_BOX_COLOR, -1)
                    cv.putText(frame, alert_text, (alert_x, alert_y),
                               cv.FONT_HERSHEY_SIMPLEX, 0.4, text_color, 1, cv.LINE_AA)

    cv.imshow('Road Object Detection', frame)
    
    if cv.waitKey(1) == ord('q'):
        break

cap.release()
cv.destroyAllWindows()

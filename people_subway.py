
from ultralytics import YOLO
import cv2 as cv
import argparse
import matplotlib.pyplot as plt
import math
import numpy as np
from sort import *
import imutils

import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'


parser = argparse.ArgumentParser('run webcamera')
parser.add_argument('-w', '--webcam', type = int, help ='number of webcam')
parser.add_argument('-m', '--model_dir', type = str, help ='model directory')
parser.add_argument('-v', '--video_dir', type = str, help ='video directory')
parser.add_argument('-mask', '--mask_dir', type = str, required = False, help ='mask directory') # make mask an optional arg

args = parser.parse_args()


def draw_border(img, pt1, pt2, color, thickness, r, d):
    x1,y1 = pt1
    x2,y2 = pt2

    # Top left
    cv.line(img, (x1 + r, y1), (x1 + r + d, y1), color, thickness)
    cv.line(img, (x1, y1 + r), (x1, y1 + r + d), color, thickness)
    cv.ellipse(img, (x1 + r, y1 + r), (r, r), 180, 0, 90, color, thickness)

    # Top right
    cv.line(img, (x2 - r, y1), (x2 - r - d, y1), color, thickness)
    cv.line(img, (x2, y1 + r), (x2, y1 + r + d), color, thickness)
    cv.ellipse(img, (x2 - r, y1 + r), (r, r), 270, 0, 90, color, thickness)

    # Bottom left
    cv.line(img, (x1 + r, y2), (x1 + r + d, y2), color, thickness)
    cv.line(img, (x1, y2 - r), (x1, y2 - r - d), color, thickness)
    cv.ellipse(img, (x1 + r, y2 - r), (r, r), 90, 0, 90, color, thickness)

    # Bottom right
    cv.line(img, (x2 - r, y2), (x2 - r - d, y2), color, thickness)
    cv.line(img, (x2, y2 - r), (x2, y2 - r - d), color, thickness)
    cv.ellipse(img, (x2 - r, y2 - r), (r, r), 0, 0, 90, color, thickness)
    

def run_yolo(video_dir, model_weights, mask_dir):
    
    cap = cv.VideoCapture(video_dir)   #for webcam: 0 if you have  single Webcam; if multiple - choose for e.g 1

    width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
    size = (width, height)
    
    #  add VideoWriter for output video
    output_dir = video_dir[:-4] +'_output.mp4'
    out = cv.VideoWriter(output_dir, -1, 20.0, size)

    
    model = YOLO(model_weights)
    
    mask = None
    if mask_dir:
        mask = cv.imread(mask_dir)
    
    names_dict = model.model.names
    names = list(names_dict.values())
    colors = [tuple([np.random.randint(0, 255) for _ in range(3)]) for _ in range(len(names))] 
    class_color_dict = {k: v for k, v in zip(names, colors)}

    
    #Tracking
    tracker = Sort(max_age =20, min_hits=3, iou_threshold = 0.3)
    count= []
    
    
    
    while True:
        isTrue, frame = cap.read()
        if isTrue:
            frame = imutils.resize(frame, width=1280, height=720)

            if mask is not None:
                masked_frame = cv.bitwise_and(frame, mask)  #mask
                results = model(masked_frame , stream =True)

            else:
                results = model(frame, stream =True)

            detections = np.empty((0,5))
            
            
            for r in results:
                boxes = r.boxes
                for box in boxes:
                    # Bbox 
                    x1, y1, x2, y2 = box.xyxy[0]  # or box.xywh
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)   #x1, y1 - top left point, x2, y2 - bottom right

                    #conf (p_c)
                    conf = math.ceil(box.conf[0]*100)/100

                    #class/label 
                    label = names_dict[int(box.cls[0])]
                    
                    if label == 'person' and conf>0.25:
                        #append detection
                        detections = np.vstack(([x1, y1, x2, y2, conf], detections))
        
        
            results_tracking = tracker.update(detections) # x1, y1,x2, y2, tracker_ID

            # line for counting
            limits = [140, 500, 1200, 500]
            cv.line(frame, (limits[0], limits[1]), (limits[2], limits[3]), (0,0,255), thickness=3)
            
            color = (0, 255, 0)
            
            for r in results_tracking:

                x1, y1, x2, y2, Id = r
                x1, y1, x2, y2, Id = int(x1), int(y1), int(x2), int(y2), int(Id)
                #print('tracking:', r)
                
                
                #draw bbox
                draw_border(frame, (x1, y1), (x2, y2), color, 3, 5, 10)

                cv.putText(frame, f'{label} {conf} {Id}', (max(0, x1), max(35, y1)),
                                   cv.FONT_HERSHEY_COMPLEX_SMALL, 1, (255, 255, 255), 2)

                # center of bbox
                w, h = x2- x1, y2-y1
                cx, cy = x1 +w//2, y1+h//2
                cv.circle(frame, (cx, cy), radius= 5, color= color, thickness= cv.FILLED)

                # check if center of bbox touches the 'line'
                if  limits[0] < cx < limits[2] and limits[1]-15 <cy<limits[1] +15:
                    if Id not in count:
                        count.append(Id)
                        #when line 'detects' Id it becomes green
                        cv.line(frame, (limits[0], limits[1]), (limits[2], limits[3]), (0,255,0), thickness=3)


            #show count
            cv.putText(frame, f'People Count: {len(count)}', (50, 50), cv.FONT_HERSHEY_SCRIPT_COMPLEX, 1, (225, 40, 10), 2)

            out.write(frame)
            cv.imshow('video', frame)

            cv.waitKey(1)
            
            
        else:
            break

    cap.release()
    cv.destroyAllWindows()
    
    

if __name__=='__main__':
    run_yolo(args.video_dir, args.model_dir, args.mask_dir)

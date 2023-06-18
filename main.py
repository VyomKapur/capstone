import cv2
from ultralytics import YOLO
import supervision as sv


def main():
    model = YOLO("yolov8l.pt")
    
    boxannotator = sv.BoxAnnotator(
        thickness=2,
        text_thickness=2,
        text_scale=1
    )
    inside_outside = {}
    inside_cart = {}
    for result in model.track(source=0, stream=True):
        frame = result.orig_img
        detections = sv.Detections.from_yolov8(result)        
        
        if result.boxes.id is not None:
            detections.tracker_id = result.boxes.id.cpu().numpy().astype(int)
        
        detections = detections[detections.class_id!=0]
        
        labels = [
            f"{tracker_id} {model.model.names[class_id]} {confidence: 0.2f}"
            for _, confidence, class_id, tracker_id
            in detections
        ]

        frame = boxannotator.annotate(
            scene=frame, 
            detections=detections,
            labels=labels
        )

        cv2.line(frame, (0, 240), (640, 240), (255, 255, 255), thickness=2)
        print(detections)
        for detection in detections:
            if detection[1] == 0:
                continue

            if detection[3] not in inside_outside.keys():
                if detection[0][1] + detection[0][3] // 2 < 240:
                    inside_outside[detection[3]] = 1
                    inside_cart[detection[2]] = inside_cart.get(detection[2], 0) + 1
                
            else:
                if detection[0][1] + detection[0][3] // 2 > 240 and inside_outside[detection[3]] == 1:
                    inside_outside[detection[3]] = 0
                    if detection[2] in inside_cart.keys():
                        inside_cart[detection[2]] -= 1
                    if inside_cart[detection[2]] < 1:
                        del inside_cart[detection[2]]
                    

        print(inside_outside)
        
        print(inside_cart)

        cv2.imshow("capture", frame)
        if(cv2.waitKey(30) == 27):
            break

if __name__ == "__main__":
    main()
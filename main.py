import cv2
from ultralytics import YOLO
import supervision as sv

MODEL_NAME = "yolov8s.pt"

class detected_obj:
    def __init__(self, detection):
        self.y_mid = (detection[0][1] + detection[0][3]) / 2
        self.xyxy = detection[0]
        self.confidence = detection[1]
        self.class_id = detection[2]
        self.tracker_id = detection[3]
        self.id = f"{self.class_id}_{self.tracker_id}"
        self.inside = 0
        self.gradient = 0

def model_import(file_name):
    return YOLO(file_name)    

def get_boxannotator() -> sv.BoxAnnotator:
    return sv.BoxAnnotator(
        thickness=2,
        text_thickness=2,
        text_scale=1
    )

def get_labels(detections, model) -> list[str]:
    return [
            f"{tracker_id} {model.model.names[class_id]} {confidence: 0.2f}"
            for _, confidence, class_id, tracker_id
            in detections
        ]

def get_annotations(boxannotator, frame, detections, labels):
    return boxannotator.annotate(
            scene=frame, 
            detections=detections,
            labels=labels
        )

def main():
    
    model = model_import(MODEL_NAME)

    boxannotator = get_boxannotator()
    inside_cart = {}
    objs = {}
    for result in model.track(source=0, stream=True):

        frame = result.orig_img
        detections = sv.Detections.from_yolov8(result)        
        
        if result.boxes.id is not None:
            detections.tracker_id = result.boxes.id.cpu().numpy().astype(int)
        
        detections = detections[detections.class_id!=0]
        
        labels = get_labels(detections, model)

        frame = get_annotations(boxannotator, frame, detections, labels)

        cv2.line(frame, (0, 240), (640, 240), (0, 0, 0), thickness=3)
        
        for detection in detections:
            obj = detected_obj(detection)

            if obj.id not in objs.keys():
                objs[obj.id] = obj

            else:
                old_obj = objs[obj.id]
                if obj.y_mid > 240 and old_obj.y_mid < 240:
                    # incoming
                    inside_cart[obj.class_id] = inside_cart.get(obj.class_id, 0) + 1

                elif obj.y_mid < 240 and old_obj.y_mid > 240:
                    inside_cart[obj.class_id] = inside_cart.get(obj.class_id, 0) -1
                    if inside_cart[obj.class_id] < 1:
                        del inside_cart[obj.class_id]
                    # outgoing                
                objs[obj.id] = obj

        print(inside_cart)
        cv2.imshow("capture", frame)
        if(cv2.waitKey(30) == 27): 
            break

if __name__ == "__main__":
    main()
import os
import cv2
import pymongo
import supervision as sv
from ultralytics import YOLO
from dotenv import load_dotenv
from detected_obj import detected_obj


load_dotenv(".env")

MODEL_NAME = os.environ.get("MODEL_NAME")
CART_ID = os.environ.get("CART_ID")
CONNECTION_STRING = os.environ.get("CONNECTION_STRING")


class smart_cart:
    def __init__(self):
        self.model = self.model_import(MODEL_NAME)
        self.client = self.get_client()
        db = self.client["cartData"]
        self.collection = db[CART_ID]
        self.boxannotator = self.get_boxannotator()
        self.inside_cart = {}
        self.objs = {}
        self.price_list = {"bottle": 20, "cell phone": 50}
        self.weight_list = {"bottle": 0.4, "cell phone": 0.5}
    
    def model_import(self, file_name):
        return YOLO(file_name)    

    def get_client(self):
        return pymongo.MongoClient(CONNECTION_STRING)

    def get_boxannotator(self) -> sv.BoxAnnotator:
        return sv.BoxAnnotator(
            thickness=2,
            text_thickness=2,
            text_scale=1
        )

    def get_labels(self, detections, model) -> list[str]:
        return [
                f"{tracker_id} {model.model.names[class_id]} {confidence: 0.2f}"
                for _, confidence, class_id, tracker_id
                in detections
            ]

    def get_annotations(self, boxannotator, frame, detections, labels):
        return boxannotator.annotate(
                scene=frame, 
                detections=detections,
                labels=labels
            )

    def remove_and_update(self, obj):
        query = {
            "class_id": obj.class_id,
        }
        update = {
            "$inc": {"quantity": -1}
        }
        self.collection.update_one(query, update)
        
        remove_query = {
            "class_id": obj.class_id,
            "quantity": {"$lte": 0}
        }

        self.collection.delete_one(remove_query)

    def insert_and_update(self, obj):
        price = 0
        weight = 0
        if self.model.model.names[obj.class_id] in self.price_list.keys():
            price = self.price_list[self.model.model.names[obj.class_id]]
        if self.model.model.names[obj.class_id] in self.weight_list.keys():
            weight = self.weight_list[self.model.model.names[obj.class_id]]
        query = {
            "class_id": obj.class_id,
            "name": self.model.model.names[obj.class_id],
            "price": price,
            "weight": weight
        }
        update = {
            "$inc": {"quantity": 1}
        }
        self.collection.update_one(query, update, upsert=True)

    def track_and_run(self):
        frame_ctr = 0
        for result in self.model.track(source=0, stream=True):
            frame_ctr += 1
            frame = result.orig_img
            detections = sv.Detections.from_yolov8(result)        
            if result.boxes.id is not None:
                detections.tracker_id = result.boxes.id.cpu().numpy().astype(int)
            
            detections = detections[detections.class_id != 0]
            labels = self.get_labels(detections, self.model)
            frame = self.get_annotations(self.boxannotator, frame, detections, labels)
            cv2.line(frame, (0, 240), (640, 240), (0, 0, 0), thickness=3)
            
            for detection in detections:
                obj = detected_obj(detection)

                if obj.id not in self.objs.keys():
                    self.objs[obj.id] = obj
                else:
                    old_obj = self.objs[obj.id]
                    if obj.y_mid > 240 and old_obj.y_mid < 240:
                        # incoming
                        self.inside_cart[obj.class_id] = self.inside_cart.get(obj.class_id, 0) + 1
                        print(int(obj.class_id))
                        # Update quantity or insert new item into collection
                        self.insert_and_update(obj)

                    elif obj.y_mid < 240 and old_obj.y_mid > 240:
                        
                        # outgoing 

                        self.inside_cart[obj.class_id] = self.inside_cart.get(obj.class_id, 0) - 1
                        if self.inside_cart[obj.class_id] < 1:
                            del self.inside_cart[obj.class_id]
                        print(type(obj.class_id))
                        
                        # Update quantity or remove item from collection
                        self.remove_and_update(obj)

                    self.objs[obj.id] = obj

            print(self.inside_cart)
            cv2.imshow("capture", frame)
            if cv2.waitKey(30) == 27:
                break
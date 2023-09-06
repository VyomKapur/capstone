class detected_obj:
    def __init__(self, detection):
        self.xyxy = detection[0]
        self.y_mid = (self.xyxy[1] + self.xyxy[3]) / 2
        self.confidence = detection[1]
        self.class_id = int(detection[2])
        self.tracker_id = detection[3]
        self.id = f"{self.class_id}_{self.tracker_id}"
    
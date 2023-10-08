from smart_cart import smart_cart


if __name__ == "__main__":
    cart1 = smart_cart()
    cart1.track_and_run()

    
# import keras
# from keras import backend as K
# from keras.layers import Dense, Activation
# from keras.optimizers import Adam
# from keras.metrics import categorical_crossentropy
# from keras.preprocessing.image import ImageDataGenerator
# from keras.preprocessing import image
# from keras.models import Model
# from keras.applications import imagenet_utils
# from keras.layers import Dense,GlobalAveragePooling2D
# from keras.applications import MobileNe
# from keras.applications.mobilenet import preprocess_input
# import numpy as np
# from IPython.display import Image
# from keras.optimizers import Adam
# import cv2


# mobile = keras.applications.mobilenet_v2.MobileNetV2()
# def prepare_image(img):
#     img_array = image.img_to_array(img)
#     img_array_expanded_dims = np.expand_dims(img_array, axis=0)
#     return keras.applications.mobilenet_v2.preprocess_input(img_array_expanded_dims)

# cap = cv2.VideoCapture(0)

# while True:
#     ret, frame = cap.read()
#     frame = cv2.resize(frame, (224, 224))
#     preprocessed_image = prepare_image(frame)
#     predictions = mobile.predict(preprocessed_image)
#     results = imagenet_utils.decode_predictions(predictions)
#     print(results)
#     cv2.imshow('frame',frame)
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break
    
# cap.release()
# cv2.destroyAllWindows()

# import numpy as np
# import cv2

# # download the model as plain text as a PROTOTXT file and the trained model as a CAFFEMODEL file from  here: https://github.com/djmv/MobilNet_SSD_opencv

# # path to the prototxt file with text description of the network architecture
# prototxt = "MobileNetSSD_deploy.prototxt"
# # path to the .caffemodel file with learned network
# caffe_model = "MobileNetSSD_deploy.caffemodel"

# # read a network model (pre-trained) stored in Caffe framework's format
# net = cv2.dnn.readNetFromCaffe(prototxt, caffe_model)

# # dictionary with the object class id and names on which the model is trained
# classNames = { 0: 'background',
#     1: 'aeroplane', 2: 'bicycle', 3: 'bird', 4: 'boat',
#     5: 'bottle', 6: 'bus', 7: 'car', 8: 'cat', 9: 'chair',
#     10: 'cow', 11: 'diningtable', 12: 'dog', 13: 'horse',
#     14: 'motorbike', 15: 'person', 16: 'pottedplant',
#     17: 'sheep', 18: 'sofa', 19: 'train', 20: 'tvmonitor'}

# trackers = cv2.MultiTracker_create()
# cap = cv2.VideoCapture(0)
# object_trackers = {}

# while True:
#     ret, frame = cap.read()
    
#     # size of image
#     width = frame.shape[1] 
#     height = frame.shape[0]
#     # construct a blob from the image
#     blob = cv2.dnn.blobFromImage(frame, scalefactor = 1/127.5, size = (300, 300), mean = (127.5, 127.5, 127.5), swapRB=True, crop=False)
#     # blob object is passed as input to the object
#     net.setInput(blob)
#     # network prediction
#     detections = net.forward()

#     objects_to_remove = []

#     for object_id in object_trackers.keys():
#         tracker, bbox = object_trackers[object_id]

#         # Update the tracker with the current frame
#         success, new_bbox = tracker.update(frame)

#         if success:
#             (x, y, w, h) = [int(v) for v in new_bbox]
#             cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
#         else:
#             # If tracking is lost, mark the object for removal
#             objects_to_remove.append(object_id)

#     # Remove objects that are no longer tracked
#     for object_id in objects_to_remove:
#         del object_trackers[object_id]

#     for i in range(detections.shape[2]):
#         confidence = detections[0, 0, i, 2]
#         if confidence > 0.5:
#             class_id = int(detections[0, 0, i, 1])
#             x_top_left = int(detections[0, 0, i, 3] * width)
#             y_top_left = int(detections[0, 0, i, 4] * height)
#             x_bottom_right = int(detections[0, 0, i, 5] * width)
#             y_bottom_right = int(detections[0, 0, i, 6] * height)

#             # cv2.rectangle(frame, (x_top_left, y_top_left), (x_bottom_right, y_bottom_right), (0, 255, 0))

#             if class_id in classNames:
#                 label = classNames[class_id] + ": " + str(confidence)
#                 (w, h), t = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
#                 y_top_left = max(y_top_left, h)
#                 # cv2.rectangle(frame, (x_top_left, y_top_left - h), (x_top_left + w, y_top_left + t), (0, 0, 0), cv2.FILLED)
#                 cv2.putText(frame, label, (x_top_left, y_top_left), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0))

#             # Check if the object is already being tracked
#             object_id = class_id
#             if object_id in object_trackers:
#                 continue

#             # Initialize a new tracker for this object
#             tracker = cv2.TrackerKCF_create()
#             bbox = (x_top_left, y_top_left, x_bottom_right - x_top_left, y_bottom_right - y_top_left)
#             tracker.init(frame, bbox)

#             # Store the tracker and bbox in the object_trackers dictionary
#             object_trackers[object_id] = (tracker, bbox)

#     cv2.namedWindow("frame", cv2.WINDOW_NORMAL)
#     cv2.imshow("frame", frame)
#     if cv2.waitKey(1) >= 0:
#         break

# cap.release()
# cv2.destroyAllWindows()
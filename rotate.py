import numpy as np
import cv2
import os
import time
prototxtPath = "deploy.prototxt"
weightsPath = "res10_300x300_ssd_iter_140000.caffemodel"
net = cv2.dnn.readNet(prototxtPath, weightsPath)
confidence_with_direction = {}
start_time = time.time()
print("[INFO] Start get confidence in each direction")
for degree in ["",cv2.ROTATE_180,cv2.ROTATE_90_CLOCKWISE, cv2.ROTATE_90_COUNTERCLOCKWISE]:
    print("[INFO] Start get confidence in degree: {}".format(degree))
    images_list = sorted(os.listdir("images"))
    found_face = {}
    confidences = {}
    count = 0
    for path in images_list:
        image = cv2.imread(os.path.join("images",path))
        if degree != "":
            image = cv2.rotate(image, degree)
        (h, w) = image.shape[:2]
        blob = cv2.dnn.blobFromImage(image, 1.0, (224, 224), (104.0, 177.0, 123.0))
        net.setInput(blob)
        detections = net.forward()

        id = path.split("_")[0]
        if id not in found_face:
            found_face[id] = []
        if id not in confidences:
            confidences[id] = []
        if id not in confidence_with_direction:
            confidence_with_direction[id] = {}

        for i in range(0, detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > 0.98:
                confidences[id].append(confidence)
                found_face[id].append(path)
        count += 1
        if count % 100 == 0:
            print("[INFO] {}/{} images checked".format(count, len(images_list)))
    print("[INFO] Start Checking Found Face in each Image")
    for human in found_face:
        length = len(found_face[human])
        if length == 4:
            if degree in confidence_with_direction[human]:
                confidence_with_direction[human][degree] = max(np.mean(confidences[human]),confidence_with_direction[human][degree])
            else:
                confidence_with_direction[human][degree] = np.mean(confidences[human])
print("[INFO] Start rotate and save images")
images_list = sorted(os.listdir("images"))
count = 0
for img_path in images_list:
    path = img_path.split("_")[0]
    if path in confidence_with_direction and len(confidence_with_direction[path]) >= 1:
        degree = max(confidence_with_direction[path], key=confidence_with_direction[path].get)
        image = cv2.imread(os.path.join("images",img_path))
        if degree != "":
            image = cv2.rotate(image, degree)
        cv2.imwrite(os.path.join("output",img_path), image)
        os.remove(os.path.join("images",img_path))
        count += 1
        if count % 100 == 0:
            print("[INFO] {}/{} images rotated".format(count, 10000))
        if count == 10000:
            break
    else:
        print("[ERROR] Cannot Process Image: {}".format(img_path))
end_time = time.time()
print("[INFO] Finish rotate and save images")
print("[INFO] Time: {} s".format(end_time - start_time))
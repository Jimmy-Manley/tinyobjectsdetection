from ultralytics import YOLO

import numpy as np
# Load a custom YOLO model
model = YOLO("./last.pt")
print(model.task)
# Predict on an image
results = model("/home/jman/Documents/PROYECTO/Detección_particulas/Datasets/Kaggle/archive/Small Object dataset/test/honeybee/img/honeybee009.jpg")

# Get class names
names_dict = model.names  # safer than results[0].names

probs = results[0].probs.numpy()# Tensor of class probabilities

print(probs)

probs_array = results[0].probs.data.cpu().numpy()  # safely convert to NumPy

print(names_dict[np.argmax(probs_array)])

print(names_dict)
# Get confidences for detected objects
#confidences = results[0].boxes.conf.tolist()  # list of confidence scores
#classes = results[0].boxes.cls.tolist()       # list of predicted class indices

# Map class indices to names
#class_names = [names_dict[int(cls)] for cls in classes]

# Print predictions
#for name, conf in zip(class_names, confidences):
#    print(f"Detected {name} with confidence {conf:.2f}")

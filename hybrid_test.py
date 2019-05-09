from joblib import dump, load

import os
import numpy as np
from keras.models import load_model
from keras.preprocessing.image import img_to_array, load_img
from keras.applications.inception_v3 import preprocess_input

model = load_model("hybrid.h5")
svmFit = load('hybrid_svmFit.joblib')
data = []

folderPath = "/Users/lvz/PycharmProjects/pothole_classifier/test"
for imageName in os.listdir(folderPath):
    if imageName != ".DS_Store":
        print("[INFO] classifying {}".format(imageName))
        image = load_img(folderPath + "/" + imageName, target_size=(299, 299))
        img_data = img_to_array(image)
        img_data = np.expand_dims(img_data, axis=0)

        # extract a color histogram feature from the image
        img_data = preprocess_input(img_data)
        feature = model.predict(img_data)
        feature_np = np.array(feature)
        data.append(feature_np.flatten())

    else:
        continue

labels = svmFit.predict(data)
print(labels)

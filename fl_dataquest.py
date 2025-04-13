import numpy as np
import random
import cv2
import os
from imutils import paths # https://github.com/PyImageSearch/imutils

import matplotlib.pyplot as plt

import sklearn as skl
from sklearn.utils import shuffle
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split

import tensorflow as tf

def load_and_preprocess(paths, verbose=None):
    data = []
    labels = []
    for (i, imgpath) in enumerate(paths):
        im_gray = cv2.imread(imgpath, cv2.IMREAD_GRAYSCALE)
        image = np.array(im_gray, dtype=np.float32)
        image = image / 255.0   # Normalisation [0,1]
        data.append(image)

        # Label = nom du dossier parent
        label = imgpath.split(os.path.sep)[-2]
        labels.append(label)

        if verbose is not None and verbose > 0 and i > 0 and (i + 1) % verbose == 0:
            print("[INFO] processed {}/{}".format(i + 1, len(paths)))
    return data, labels

def get_data(img_path='./Mnist/trainingSet/trainingSet/',verbose=1):
    # 1) Récupération chemins d'images
    image_paths = list(paths.list_images(img_path))
    if verbose > 0:
        print(f"Nombre total d'images trouvées : {len(image_paths)}")

    # 2) Chargement
    il, ll = load_and_preprocess(image_paths, verbose=10000)

    # 3) One-hot encoding des labels (0,1,2,...)
    lb = skl.preprocessing.LabelBinarizer()
    ll = lb.fit_transform(ll)

    # 4) Split (train / test)
    X_train, X_test, y_train, y_test = train_test_split(il, ll, test_size=0.1, random_state=19)

    # 5) Contrôle : forme attendue (28x28)
    input_shape = X_train[0].shape  # (28,28) normal
    if verbose > 0:
        print("Train set size =", len(X_train), "Test set size =", len(X_test))
        print("Example input shape =", input_shape)

    return X_train, X_test, y_train, y_test, input_shape

def get_dataset(X_train, X_test, y_train, y_test, batch_size=32, verbose=0):
    # Création dataset pour TF
    dtt = tf.data.Dataset.from_tensor_slices((X_train, y_train))
    dtt = dtt.shuffle(buffer_size=len(y_train)).batch(batch_size)

    dts = tf.data.Dataset.from_tensor_slices((X_test, y_test))
    dts = dts.batch(batch_size)

    if verbose>0:
        print("Train dataset shape:", dtt.element_spec)
        print("Test dataset shape:", dts.element_spec)
    return dtt, dts

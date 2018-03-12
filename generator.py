import os
import cv2
import numpy as np

DATASET_DIR = 'D:/CycleGAN/apple2orange'
IMG_SIZE = 128


def data_generator():
    folder_A = os.path.join(DATASET_DIR, 'trainA')
    images_A = []
    for filename in os.listdir(folder_A):
        img = cv2.imread(os.path.join(folder_A, filename))
        if img is not None:
            images_A.append(cv2.resize(img, (IMG_SIZE, IMG_SIZE)))

    folder_B = os.path.join(DATASET_DIR, 'trainB')
    images_B = []
    for filename in os.listdir(folder_B):
        img = cv2.imread(os.path.join(folder_B, filename))
        if img is not None:
            images_B.append(cv2.resize(img, (IMG_SIZE, IMG_SIZE)))

    print('All images loaded into memory.')
    while True:
        yield images_A[np.random.choice(len(images_A))],\
              images_B[np.random.choice(len(images_B))]
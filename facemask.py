import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Conv2D, MaxPooling2D, SpatialDropout2D, Flatten, Dropout, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import cv2
import datetime

# Define constants for image size and batch size
IMAGE_SIZE = (150, 150)
BATCH_SIZE = 16

# Build the CNN model
model = keras.Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3)),
    MaxPooling2D(),
    Conv2D(32, (3, 3), activation='relu'),
    MaxPooling2D(),
    Conv2D(32, (3, 3), activation='relu'),
    MaxPooling2D(),
    Flatten(),
    Dense(100, activation='relu'),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Create data generators for training and testing
train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)

test_datagen = ImageDataGenerator(rescale=1. / 255)

training_set = train_datagen.flow_from_directory(
    'train',
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='binary'
)

test_set = test_datagen.flow_from_directory(
    'test',
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='binary'
)

# Train the model
model.fit(
    training_set,
    epochs=2,
    validation_data=test_set,
)

# Save the model
model.save('mymodel.h5')

# Implement live detection of face mask
model = keras.models.load_model('mymodel.h5')

cap = cv2.VideoCapture(0)
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

while cap.isOpened():
    _, img = cap.read()

    # Reverse the frame horizontally
    img = cv2.flip(img, 1)

    faces = face_cascade.detectMultiScale(img, scaleFactor=1.1, minNeighbors=4)

    for (x, y, w, h) in faces:
        face_img = img[y:y + h, x:x + w]
        cv2.imwrite('temp.jpg', face_img)

        test_image = image.load_img('temp.jpg', target_size=IMAGE_SIZE)
        test_image = image.img_to_array(test_image)
        test_image = np.expand_dims(test_image, axis=0)
        pred = model.predict(test_image)[0][0]

        label = 'NO MASK' if pred == 1 else 'MASK'
        color = (0, 0, 255) if pred == 1 else (0, 255, 0)

        cv2.rectangle(img, (x, y), (x + w, y + h), color, 3)
        cv2.putText(img, label, ((x + w) // 2, y + h + 20), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 3)
        date_str = str(datetime.datetime.now())
        cv2.putText(img, date_str, (400, 450), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    cv2.imshow('img', img)

    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

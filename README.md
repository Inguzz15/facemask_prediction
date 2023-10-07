# Face-Mask Detector
Real time face-mask detection using Deep Learning and OpenCV

The provided code in facemask.py is a Python script that uses TensorFlow and OpenCV to create a simple face mask detection system using a Convolutional Neural Network (CNN) for image classification. Here's a step-by-step explanation of the code:

1. **Importing Libraries**:

    - The script starts by importing necessary libraries, including NumPy, TensorFlow, Keras, and OpenCV.

2. **Defining Constants**:

    - It defines two constants, `IMAGE_SIZE` and `BATCH_SIZE`, to specify the size of the images that the CNN model will process and the batch size for data generators.

3. **Building the CNN Model**:

    - The code defines a CNN model using Keras Sequential API. It consists of three convolutional layers followed by max-pooling layers, a flattening layer, and two fully connected (dense) layers.
    
4. **Compiling the Model**:

    - The model is compiled with the Adam optimizer and binary cross-entropy loss, which is common for binary classification problems. It also uses accuracy as a metric for evaluation.

5. **Creating Data Generators**:

    - Two ImageDataGenerator objects are created, `train_datagen` and `test_datagen`, to perform real-time data augmentation and normalization. These generators will preprocess the training and testing images.

6. **Loading Training and Testing Data**:

    - The script loads training and testing data using the `flow_from_directory` method from the data generators. The training and testing data should be organized in separate directories, and the generator loads data from these directories while applying the specified transformations.

7. **Training the Model**:

    - The model is trained using the training data generator with `model.fit()`. It runs for 2 epochs in this example, but you can adjust this number as needed.

8. **Saving the Model**:

    - After training, the model is saved to a file called "mymodel.h5" using `model.save()`.

9. **Loading the Model for Live Detection**:

    - The saved model is loaded again using `keras.models.load_model()` for live face mask detection.

10. **Live Detection Loop**:

    - The script enters a loop that captures frames from the default camera using OpenCV. It uses the Haar Cascade Classifier for face detection to identify faces in each frame.

    - For each detected face:
        - The face region is extracted and temporarily saved as "temp.jpg."
        - The image is loaded and preprocessed for the model.
        - The model predicts whether the person is wearing a mask or not.
        - A bounding box and label ("MASK" or "NO MASK") are drawn on the frame based on the prediction.
        - The current date and time are also displayed on the frame.

11. **Displaying the Frame**:

    - The modified frame with bounding boxes and labels is displayed using OpenCV's `cv2.imshow()`.

12. **Exiting the Loop**:

    - The loop continues until the 'q' key is pressed, at which point the script releases the camera and closes all OpenCV windows.

In summary, this code demonstrates a face mask detection system using a pre-trained CNN model and real-time camera feed processing. It identifies whether a person is wearing a mask and visually annotates the frame accordingly. The model is trained on a dataset of masked and unmasked faces, and the Haar Cascade Classifier is used for face detection.

## Dataset

The data used can be downloaded through this [link](https://data-flair.training/blogs/download-face-mask-data/)  There are 1314 training images and 194 test images divided into two catgories, with and without mask.

## How to Use

To use this project on your system, follow these steps:
``

1. Download all libaries using::
```
pip install -r requirements.txt
```

2. Run facemask.py by typing the following command on your Command Prompt:
```
python facemask.py
```




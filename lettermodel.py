import tensorflow as tf
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import cv2
import os
import datetime
from emnist import list_datasets
from tensorflow import keras
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import recall_score, precision_score, f1_score, accuracy_score



# Import the necessary functions from the emnist library
from emnist import extract_training_samples
from emnist import extract_test_samples


# Define the labels for the EMNIST dataset
LABELS = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 
          'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z',
          'a', 'b', 'd', 'e', 'f', 'g', 'h', 'n', 'q', 'r', 't']

save_frame = False

# Define a function to remodulate the labels
def remodulate(y):
    """
    Remodulates the input array 'y' based on the labels.

    Parameters:
    - y (numpy.ndarray): The input array to be remodulated.

    Returns:
    - numpy.ndarray: The remodulated array.
    """
    # Replace the labels 'T', 'B', 'Y', 'G' with 0, 1, 2, 3 respectively
    # Define a dictionary to map the labels to their new values
    label_map = {"T": 0, "B": 1, "Y": 2, "G": 3}

    # Loop through the label_map to replace the labels with their new values
    for label, value in label_map.items():
        y = np.where(y == LABELS.index(label), value, y)

    return y

# Define a function to filter the dataset based on specific label conditions
def filterDataset(X_data, y_data):
    """
    Filters the dataset based on lables we need to use (T, B, G, Y).

    Args:
        X_data (ndarray): Input data array of shape (n_samples, 28, 28).
        y_data (ndarray): Target labels array of shape (n_samples,).

    Returns:
        tuple: A tuple containing the filtered input data array and target labels array.

    Raises:
        AssertionError: If the dtype of y_data is not np.uint8.
    """
    # Assert that the dtype of y_data is np.uint8
    assert y_data.dtype == np.uint8
    # Define the classes
    classes = LABELS
    # Initialize the new_data_size
    new_data_size = 0
    # Initialize the new_data_index
    new_data_index = 0

    # Loop through the y_data to count the number of records with labels 'B', 'Y', 'G', 'T'
    for recordIndex in range(0, y_data.shape[0]):
        currentLabel = classes[y_data[recordIndex]]
        if currentLabel == 'B' or currentLabel == 'Y' or currentLabel == 'G' or currentLabel == 'T':
            new_data_size += 1
    
    # Initialize the new_X_data and new_y_data with zeros
    new_X_data = np.zeros((new_data_size, 28, 28), dtype = X_data.dtype)
    new_y_data = np.zeros((new_data_size,), dtype = np.uint8)

    # Loop through the y_data to filter the records with labels 'B', 'Y', 'G', 'T'
    for recordIndex in range(0, y_data.shape[0]):
        currentLabel = classes[y_data[recordIndex]]

        if currentLabel != 'B' and currentLabel != 'Y' and currentLabel != 'G' and currentLabel != 'T':
            continue
        
        new_X_data[new_data_index] = X_data[recordIndex]
        new_y_data[new_data_index] = y_data[recordIndex]
        new_data_index += 1
    
    # Assert that the new_data_index is equal to the shape of new_X_data
    assert new_data_index == new_X_data.shape[0]
    return (new_X_data, new_y_data)

def create_model():
    """
    Creates a convolutional neural network model for image classification.

    Returns:
        keras.Sequential: The created model.
    """
    seq_lett_model = keras.Sequential([
        keras.Input(shape=(28, 28, 1)), # Input shape: 28x28 pixels, 1 color channel
        keras.layers.Conv2D(28, (3, 3), activation='relu'),
        keras.layers.MaxPooling2D((2, 2)),
        keras.layers.Conv2D(128, (3, 3), activation='relu'),
        keras.layers.Dropout(0.5),
        keras.layers.MaxPooling2D((2, 2)),
        keras.layers.Conv2D(128, (3, 3), activation='relu'),
        keras.layers.Flatten(),
        keras.layers.Dense(512, activation='relu'),
        keras.layers.Dropout(0.5),
        keras.layers.Dense(4, activation='softmax')  # Output layer for 4 letters
    ])
    seq_lett_model.summary()
    return seq_lett_model

def model_statistics(training_operation, X_test, y_test, seq_lett_model):
    
    plot_path = f'./static/plots/'

    # Plot training & validation accuracy values
    plt.plot(training_operation.history['accuracy'])
    plt.plot(training_operation.history['val_accuracy'])
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Val'], loc='upper left')
    plt.savefig(os.path.join(plot_path, 'accuracy_plot.png'))  # Saves confusion matrix to the plots folder
    plt.close()


    # Plot training & validation loss values
    plt.plot(training_operation.history['loss'])
    plt.plot(training_operation.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Val'], loc='upper left')
    plt.savefig(os.path.join(plot_path, 'loss_plot.png'))  # Imposta la risoluzione a 300 DPI
    plt.close()

    # Use the trained model to make predictions on the test data.
    # The predict function returns the output of the last layer in the model, which in this case is the output of the softmax layer.
    y_pred = seq_lett_model.predict(X_test)

    # The output of the softmax layer is a vector of probabilities for each class. 
    # We use the argmax function to find the index of the maximum probability, which gives us the predicted class.
    y_pred = np.argmax(y_pred, axis = 1)

    # Compute the confusion matrix to evaluate the accuracy of the classification.
    # The confusion matrix is a table that is often used to describe the performance of a classification model.
    # Each row of the matrix represents the instances in a predicted class, while each column represents the instances in an actual class.
    # The 'normalize' parameter is set to 'true', which means the confusion matrix will be normalized by row (i.e., by the number of samples in each class).
    confusionMatrix = confusion_matrix(y_test, y_pred, normalize = 'true')

    # Create a ConfusionMatrixDisplay object from the confusion matrix.
    # The display_labels parameter is set to the names of the classes.
    disp = ConfusionMatrixDisplay(confusion_matrix = confusionMatrix, display_labels = ['T','B','Y','G'])

    # Plot the confusion matrix.
    disp.plot()

    # Saves confusion matrix to the plots folder
    plt.savefig(os.path.join(plot_path, 'confusion_matrix.png')) 

    from sklearn.metrics import classification_report
    #class_report = classification_report(y_test, y_pred)
    #print(class_report)
    metrics_per_class = {}

    for classe in set(y_test):
        precision = round(precision_score(y_test, y_pred, labels=[classe], average='micro'), 2)
        recall = round(recall_score(y_test, y_pred, labels=[classe], average='micro'), 2)
        f1 = round(f1_score(y_test, y_pred, labels=[classe], average='micro'), 2)
        accuracy = round(accuracy_score(y_test, y_pred), 2)

        metrics_per_class[classe] = {
            'precision': precision,
            'recall': recall,
            'f1-score': f1,
            'accuracy': accuracy
        }
    return metrics_per_class

def train_model():
    # Extract the training and test samples from the EMNIST dataset
    X_train, y_train = extract_training_samples('balanced')
    X_test, y_test = extract_test_samples('balanced')

    # Filter the training and test datasets
    (X_train, y_train_library) = filterDataset(X_train, y_train)
    (X_test, y_test) = filterDataset(X_test, y_test)

    # Remodulate the labels of the training and test datasets
    y_train_library = remodulate(y_train_library)
    y_test = remodulate(y_test)

    # Normalize the training and test datasets
    X_train = X_train.astype("float32") / 255
    X_test = X_test.astype("float32") / 255

    # Print the shape of the training data
    #print(X_train.shape)

    # Reshape the training data from 3D to 2D. The new shape is (number of samples, image width * image height)
    X_train = X_train.reshape((-1, 28, 28, 1))

    # Reshape the test data from 3D to 2D. The new shape is (number of samples, image width * image height)
    X_test = X_test.reshape((-1, 28, 28, 1))

    # Print the new shape of the training data
    #print(X_train.shape)

    # Print the shape of the test data
    #print(X_test.shape)

    seq_lett_model = create_model()

    # Set the batch size. This is the number of samples that will be passed through the network at once.
    batch_size = 32

    # Set the number of epochs. An epoch is one complete pass through the entire training dataset.
    epochs = 25

    # Compile the model. 
    # We use the sparse_categorical_crossentropy loss function, which is suitable for multi-class classification problems.
    # The optimizer is set to 'adam', which is a popular choice due to its efficiency and good performance on a wide range of problems.
    # We also specify that we want to evaluate the model's accuracy during training.
    seq_lett_model.compile(loss="sparse_categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

    # Fit the model to the training data. 
    # We also specify a validation split of 0.1, meaning that 10% of the training data will be used as validation data.
    # The model's performance is evaluated on this validation data at the end of each epoch.
    early_stopping = keras.callbacks.EarlyStopping(monitor='val_loss', patience=2, restore_best_weights=True)
    training_operation = seq_lett_model.fit(
        X_train, y_train_library, 
        batch_size=batch_size, 
        epochs=epochs, 
        validation_split=0.1,
        callbacks=[early_stopping])

    # Save the trained model to a file so that it can be loaded later for making predictions or continuing training.
    seq_lett_model.save('seq_lett_model.keras')
    return training_operation, X_test, y_test, seq_lett_model

def capture_image_from_webcam():
    """
    Captures an image from the webcam and returns the captured frame.

    Returns:
        numpy.ndarray: The captured frame from the webcam.
    """
    global save_frame

    import platform
    # Determina il dispositivo video in base al sistema operativo
    if platform.system() == 'Linux':
        video_device = 0
    else:
        video_device = 1

    cap = cv2.VideoCapture(video_device)
    while True:
        success, frame = cap.read()  # Legge un frame dalla telecamera
        if not success:
            break
        else:
            ret, buffer = cv2.imencode('.png', frame)  # Codifica il frame in formato PNG
            byte_frame = buffer.tobytes()
            if save_frame:
                timestamp = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
                image_path = f'./grids/grid_{timestamp}.png'
                cv2.imwrite(image_path, frame)
                save_frame = False
                return frame
            yield (b'--frame\r\n'
                   b'Content-Type: image/png\r\n\r\n' + byte_frame + b'\r\n')  # Restituisce il frame come un'immagine PNG

def save_image(image, filename):
    cv2.imwrite(filename, image)

def preprocess_image_for_detection(image):
    """
    Preprocesses an image for object detection by performing the following steps:
    1. Converts the image to grayscale.
    2. Applies Gaussian blur to reduce noise.
    3. Applies adaptive thresholding to create a binary image.
    4. Performs dilation and erosion to emphasize grid lines.
    
    Args:
        image (numpy.ndarray): The input image to be preprocessed.
        
    Returns:
        numpy.ndarray: The preprocessed image.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (7,7), 0)
    thresh = cv2.adaptiveThreshold(blur, 255, 1, 1, 15, 2)
    
    # Use dilation and erosion to emphasize the grid lines
    kernel = np.ones((3,3),np.uint8)
    dilation = cv2.dilate(thresh,kernel,iterations = 1)
    erosion = cv2.erode(dilation,kernel,iterations = 1)
    
    return erosion

def extract_ROIs(contours, original, coefficient):
    """
    Extracts regions of interest (ROIs) from the given contours based on the provided coefficient.

    Parameters:
    - contours: A list of contours.
    - original: The original image.
    - coefficient: A coefficient used to calculate the minimum contour area.

    Returns:
    - ROIs: A list of extracted regions of interest.
    - n: The count of ROIs with the same y-coordinate.

    """
    MIN_CONTOUR_AREA = coefficient * original.shape[0] * original.shape[1]
    ROIs = []
    yt = None  # y-coordinate of the previous ROI
    n = 1  # count of ROIs with the same y-coordinate
    tolerance = 10  # tolerance for the y-coordinate

    # Sort contours by their y-coordinate, then by their x-coordinate
    contours = list(contours)
    contours.sort(key=lambda c: (cv2.boundingRect(c)[1], cv2.boundingRect(c)[0]))

    for i in contours:
        area = cv2.contourArea(i)
        if area > MIN_CONTOUR_AREA:
            x, y, w, h = cv2.boundingRect(i)
            if yt is not None and abs(y - yt) <= tolerance:
                n += 1
            else:
                yt = y
                n = 1
            ROI = original[y:y+h, x:x+w]
            ROIs.append(ROI)
    return ROIs, n

def preprocess_image(image_path):
    """
    Preprocesses an image for further analysis.

    Args:
        image_path (str): The path to the image file.

    Returns:
        numpy.ndarray: The preprocessed image as a numpy array.

    Raises:
        None

    """
    im = cv2.imread(image_path)
    if im is None:
        print(f"Error loading image: {image_path}")
        return None

    im = cv2.bitwise_not(im)
    im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    _, im = cv2.threshold(im, 127, 255, cv2.THRESH_BINARY)
    im = cv2.resize(im, dsize=(28, 28), interpolation=cv2.INTER_LINEAR)
    im = im.astype("float32") / 255
    im = np.reshape(im, (28, 28, 1))
    im = tf.expand_dims(im, axis=0)
    return im

def prediction(ROIs, n, seq_lett_model):
    # Preprocess the ROIs and make predictions
    l = []
    for i in range(1, len(ROIs)):
        im = preprocess_image(f"./manipulated_grids/ROI_{i}.png")
        if im is not None:
            prediction = seq_lett_model.predict(im)
            max = np.where(prediction == np.amax(prediction))
            l.append(int(max[1][0]))

    # Check if T is present in the predictions
    if 0 not in l:
        print("T not found in the grid")
        return None  # Return none as an error

    # Create the grid from the predictions
    nrow = len(l) // n if n < len(l) else n // len(l)
    nrow = int(nrow)

    mat = np.array(list(reversed(l)))
    if nrow == 1:
        try:
            grid = mat.reshape(nrow, n-1)  
        except:
            print("Reshape error")
            return None
    else:
        try:
            grid = mat.reshape(nrow, n)  
        except:
            print("Reshape error")
            return None

    # print the grid
    print(grid)

    return grid

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import cv2
import datetime
import os
from emnist import list_datasets
from tensorflow import keras    
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import classification_report
from utils import *
from search import *
#from aima import Problem, Node #  Import aimacode from the aima-python library
from enum import Enum


# Import the necessary functions from the emnist library
from emnist import extract_training_samples
from emnist import extract_test_samples

'''
Uniform Coloring è un dominio in cui si hanno a disposizione alcune celle da colorare, e vari colori
a disposizione. Per semplicità immaginiamo una griglia rettangolare in cui è possibile spostare una
testina colorante fra le celle attigue secondo le 4 direzioni cardinali (N,S,E,W), senza uscire dalla
griglia. Tutte le celle hanno un colore di partenza (B=blu, Y=yellow, G=green) ad eccezione di
quella in cui si trova la testina indicata con T. La testina può colorare la cella in cui si trova con uno
qualsiasi dei colori disponibili a differenti costi (cost(B)=1, cost(Y)=2, cost(G)=3), mentre gli
spostamenti hanno tutti costo uniforme pari a 1. L’obiettivo è colorare tutte le celle dello stesso
colore (non importa quale) e riportare la testina nella sua posizione di partenza.
La codifica di tutto il dominio (topologia della griglia, definizione delle azioni etc.) è parte
dell’esercizio. Partendo dalla posizione iniziale della testina e combinando azioni di spostamento e
colorazione, si chiede di trovare la sequenza di azioni dell’agente per raggiungere l’obiettivo.
La posizione iniziale della testina, la struttura della griglia e la colorazione iniziale delle celle sono
passati al sistema tramite un’immagine.
'''

MOV_COST = 1

class Colors():
    BLUE = 1
    YELLOW = 2
    GREEN = 3


class Directions():
    UP = (-1, 0)
    DOWN = (1, 0)
    LEFT = (0, -1)
    RIGHT = (0, 1)

class State():
    def __init__(self, grid, i, j):
        self.grid = grid
        self.i = i # row
        self.j = j # column
        self.id = str(i)+str(j)
        for row in grid:
            for tile in row:
                self.id = self.id+str(tile)
    
    def __lt__(self, state):
        return False

class UniformColoring(Problem):
    """The abstract class for a formal problem.  You should subclass this and
    implement the method successor, and possibly __init__, goal_test, and
    path_cost. Then you will create instances of your subclass and solve them
    with the various search functions."""

    def __init__(self, initial, goal=None):
        """The constructor specifies the initial state, and possibly a goal
        state, if there is a unique goal.  Your subclass's constructor can add
        other arguments."""
        super().__init__(initial, goal)
    
    def actions(self, state):
        """Return the actions that can be executed in the given
        state. The result would typically be a list, but if there are many
        actions, consider yielding them one at a time in an iterator, rather
        than building them all at once."""
        actions = []
        if state.i > 0:
            actions.append(Directions.UP)
        if state.i < len(state.grid)-1:
            actions.append(Directions.DOWN)
        if state.j > 0:
            actions.append(Directions.LEFT)
        if state.j < len(state.grid[0])-1:
            actions.append(Directions.RIGHT)
        return actions
        '''actions=[]
        if (state.i != self.initial.i) or (state.j != self.initial.j):
            for color in Colors:  # action color tile
                if (color.value != state.grid[state.i][state.j]):
                    actions.append(color)
        for direction in Directions:  # action move
            coords=(state.i+direction.value[0],state.j+direction.value[1])
            if coords[0] in range(grid.shape[0]) and coords[1] in range(grid.shape[1]):
                actions.append(direction)
        return actions'''
    
    def result(self, state, action):
        """Return the state that results from executing the given
        action in the given state. The action must be one of
        self.actions(state)."""
        if action in Directions:
            return State(state.grid,state.i+action.value[0],state.j+action.value[1])
        else:
            grid=np.copy(state.grid)
            grid[state.i][state.j]=action.value
            return State(grid,state.i,state.j)

    def goal_test(self, state):
        """Return True if the state is a goal. The default method compares the
        state to self.goal, as specified in the constructor. Implement this
        method if checking against a single self.goal is not enough."""
        test_color = None
        if ((state.i, state.j) == (self.initial.i, self.initial.j)):  # if it's back at the start position
            for color in Colors:
                if color.value in state.grid:
                    if test_color == None:
                        test_color = color.value
                    if color.value != test_color:
                        return False
        else:
            return False
        return True

    def path_cost(self, c, state1, action, state2):
        """Return the cost of a solution path that arrives at state2 from
        state1 via action, assuming cost c to get up to state1. If the problem
        is such that the path doesn't matter, this function will only look at
        state2.  If the path does matter, it will consider c and maybe state1
        and action. The default method costs 1 for every step in the path."""
        if action in Colors:
            return c + action.value
        return c + MOV_COST

    def value(self):
        """For optimization problems, each state has a value.  Hill-climbing
        and related algorithms try to maximize this value."""
        raise NotImplementedError
    
def best_first_graph_search(problem, f, display=False):
    """Search the nodes with the lowest f scores first.
    You specify the function f(node) that you want to minimize; for example,
    if f is a heuristic estimate to the goal, then we have greedy best
    first search; if f is node.depth then we have breadth-first search.
    There is a subtlety: the line "f = memoize(f, 'f')" means that the f
    values will be cached on the nodes as they are computed. So after doing
    a best first search you can examine the f values of the path returned."""
    f = memoize(f, 'f')
    node = Node(problem.INITIAL)
    #print("#COORDS0:", node.state.i, node.state.j, node.state.grid)
    frontier = PriorityQueue('min', f)
    frontier.append(node)
    lookup_frontier=set()
    lookup_frontier.add(id(node))
    explored = set()
    while frontier:
        #print("frontiera:", len(frontier))
        node = frontier.pop()
        #print("#NODE:", node.state.i, node.state.j, node.state.grid)
        lookup_frontier.remove(id(node))
        if problem.goal_test(node.state):
            if display:
                print(len(explored), "paths have been expanded and", len(frontier), "paths remain in the frontier")
            return node
        explored.add(node.state.id)
        for child in node.expand(problem):
            #print("#CHILD:", node.state.i, node.state.j, node.state.grid)
            if child.state.id not in explored and id(child) not in lookup_frontier:
                lookup_frontier.add(id(child))
                frontier.append(child)
            elif id(child) in lookup_frontier:
                if f(child) < frontier[child]:
                    del frontier[child]
                    lookup_frontier.add(id(child))
                    frontier.append(child)
    return None


# Define the labels for the EMNIST dataset
LABELS = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 
          'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z',
          'a', 'b', 'd', 'e', 'f', 'g', 'h', 'n', 'q', 'r', 't']

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
    seq_lett_model = keras.Sequential([
        keras.Input(shape=(28, 28, 1)), # Input shape: 28x28 pixels, 1 color channel
        keras.layers.Conv2D(28, (3, 3), activation='relu'),
        keras.layers.MaxPooling2D((2, 2)),
        keras.layers.Conv2D(128, (3, 3), activation='relu'),
        keras.layers.MaxPooling2D((2, 2)),
        keras.layers.Conv2D(128, (3, 3), activation='relu'),
        keras.layers.Flatten(),
        keras.layers.Dense(128, activation='relu'),
        keras.layers.Dropout(0.5),
        keras.layers.Dense(4, activation='softmax')  # Output layer for 4 letters
    ])
    seq_lett_model.summary()
    return seq_lett_model

def model_statistics(training_operation, X_test, y_test, seq_lett_model):
    # Plot training & validation accuracy values
    plt.plot(training_operation.history['accuracy'])
    plt.plot(training_operation.history['val_accuracy'])
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Val'], loc='upper left')
    plt.show()

    # Plot training & validation loss values
    plt.plot(training_operation.history['loss'])
    plt.plot(training_operation.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Val'], loc='upper left')
    plt.show()

    # Evaluate the model's performance on the test data. 
    # The evaluate function returns the loss value and metrics values for the model in test mode.
    # We set verbose=0 to avoid logging the detailed output during the evaluation.
    loss, accuracy =  seq_lett_model.evaluate(X_test, y_test, verbose=0)

    # Print the loss value that the model achieved on the test data.
    # The loss value represents how well the model can estimate the target variables. Lower values are better.
    print("Test loss:", loss)

    # Print the accuracy that the model achieved on the test data.
    # The accuracy is the proportion of correct predictions that the model made. Higher values are better.
    print("Test accuracy:", accuracy)

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

    # Display the plot.
    plt.show()

    from sklearn.metrics import classification_report
    print(classification_report(y_test, y_pred))

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
    print(X_train.shape)

    # Reshape the training data from 3D to 2D. The new shape is (number of samples, image width * image height)
    X_train = X_train.reshape((-1, 28, 28, 1))

    # Reshape the test data from 3D to 2D. The new shape is (number of samples, image width * image height)
    X_test = X_test.reshape((-1, 28, 28, 1))

    # Print the new shape of the training data
    print(X_train.shape)

    # Print the shape of the test data
    print(X_test.shape)

    seq_lett_model = create_model()

    # Set the batch size. This is the number of samples that will be passed through the network at once.
    batch_size = 32

    # Set the number of epochs. An epoch is one complete pass through the entire training dataset.
    epochs = 15

    # Compile the model. 
    # We use the sparse_categorical_crossentropy loss function, which is suitable for multi-class classification problems.
    # The optimizer is set to 'adam', which is a popular choice due to its efficiency and good performance on a wide range of problems.
    # We also specify that we want to evaluate the model's accuracy during training.
    seq_lett_model.compile(loss="sparse_categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

    # Fit the model to the training data. 
    # We also specify a validation split of 0.1, meaning that 10% of the training data will be used as validation data.
    # The model's performance is evaluated on this validation data at the end of each epoch.
    early_stopping = keras.callbacks.EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
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
    import platform
    # Determina il dispositivo video in base al sistema operativo
    if platform.system() == 'Linux':
        video_device = 0
    else:
        video_device = 1

    # Crea l'oggetto VideoCapture utilizzando il dispositivo video appropriato
    cap = cv2.VideoCapture(video_device)

    while True:
        ret, frame = cap.read()
        if ret:
            cv2.imshow('Input', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            break
    cap.release()
    cv2.destroyAllWindows()
    return frame

def save_image(image, filename):
    cv2.imwrite(filename, image)

def preprocess_image_for_detection(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (7,7), 0)
    thresh = cv2.adaptiveThreshold(blur, 255, 1, 1, 15, 2)
    
    # Use dilation and erosion to emphasize the grid lines
    kernel = np.ones((3,3),np.uint8)
    dilation = cv2.dilate(thresh,kernel,iterations = 1)
    erosion = cv2.erode(dilation,kernel,iterations = 1)
    
    return erosion

def find_largest_contour(contours):
    max_area = 0
    for i in contours:
        area = cv2.contourArea(i)
        if area > max_area:
            max_area = area
            largest_contour = i
    return largest_contour

def extract_ROIs(contours, original, coefficient):
    MIN_CONTOUR_AREA = coefficient * original.shape[0] * original.shape[1]
    ROIs = []
    yt = None  # y-coordinate of the previous ROI
    n = 1  # count of ROIs with the same y-coordinate
    tolerance = 10  # tolerance for the y-coordinate
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

def main():
    '''list_datasets()
    #ask user terminale input for training model
    train_flag  = input("Do you want to train the model? (y/n): ")  
    if train_flag == 'y':
        training_operation, X_test, y_test, seq_lett_model = train_model()
        model_statistics(training_operation, X_test, y_test, seq_lett_model)
    
    seq_lett_model = keras.models.load_model('seq_lett_model.keras')

    os.system("find './manipulated_grids/' -name 'ROI_*' -exec rm {} \;")

    # Ask user to take a picture of the grid or if they want to use a default image from file explorer
    if input("Do you want to take a picture of the grid? If you press n you have to pick an image from your filesystem (y/n): ") == 'y':
        frame = capture_image_from_webcam()
        timestamp = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        image_path = f'./grids/grid_{timestamp}.png'

        save_image(frame, image_path)
    else:
        path = input("Enter the name of the image: ")
        image_path = f'./grids/{path}.png'

    image = cv2.imread(image_path)
    if image is None:
        print(f"Error loading image: image_path")
        return
    processed_image = preprocess_image_for_detection(image)
    canny = cv2.Canny(np.asarray(processed_image), 0, 200)
    contours, _ = cv2.findContours(processed_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    #largest_contour = find_largest_contour(contours)
    ROIs, n = extract_ROIs(contours, image.copy(), 0.02)

    for i, ROI in enumerate(ROIs):
        save_image(ROI, f'./manipulated_grids/ROI_{i}.png')

    l = []
    for i in range(1, len(ROIs)):
        im = preprocess_image(f"./manipulated_grids/ROI_{i}.png")
        if im is not None:
            prediction = seq_lett_model.predict(im)
            max = np.where(prediction == np.amax(prediction))
            l.append(int(max[1][0]))

    nrow = len(l) // n if n < len(l) else n // len(l)
    nrow = int(nrow)

    mat = np.array(list(reversed(l)))
    grid = mat.reshape(nrow, n)

    label_mapping = {0: 'T', 1: 'B', 2: 'Y', 3: 'G'}
    show = np.vectorize(label_mapping.get)(grid)
    print(show)
    #os.system("find './manipulated_grids/' -name 'ROI_*' -exec rm {} \;")'''

    grid=np.array([[0, 2, 2, 3], [2, 2, 3, 2], [3, 1, 2, 2], [1, 3, 3, 1]])
    """grid=np.array([[2, 1, 2, 3, 1],
        [2, 3, 1, 3, 1],
        [0, 1, 2, 3, 2],
        [2, 1, 3, 3, 2]])"""
    for i in range(grid.shape[0]):
        for j in range(grid.shape[1]):
            if grid[i][j]==0:
                initial_state=State(grid,i,j)
                break
    
    problem=UniformColoring(initial_state)

    end = best_first_graph_search(problem, lambda n: n.path_cost, display=True)
    print("Final state: \n",end.state.grid)
    print("Solution cost:",end.path_cost)
    print(end.solution())

if __name__ == "__main__":
    main()
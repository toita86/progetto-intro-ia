import uniformcoloring as uc
import lettermodel as lm
import os
import sys
import time
import datetime
import cv2
import numpy as np
import signal
import warnings
from flask import Flask, Response, render_template

'''
Uniform Coloring è un dominio in cui si hanno a disposizione alcune celle da colorare, e vari colori
a disposizione. 
Per semplicità immaginiamo una griglia rettangolare in cui è possibile spostare una
testina colorante fra le celle attigue secondo le 4 direzioni cardinali (N,S,E,W), senza uscire dalla
griglia. 
Tutte le celle hanno un colore di partenza (B=blu, Y=yellow, G=green) ad eccezione di
quella in cui si trova la testina indicata con T. 
La testina può colorare la cella in cui si trova con uno
qualsiasi dei colori disponibili a differenti costi (cost(B)=1, cost(Y)=2, cost(G)=3), mentre gli
spostamenti hanno tutti costo uniforme pari a 1. 

L'obiettivo è colorare tutte le celle dello stesso
colore (non importa quale) e riportare la testina nella sua posizione di partenza.
La codifica di tutto il dominio (topologia della griglia, definizione delle azioni etc.) è parte
dell'esercizio. 
Partendo dalla posizione iniziale della testina e combinando azioni di spostamento e
colorazione, si chiede di trovare la sequenza di azioni dell'agente per raggiungere l'obiettivo.
La posizione iniziale della testina, la struttura della griglia e la colorazione iniziale delle celle sono
passati al sistema tramite un'immagine.
'''

app = Flask(__name__)

class TimeoutError(Exception):
    pass

def handler(signum, frame, time):
    raise TimeoutError(f"Time expired!! Exceded {time} seconds")

def get_last_image_path(directory):
    # List all files in the directory
    files = [os.path.join(directory, f) for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))]
    
    # Sort files by modification time
    files.sort(key=lambda x: os.path.getmtime(x))
    
    # Return the last file
    return files[-1] if files else None

def grid_translation(grid):
    label_mapping = {0: 'T', 1: 'B', 2: 'Y', 3: 'G'}
    res = np.vectorize(label_mapping.get)(grid)
    return res

def main():
    #ask user terminale input for training model
    train_flag  = input("Do you want to train the model? (y/n): ")  
    if train_flag == 'y':
        training_operation, X_test, y_test, seq_lett_model = lm.train_model()
        lm.model_statistics(training_operation, X_test, y_test, seq_lett_model)
    
    # Load the trained model from the file
    seq_lett_model = lm.keras.models.load_model('seq_lett_model.keras')

    while True:
        try:
            warnings.filterwarnings("ignore")
            # Delete all the files in the manipulated_grids folder
            lm.os.system("find './manipulated_grids/' -name 'ROI_*' -exec rm {} \;")
            # Ask user to take a picture of the grid or if they want to use a default image from file explorer
            if input("Do you want to take a picture of the grid? If you press n you have to pick an image from your filesystem (y/n): ") == 'y':
                frame = lm.capture_image_from_webcam()
                timestamp = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
                image_path = f'./grids/grid_{timestamp}.png'

                lm.save_image(frame, image_path)
            else:
                path = input("Enter the name of the image: ")
                image_path = f'./grids/{path}.png'

            # Load the image
            image = cv2.imread(image_path)
            if image is None:
                raise Exception(f"Error loading image: {image_path}")

            processed_image = lm.preprocess_image_for_detection(image)

            contours, _ = cv2.findContours(processed_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            #contours = sorted(contours, key=cv2.contourArea, reverse=True)
            ROIs, n = lm.extract_ROIs(contours, image.copy(), 0.02)

            # Save the ROIs to the manipulated_grids folder
            for i, ROI in enumerate(ROIs):
                lm.save_image(ROI, f'./manipulated_grids/ROI_{i}.png')

            grid = lm.prediction(ROIs, n, seq_lett_model)

            break
        except Exception as e:
            print(f"An error occurred: {e}")
            retry = input("Do you want to retry? (y/n): ")
            if retry.lower() != 'y':
                sys.exit(1)                

    """
    grid=np.array([
        [0, 3, 3], 
        [3, 3, 2],
        [3, 3, 3],])
    
    grid=np.array([[2, 1, 2, 3, 1],
        [2, 3, 1, 3, 1],
        [0, 1, 2, 3, 2],
        [2, 1, 3, 3, 2]])
    """
    
    # Define the initial state
    problem=uc.UniformColoring(uc.initialize_state(grid),uc.Heuristic.heuristic_color_use_most_present)

    t = 10
    signal.signal(signal.SIGALRM, lambda signum, frame: handler(signum, frame, t))
    signal.alarm(t)

    try:
        # function call for the UCS 
        start_time = time.time()
        print("Compute uniform_cost_search...")
        ucs = uc.best_first_graph_search(problem, lambda node: node.path_cost)
        print("\n")
        print("--- %s seconds ---" % (time.time() - start_time))
        print("Final state: \n",ucs.state.grid)
        print("\n")
        print("Solution cost:",ucs.path_cost)
        print(ucs.solution())
    except TimeoutError as e:
        # Exception handler if time is expired
        print(e)
    finally:
        # Timer reset if the function returns before time 
        signal.alarm(0)

    signal.signal(signal.SIGALRM, lambda signum, frame: handler(signum, frame, t))
    signal.alarm(t)
        
    try:
        # function call for the ids 
        start_time = time.time()
        ids = uc.iterative_deepening_search(problem)
        print("Final state: \n",ids.state.grid)
        print("\n")
        print("Solution cost:",ids.path_cost)
        print("\n")
        print(ids.solution())
    except TimeoutError as e:
        # Exception handler if time is expired
        print(e)
    finally:
        # Timer reset if the function returns before time 
        signal.alarm(0)

    astar = uc.astar_search(problem, display=True)
    print("Final state: \n",astar.state.grid)
    print("\n")
    print("Solution cost:",astar.path_cost)
    print("\n")
    print(astar.solution())

    greedy = uc.greedy_search(problem, display=True)
    print("Final state: \n",greedy.state.grid)
    print("\n")
    print("Solution cost:",greedy.path_cost)
    print("\n")
    print(greedy.solution())

    # Delete all the files in the manipulated_grids folder after computing
    #os.system("find './manipulated_grids/' -name 'ROI_*' -exec rm {} \;")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(lm.capture_image_from_webcam(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/save_image', methods=['POST'])
def save_image():
    lm.save_frame = True
    return 'OK', 200

# an @app route to choose the image from the grids folder and process it
@app.route('/process_image' , methods=['POST'])
def process_image():
    seq_lett_model = lm.keras.models.load_model('seq_lett_model.keras')

    ucs = ids = astar = greedy = None

    #image_path = get_last_image_path('./grids')
    image_path = './grids/3x3.png'
    image = cv2.imread(image_path)
    if image is None:
        return 'Error loading image', 404

    processed_image = lm.preprocess_image_for_detection(image)

    contours, _ = cv2.findContours(processed_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    #contours = sorted(contours, key=cv2.contourArea, reverse=True)
    ROIs, n = lm.extract_ROIs(contours, image.copy(), 0.02)

    # Save the ROIs to the manipulated_grids folder
    for i, ROI in enumerate(ROIs):
        lm.save_image(ROI, f'./manipulated_grids/ROI_{i}.png')

    grid = lm.prediction(ROIs, n, seq_lett_model)

    # Define the initial state
    problem=uc.UniformColoring(uc.initialize_state(grid),uc.Heuristic.heuristic_color_use_most_present)

    print("Compute uniform_cost_search...")
    ucs = uc.best_first_graph_search(problem, lambda node: node.path_cost)
        
    ids = uc.iterative_deepening_search(problem)

    astar = uc.astar_search(problem, display=True)

    greedy = uc.greedy_search(problem, display=True)

    grid_show=grid_translation(grid)
    grid_show = grid_show.tolist()

    # Delete all the files in the manipulated_grids folder after computing
    os.system("find './manipulated_grids/' -name 'ROI_*' -exec rm {} \;")
    return render_template('results.html', 
                           grid=grid_show, 
                           ucs=ucs, 
                           ids=ids, 
                           astar=astar, 
                           greedy=greedy)

if __name__ == "__main__":
    #main()
    app.run(debug=True)
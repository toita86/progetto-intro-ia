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

@app.route('/train_model', methods=['POST'])
def train_model():
    training_operation, X_test, y_test, seq_lett_model = lm.train_model()
    os.system("find './static/plots/' -name 'ROI_*' -exec rm {} \;")
    metrics_per_class = lm.model_statistics(training_operation, X_test, y_test, seq_lett_model)

    # Evaluate the model's performance on the test data. 
    # The evaluate function returns the loss value and metrics values for the model in test mode.
    # We set verbose=0 to avoid logging the detailed output during the evaluation.
    # The loss value represents how well the model can estimate the target variables. Lower values are better.
    # The accuracy value represents the percentage of correct predictions made by the model.
    loss, accuracy =  seq_lett_model.evaluate(X_test, y_test, verbose=0)

    return render_template('train.html',
                           metrics_per_class=metrics_per_class,
                           loss=loss,
                           accuracy=accuracy)


# an @app route to choose the image from the grids folder and process it
@app.route('/process_image' , methods=['POST'])
def process_image():
    seq_lett_model = lm.keras.models.load_model('seq_lett_model.keras')

    ucs = ids = astar = greedy = None

    #image_path = get_last_image_path('./grids/')
    image_path = './grids/3x3.png' # for testing purposes
    print(f"Processing image: {image_path}")
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

    if grid is None:
        print("Cannot execute search algorithms")
        return render_template('Error.html')
       
    # Define the initial state
    problem=uc.UniformColoring(uc.initialize_state(grid),uc.Heuristic.heuristic_color_use_most_present)

    # Measure the execution time of UCS
    ucs_succ = "UCS successfully found a solution"
    start_time = time.time()
    ucs, ucs_succ = uc.best_first_graph_search(problem, lambda node: node.path_cost)
    ucs_time = time.time() - start_time
    if ucs_succ == -1:
        ucs_succ = "UCS did not found a solution"

    # Measure the execution time of IDS
    ids_succ = "IDS successfully found a solution"
    start_time = time.time()
    ids, f_succ = uc.iterative_deepening_search(problem)
    ids_time = time.time() - start_time
    if f_succ == -1:
        ids_succ = "IDS did not found a solution"

    # Measure the execution time of A*
    start_time = time.time()
    astar, gr_succ = uc.astar_search(problem)
    astar_time = time.time() - start_time

    # Measure the execution time of Greedy search
    start_time = time.time()
    greedy, gr_succ = uc.greedy_search(problem)
    greedy_time = time.time() - start_time

    grid_show=grid_translation(grid)
    grid_show = grid_show.tolist()
    grid_ucs = grid_translation(ucs.state.grid)
    grid_ucs = grid_ucs.tolist()
    grid_ids = grid_translation(ids.state.grid)
    grid_ids = grid_ids.tolist()
    grid_astar = grid_translation(astar.state.grid)
    grid_astar = grid_astar.tolist()
    grid_greedy = grid_translation(greedy.state.grid)
    grid_greedy = grid_greedy.tolist()

    # Delete all the files in the manipulated_grids folder after computing
    os.system("find './manipulated_grids/' -name 'ROI_*' -exec rm {} \;")
    return render_template('results.html', 
                           grid=grid_show, 
                           ucs=ucs, 
                           ids=ids, 
                           astar=astar, 
                           greedy=greedy,
                           ucs_time=ucs_time,
                           ids_time=ids_time,
                           astar_time=astar_time,
                           greedy_time=greedy_time,
                           ucs_succ=ucs_succ,
                           ids_succ=ids_succ,
                           grid_ucs=grid_ucs,
                           grid_ids=grid_ids,
                           grid_astar=grid_astar,
                           grid_greedy=grid_greedy)

if __name__ == "__main__":
    #main()
    app.run(host='0.0.0.0', port=5050,debug=True)
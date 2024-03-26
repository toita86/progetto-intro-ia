##### Studenti: 
###### Brahas Eduard
###### Chionne Daniel 

##### Docente
###### Poggioni Valentina

Anno accademico 2023/2024
Inserire indice

## Prefazione
Il progetto consiste nella realizzazione di una applicazione di Intelligenza Artificiale completa degli aspetti di gestione di: sensori per l’acquisizione dei dati dall’esterno relativi a stati e obiettivi, ragionamento/ricerca della soluzione per i goal acquisiti, esecutori per la realizzazione delle azioni che conducono alla soluzione.
## Progetto Uniform coloring
Uniform Coloring è un dominio in cui si hanno a disposizione alcune celle da colorare, e vari colori a disposizione.

Per semplicità immaginiamo una griglia rettangolare in cui è possibile spostare una testina colorante fra le celle attigue secondo le 4 direzioni cardinali (N,S,E,W), senza uscire dalla griglia.

Tutte le celle hanno un colore di partenza (B=blu, Y=yellow, G=green) ad eccezione di quella in cui si trova la testina indicata con T. La testina può colorare la cella in cui si trova con uno qualsiasi dei colori disponibili a differenti costi (cost(B)=1, cost(Y)=2, cost(G)=3), mentre gli spostamenti hanno tutti costo uniforme pari a 1.

>L’obiettivo è colorare tutte le celle dello stesso colore (non importa quale) e riportare la testina nella sua posizione di partenza.

La codifica di tutto il dominio (topologia della griglia, definizione delle azioni etc.) è parte dell’esercizio. Partendo dalla posizione iniziale della testina e combinando azioni di spostamento e colorazione, si chiede di trovare la sequenza di azioni dell’agente per raggiungere l’obiettivo.

La posizione iniziale della testina, la struttura della griglia e la colorazione iniziale delle celle sono passati al sistema tramite un’immagine.

### Inizializzazione delle librerie e moduli
```python
#prog-intro.py
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

#lettermodel.py
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
from emnist import extract_training_samples
from emnist import extract_test_samplesmanca classfic
#manca classification report se va implementato

#uniformcoloring.py
from utils import *
from search import *
from enum import Enum
import time
```

Il file python del progetto è suddiviso in 3 files:
- prog-intro-ai.py: 
## 1. Descrizione formale del dominio e vincoli su cui l'agente può operare
Descrizione formale del dominio e dei vincoli che le azioni eseguibili dall’agente (la testina) nella griglia devono rispettare (esempio vincoli: v1=”l’agente può compiere un solo passo alla volta”, v2=”l’agente si può muovere solo fra celle adiacenti”, etc.). La descrizione sarà un testo che descrive le regole da rispettare e le assunzioni desunte dall’analisi del dominio.

### Descrizione del dominio e vincoli

**Elementi del dominio:**
- _Celle_: Presenti in una griglia rettangolare nella quale è possibile spostare una testina colorante. Ogni cella ha un colore di partenza ed è possibile ricolorarle con i tre colori disponibili (yellow, blue, green).
- _Testina colorante_: È l'agente che, nella griglia fornita in input come immagine, può spostarsi tra le celle e cambiarne il colore. Una delle celle rappresenta la posizione iniziale della testina prima di muoversi e sulla quale dovrà ritornare dopo aver svolto le azioni richieste.

**Relazioni:**
- Ad ogni cella dell'immagine è associata un'etichetta rappresentante uno dei colori disponibili (Y, B, G).
- La testina inizialmente dovrà essere posizionata sempre sulla cella con etichetta T.

**Regole:**
- La testina può spostarsi nelle sole direzioni nord, sud, est, ovest.
- La testina può cambiare colore nella cella in cui è posizionata. Colorare le celle ha un costo che varia in base al colore (cost(B) = 1, cost(Y) = 2, cost(G) = 3).
- Il passaggio da una cella all'altra ha sempre costo 1.
- **GOAL**: Colorare tutte le celle dello stesso colore. La testina dovrà trovare un modo per farlo nella maniera più efficiente possibile, sia in termini di colori che di numero di passi effettuati per muoversi tra le celle.

**Vincoli:**
- v0="La griglia può essere rettangolare e quadrata".
- v1=”L’agente può compiere un solo passo alla volta”.
- v2=”L’agente si può muovere solo fra celle adiacenti”.
- v3="Nella griglia non esistono celle vuote, tutte devono essere colorate in partenza".
- v4="Le celle devono essere tutte raggiungibili dalla posizione iniziale della testina".
- v5="Il costo delle azioni di movimento è uniforme (1), mentre il costo della colorazione dipende dal colore scelto".
- v6="L'agente non può colorare la posizione di partenza (quindi bisogna trovare un modo per evitarlo)".
- v7="Dopo che l'agente ha colorato tutte le celle, deve ritornare alla posizione di partenza".

**Esempi di problemi, con possibili soluzioni e costi**:
![[esempio.png|500]]

## 2 Implementazione delle classi per la ricerca nello spazio degli stati di Smart Vacuum.
Utilizzando le classi di AIMA-python si implementi quindi un dominio UniformColoring come classe derivata da Problem, scegliendo e definendo una rappresentazione per gli stati e ridefinendo opportuni metodi actions e result e tutti gli eventuali metodi aggiuntivi , es. goal_test, che si rendessero necessari.

Si descriva e si implementi almeno una euristica definendone le caratteristiche di consistenza e ammissibilità rispetto al dominio dato. L’euristica definita mantiene le stesse proprietà nel caso in cui le azioni di spostamento costassero 0 o le colorazioni avessero tutte lo stesso costo?

```python

```

## 3 Acquisizione e classificazione degli input, stato iniziale e goal.

Si realizzi un programma che, passata in input un’immagine contenente la configurazione della griglia e la posizione iniziale dell’agente:

-  interpreti l’immagine individuando la configurazione della griglia e, attraverso un metodo di classificazione, la posizione iniziale dell’agente e la colorazione iniziale delle celle. Non ci sono vincoli sul metodo/modello utilizzato per la classificazione. Si consiglia di utilizzare il dataset eMNIST/MNIST per la classificazione di lettere e cifre scritte a mano, visto a lezione e facilmente reperibile su Web. Assumiamo che le lettere siano solo maiuscole;

- b) traduca i dati risultanti dall’analisi delle immagini negli stati stato_iniziale e stato_goal del problema, secondo la rappresentazione definita per il punto 2;

- c) invochi il solutore (vedi punto 2), tramite una tecnica di ricerca informata e almeno una non informata, della classe UniformColoring e produca, se esiste, la soluzione del problema, ovvero la sequenza azioni da eseguire per raggiungere lo stato goal. E’ interessante mostrare come algoritmi diversi possano portare a soluzioni diverse nel caso in cui ottimizzino rispetto al costo della soluzione oppure rispetto la sua lunghezza.
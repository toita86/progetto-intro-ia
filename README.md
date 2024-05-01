# progetto-intro-ia

- [Consegna del progetto](#consegna-del-progetto)
- [Set up del virtual environment](#set-up-del-virtual-environment)
- [Struttura dei Commit](#struttura-dei-commit)

## Consegna del Progetto

Il file con la consegna del progetto può essere trovato [qui](https://github.com/toita86/progetto-intro-ia/blob/main/ProgettoIntroAI_2122_Coloring.pdf).

### La struttura del progetto con relative spiegazioni

Si trova nella [documentazione](./Progetto%20Introduzione%20all'Intelligenza%20Artificiale.pdf). È stata scritta in obsidian, quindi per comodità è stata esportata in pdf però c'è anche il file in `.md`.

## Set up del virtual environment

creare l'ambiente virtuale usando il seguente comando:

```
pip install -r requirements.txt
```

Questo comando installerà tutte le dipendenze del progetto elencate nel file requirements.txt nel loro ambiente virtuale.

Per qualche motivo il comando `pip install emnist` non scaricherà nulla perciò dopo aver scaricato
a mano il dataset da [qui](https://drive.google.com/uc?id=1R0blrtCsGEVLjVL3eijHMxrwahRUDK26).

Mettere il file `emnist.zip` nella directory `.venv/lib/python3.10/site-packages/emnist`

Occore andare a modificare il file `__init__.py` che si trova qui:

```
.venv/lib/python3.10/site-packages/emnist/`
```

ed modificare la voce `CACHE_FILE_PATH` con la path relativa del file che avete appena spostato nella cartella.
Che se messo nella stessa cartella basta incollare questo:

```
'.venv/lib/python3.10/site-packages/emnist/emnist.zip'
```

## Struttura dei Commit

Per mantenere la storia del progetto pulita e facilmente comprensibile, è importante seguire una struttura standard per i commit.
Seguire queste linee guida:

- **Sintassi del Commit Message:**
  La forma generica è

  ```
  [ai][<commit_type>]: <description_off_the_change>
  ```

  Ogni messaggio di commit dovrebbe essere chiaro e descrittivo. Utilizzare una sintassi verbosa e specifica.
  Ad esempio:

  ```
  [ai][ADD]: Aggiunta della funzionalità X
  [ai][FIX]: Correzione del bug Y
  [ai][IMP]: Miglioramento delle prestazioni di Z
  ```

- **Atomicità dei Commit:**
  Cerca di mantenere i commit atomici, cioè con una singola funzionalità o correzione per commit.
  Questo rende più semplice la gestione delle modifiche e il roll-back se necessario.
  Per ogni singola funzionalità si crea un commit poi il push può essere fatto alla fine.
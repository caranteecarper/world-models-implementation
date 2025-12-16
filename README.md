# World Models Implementation (PyTorch)

This repository contains a PyTorch implementation of the ["World Models"](https://worldmodels.github.io/) paper (Ha & Schmidhuber, 2018). The system acts within the `CarRacing-v3` Gymnasium environment, learning a compressed representation of the world to train a controller via evolution strategies.

## Prerequisites

### System Dependencies
This project relies on Box2D (via Gymnasium), which requires **SWIG** to be installed on your system.

**macOS (Homebrew):**
```bash
brew install swig
```

**Ubuntu/Debian:**
```bash
sudo apt-get install -y swig
```

### Python Requirements
Clone the repository and install the dependencies:

```bash
pip install -r requirements.txt
```

## Configuration

### Weights & Biases (WandB)
This implementation uses [Weights & Biases](https://wandb.ai/) for logging training metrics. You must provide your API key before running the training notebooks.

**Option A: Local Environment (macOS/Linux)**
Set the `wandbApiKey` environment variable. You can do this in your terminal or add it to your shell profile (`~/.zshrc` or `~/.bashrc`):

```bash
export wandbApiKey="YOUR_WANDB_API_KEY"
```

**Option B: Google Colab**
If running in Colab, use the **Secrets** manager (key icon on the left):
1. Add a new secret named `wandbApiKey`.
2. Paste your API key as the value.
3. Toggle "Notebook access" to enable it.

## Usage Guide

The training pipeline consists of four distinct stages. Please run them in the numbered order.

### 1. Dataset Generation
* **File:** `02-Build_Dataset.ipynb`
* **Description:** Generates random rollouts of the environment to create a dataset of observations and actions.

### 2. Train VAE (Vision)
* **File:** `03-Train_VAE.ipynb`
* **Description:** Trains the Variational Autoencoder (VAE) to compress 64x64 game frames into a 32-dimensional latent vector ($z$).

### 3. Train World Model (Memory)
* **File:** `05-Train_World_Model.ipynb`
* **Description:** Trains the MDN-RNN to predict the next latent state ($z_{t+1}$) based on current state and action.

### 4. Train Controller (Agent)
* **File:** `07-Train_Controller.ipynb`
* **Description:** Uses CMA-ES (Evolution Strategies) to optimize a linear controller that solves the task using inputs from the VAE and World Model.

## Testing & Visualization

After training, use the following scripts to evaluate the model:

* **Run Agent (Rendered):** `python 08-test_final_worldmodel.py`  
    * Watch the trained agent drive in real-time.

* **Generate Replay Video:** `python 09-generate_worldmodel_test_video.py`  
    * Saves a video file of the agent's performance.  

* **Dream Mode:** `python 06-world_model_manual_test.py`  
    * Explore the "dream" environment generated purely by the World Model.  

* **VAE Analysis:** `python 04-compare_vae_to_observation.py`  
    * Visualizes how well the VAE reconstructs the game environment.  

* **Manual Play:** `python 01-manual_test.py`  
    * Drive the car yourself to test the environment mechanics.  
Evaluation of Optimizers on KMNIST Dataset

Project Overview

This project compares the performance of three deep learning optimizers—Adam, RMSprop, and AdamW—on the KMNIST dataset using a feedforward neural network (FNN). The model is trained and evaluated using TensorFlow and Keras, with hyperparameter tuning via 5-fold cross-validation.

Dataset

Kuzushiji-MNIST (KMNIST) is a dataset containing 28x28 grayscale images of Hiragana characters:

60,000 training images

10,000 test images

10 classes

The dataset is normalized and reshaped for use in a fully connected neural network.

Requirements

To run this project, install the required Python libraries:

pip install tensorflow tensorflow-datasets numpy pandas matplotlib seaborn

How to Run the Code

Load KMNIST Dataset

The script automatically downloads and preprocesses KMNIST using TensorFlow Datasets.

Preprocessing Steps

Images are normalized to [0,1] and flattened to 784 features.

Labels are converted to one-hot encoding.

Building the Neural Network

Input: 784 neurons

Hidden Layers:

Dense (128 neurons, ReLU)

Dense (64 neurons, ReLU)

Output: 10 neurons (Softmax activation)

Cross-Validation & Training

The script uses 5-fold cross-validation to evaluate each optimizer.

Optimizers compared: Adam, RMSprop, and AdamW.

Learning rates and weight decay are tuned.

Running the Script

Execute the Python script:

python MidtermProject.py

The script will:

Train and validate models using different optimizers.

Print the best hyperparameters and validation accuracy.

Train a final model with optimal parameters.

Results

Training accuracy and test loss are visualized.

A CSV file (optimizer_results.csv) logs the performance of all optimizers.

Best_optimizer_results.csv records the final model's performance.

Expected Output

The script prints validation accuracy for each optimizer.

It generates heatmaps and plots for comparison.

Final accuracy and best hyperparameters are displayed.

Files in Repository

MidtermProject.py: Main script for training and evaluation.

optimizer_results.csv: Results from hyperparameter tuning.

Best_optimizer_results.csv: Final optimizer results.

README.txt: Instructions for running the project.

Authors

Kartik Kalra - Implemented RMSprop optimizer, visualizations, and GitHub setup.

Antra Nayudu - Implemented Adam optimizer, experimental setup documentation.

Pushkar Kafley - Implemented AdamW optimizer, designed PowerPoint presentation.

Each team member contributed equally to research, discussions, and presentation.



# Deep Learning Optimizers Comparison: Adam, AdamW, and RMSprop on KMNIST Dataset

## Project Overview
This project compares three popular deep learning optimizers—Adam, AdamW, and RMSprop—on the KMNIST dataset. The goal is to evaluate the performance of these optimizers in terms of accuracy and loss for a basic feedforward neural network model. The project involves hyperparameter tuning, cross-validation, and visualization of training results.

## Table of Contents
1. [Getting Started](#getting-started)
2. [Project Setup](#project-setup)
3. [Dataset](#dataset)
4. [Model Architecture](#model-architecture)
5. [Hyperparameter Tuning](#hyperparameter-tuning)
6. [Results](#results)
7. [Visualization](#visualization)
8. [License](#license)

## Getting Started
These instructions will help you set up the project locally to run and experiment with the code.

### Prerequisites
- Python 3.6+
- TensorFlow 2.x
- TensorFlow Datasets
- Matplotlib
- Numpy
- Pandas
- scikit-learn

You can install the required libraries using pip:

```bash
pip install tensorflow tensorflow-datasets numpy pandas matplotlib scikit-learn
```

### Project Setup
1. Clone this repository:

```bash
git clone https://github.com/yourusername/deep-learning-optimizers-comparison.git
cd deep-learning-optimizers-comparison
```

2. Install the required dependencies:

```bash
pip install -r requirements.txt
```

3. Run the project:

```bash
python main.py
```

## Dataset
The dataset used for this project is **KMNIST (Kuzushiji-MNIST)**, a dataset of handwritten Japanese characters. The dataset is available via TensorFlow Datasets (`tensorflow_datasets`).

- **Train set**: 60,000 samples
- **Test set**: 10,000 samples
- **Classes**: 10 characters

## Model Architecture
The model used in this project is a simple feedforward neural network with the following layers:
- **Flatten layer**: Converts input images (28x28 pixels) into a 1D vector.
- **Dense layer**: 128 neurons with ReLU activation.
- **Output layer**: 10 neurons with Softmax activation for classification.

## Hyperparameter Tuning
The following hyperparameters were tuned using **k-fold cross-validation**:
- **Learning rate**: [0.001, 0.0005]
- **Weight decay**: [0.001, 0.0005] (for AdamW optimizer)
- **Rho**: 0.9 (for RMSprop optimizer)

The goal was to find the best hyperparameter settings for each optimizer based on validation accuracy.

## Results
The performance of the three optimizers was evaluated on the test set after training for 10 epochs. The best hyperparameters for each optimizer were selected based on the highest validation accuracy from the cross-validation process.

The final model results are saved in `final_model_results.csv`.

## Visualization
The following plots are generated:
- **Training accuracy**: Comparison of training accuracy for each optimizer.
- **Test loss**: Comparison of test loss for each optimizer.

The plots can be found in the output folder or displayed directly during the training process.

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

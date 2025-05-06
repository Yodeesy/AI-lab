------

# MLP For House Price Prediction

## Overview

This project aims to implement a house price prediction task using a **Multilayer Perceptron (MLP)** model. The model is built using the PyTorch framework and compares the performance of different optimizers (SGD and Adam) for regression tasks. During model training, various techniques such as **Automatic Mixed Precision (AMP)**, **Huber Loss**, and **learning rate scheduling** are employed to improve training efficiency and performance.

## Structure

| File            | Description                                                  |
| --------------- | ------------------------------------------------------------ |
| `configs.yaml`  | Configuration file defining model, training parameters, and optimizer settings. |
| `dataset.py`    | Data loading and preprocessing, including dataset parsing and feature engineering. |
| `model.py`      | Defines the architecture of the MLP model.                   |
| `trainer.py`    | Contains the model training and evaluation process.          |
| `main.py`       | The entry point of the project, which loads configurations, initializes the model, and starts training. |
| `utils.py`      | Utility functions, such as data normalization and denormalization. |
| `visualizer.py` | Visualizes the loss and evaluation metric changes during training. |

## Steps

### 1. Install Dependencies

First, ensure that you have installed the required dependencies. You can use the following command to install the necessary libraries:

```bash
pip install -r requirements.txt
```

### 2. Configuration File

In the `configs.yaml`, you can configure the following settings:

- **Data Path**: Set the `data_path` to the path of your dataset.
- **Training Parameters**: Such as **epochs** (number of training epochs), **batch_size** (batch size), and **learning_rate**.
- **Optimizer Configuration**: Choose **SGD** or **Adam** as the optimizer.

### 3. Run Model Training

Run the `main.py` file to start model training. Use the following command in your terminal:

```bash
python main.py
```

The program will automatically load the configuration file, initialize the model, and begin training, while evaluating during the process.

### 4. Visualize Training Process

After training, you can visualize the loss and evaluation metrics (e.g., MAE, RMSE) using `visualizer.py`. This helps you understand the convergence and optimization progress of the model.

## Technical Details

### 1. **Model Architecture**

In this project, we use **Multilayer Perceptron (MLP)** for the regression task. The model consists of two hidden layers, each followed by a **ReLU** activation function and a **Dropout** layer to prevent overfitting.

### 2. **Loss Function and Optimizer**

- **Loss Function**: We use **Huber Loss** instead of MSE as it is more robust to outliers.
- **Optimizer**: We compare **SGD** and **Adam** optimizers. Adam is chosen for its better performance and faster convergence.

### 3. **Automatic Mixed Precision (AMP)**

To accelerate training and reduce memory usage, **Automatic Mixed Precision (AMP)** is employed. AMP automatically chooses between float32 and float16 precision for computations, reducing memory usage and speeding up training without sacrificing model accuracy.

### 4. **Learning Rate Scheduling**

We use the **ReduceLROnPlateau** learning rate scheduler, which reduces the learning rate when the validation performance plateaus, thus helping to optimize training in later stages.

### 5. **Early Stopping**

**Early Stopping** is applied to halt training when model performance on the validation set stops improving. This prevents overfitting and saves computation resources.

## Notes

- Ensure that you have **PyTorch** and the necessary libraries installed in your environment.
- Depending on the dataset size and hardware resources, training time may vary. Make sure your system is capable of handling large-scale training tasks.

## Copyright

This project is developed by Yodeeshi, intended for learning and research purposes. All rights are reserved to the original author, and commercial use is prohibited.
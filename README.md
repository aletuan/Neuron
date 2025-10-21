# Neuron - Deep Learning Educational Projects

This repository contains educational Jupyter notebooks covering Python fundamentals, data analysis with Pandas/NumPy, and various neural network architectures implemented from scratch and using TensorFlow/Keras.

## Projects

### Python Fundamentals & Data Analysis

#### basic/PY_LAB.ipynb
Comprehensive Python and data analysis tutorial covering:
- **Python Basics**: Variables, data types, operators, control flow, functions
- **Data Structures**: Lists, tuples, dictionaries, sets
- **NumPy**: Array operations, mathematical functions, broadcasting
- **Pandas**: DataFrames, data filtering, grouping, aggregation
- **Data Visualization**: Creating charts with Matplotlib (bar charts, line plots)
- **File I/O**: Reading CSV files, working with external data
- Practical examples with real-world datasets

### Deep Learning Projects

#### 1. CIFAR_CNN.ipynb
Convolutional Neural Network (CNN) for image classification on the CIFAR-10 dataset. Demonstrates:
- CNN architecture with convolutional and pooling layers
- Image preprocessing and data augmentation
- Model training and evaluation on 10-class image classification
- Integration with Weights & Biases for experiment tracking

#### 2. CORPUS_SRN.ipynb
Simple Recurrent Network (SRN) / Elman Network for sequence learning. Demonstrates:
- Basic RNN architecture with recurrent connections
- Character-level language modeling on toy corpus ("hello hinton")
- Understanding of temporal sequences and context layers
- Backpropagation Through Time (BPTT)
- Text generation using trained model

#### 3. YFIN_LSTM.ipynb
Long Short-Term Memory (LSTM) Network for financial time series prediction. Demonstrates:
- LSTM architecture with forget, input, and output gates
- Stock price prediction using Yahoo Finance (AAPL) data
- Time series preprocessing with 60-day lookback windows
- Train/test split preserving temporal order
- Model evaluation metrics (RMSE, MAE, MAPE, R²)
- Future price predictions using recursive forecasting

## Setup

### 1. Install Dependencies

```bash
# For Python fundamentals and data analysis (basic/PY_LAB.ipynb)
pip install numpy pandas matplotlib

# For deep learning notebooks
pip install tensorflow numpy matplotlib

# For CIFAR_CNN.ipynb (additional)
pip install pillow wandb python-dotenv pydot graphviz seaborn

# For YFIN_LSTM.ipynb (additional)
pip install yfinance pandas scikit-learn
```

### 2. Configure Weights & Biases (Optional - for CIFAR_CNN.ipynb)

Weights & Biases is used for experiment tracking in the CNN project.

1. Copy the environment template:
   ```bash
   cp .env.example .env
   ```

2. Get your wandb API key from https://wandb.ai/settings

3. Add your API key to `.env`:
   ```
   WANDB_API_KEY=your_actual_api_key_here
   ```

**Important**: Never commit your `.env` file to version control. It's already included in `.gitignore`.

## Running the Notebooks

```bash
jupyter notebook
```

Then open the desired notebook:
- `basic/PY_LAB.ipynb` - Python fundamentals and Pandas tutorial
- `CIFAR_CNN.ipynb` - CNN image classification
- `CORPUS_SRN.ipynb` - Simple RNN for text generation
- `YFIN_LSTM.ipynb` - LSTM for stock price prediction

## Project Structure

```
Neuron/
├── basic/
│   └── PY_LAB.ipynb         # Python fundamentals and Pandas tutorial
├── data/
│   └── data.csv             # Sample dataset for tutorials
├── CIFAR_CNN.ipynb          # CNN for image classification
├── CORPUS_SRN.ipynb         # Simple RNN for sequence learning
├── YFIN_LSTM.ipynb          # LSTM for stock prediction
├── .env.example             # Environment variables template
├── .gitignore               # Git ignore rules
├── images/                  # Generated images and plots
├── models/                  # Saved model checkpoints
├── test_images/             # Test images for evaluation
└── wandb/                   # Weights & Biases logs
```

## Key Concepts Covered

### Python & Data Analysis Fundamentals
- **Python Programming**: Variables, control flow, functions, data structures
- **NumPy**: Array manipulation, mathematical operations, broadcasting
- **Pandas**: DataFrames, filtering, grouping, aggregation, pivoting
- **Data Visualization**: Matplotlib for creating informative charts
- **Data Handling**: CSV file operations, data cleaning and transformation

### Neural Network Architectures
- **CNN (Convolutional Neural Networks)**: Spatial feature extraction for images
- **RNN (Recurrent Neural Networks)**: Sequential data processing with memory
- **LSTM (Long Short-Term Memory)**: Advanced RNN with gating mechanisms

### Deep Learning Techniques
- Backpropagation and gradient descent
- Dropout and regularization for preventing overfitting
- Batch normalization and data augmentation
- Time series preprocessing and sliding windows
- Model evaluation metrics (accuracy, loss, RMSE, MAE, R²)

### Practical Applications
- Data analysis and visualization
- Computer vision (image classification)
- Natural language processing (character-level language modeling)
- Financial forecasting (stock price prediction)

## Security Notes

- API keys and sensitive credentials are stored in `.env` file (not tracked by git)
- See `.env.example` for required environment variables
- The CIFAR_CNN notebook automatically loads environment variables using `python-dotenv`

## Resources

- Python Documentation: https://docs.python.org/
- NumPy Documentation: https://numpy.org/doc/
- Pandas Documentation: https://pandas.pydata.org/docs/
- Matplotlib Documentation: https://matplotlib.org/stable/contents.html
- TensorFlow/Keras Documentation: https://www.tensorflow.org/
- Weights & Biases: https://wandb.ai/
- Yahoo Finance API: https://pypi.org/project/yfinance/

## License

Educational use only.

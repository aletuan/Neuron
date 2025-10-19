# CNN Projects

This repository contains educational Jupyter notebooks demonstrating deep learning architectures from scratch.

## Projects

- **CIFAR_CNN.ipynb**: Convolutional Neural Network for image classification on CIFAR-10 dataset
- **Transformer_Fundamentals.ipynb**: Transformer architecture from scratch using NumPy for text classification

## Setup

### 1. Install Dependencies

```bash
pip install tensorflow pillow matplotlib wandb python-dotenv pydot graphviz numpy seaborn
```

### 2. Configure Weights & Biases (wandb)

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

Then open the desired notebook (e.g., `CIFAR_CNN.ipynb` or `Transformer_Fundamentals.ipynb`).

## Security Notes

- API keys and sensitive credentials are stored in `.env` file (not tracked by git)
- See `.env.example` for required environment variables
- The notebooks automatically load environment variables using `python-dotenv`

# About the project
This project emulates how the real LLMs work. The model predicts the next token, so the overall text reminds Shakespeare's style

# Features
* Preprocessing of raw text data into a machine-readable format
* Tokenization and sequence creation
* Training a neural network for next-word prediction
* Generating text in the Shakespeare's style

# Installation
The project uses poetry as the dependency manager and Jupyter notebook as the development IDE:

## Setup

```bash
# Ensure dependencies are installed
poetry install

# Add Jupyter + kernel support (if not already added)
poetry add jupyter ipykernel

# Activate Poetry environment
poetry shell

# Register this environment as a Jupyter kernel
python -m ipykernel install --user --name=$(basename "$PWD") --display-name "Poetry ($(basename "$PWD"))"

# Launch Jupyter
poetry run jupyter notebook
```

In Jupyter notebook, please, navigate to Kernel → Change Kernel, then find the newly added kernel and the list. After set up is done, you can run the cells of main.ipynb

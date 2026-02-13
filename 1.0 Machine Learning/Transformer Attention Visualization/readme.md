# Transformer Attention Visualization (GPU-enabled)

This project provides a **local, interactive visualization of a transformer attention mechanism** using PyTorch and Hugging Face Transformers. It allows you to understand token embeddings, attention weights, and MLP outputs.

---

## Features

* Tokenizes input sentences using BERT tokenizer.
* Converts tokens to embeddings.
* Implements a **single-head attention block** (Query, Key, Value, Scaled Dot-Product, Softmax).
* Applies a feedforward MLP to attention outputs.
* Visualizes attention weights as a heatmap.
* Projects MLP output embeddings to 2D using PCA.
* Fully **GPU-accelerated** if available.

---

## Requirements

```bash
torch
transformers
matplotlib
seaborn
scikit-learn
```

Optional GPU acceleration requires a CUDA-enabled GPU and PyTorch with CUDA support.

---

## Usage

1. Clone the repository or download the script.
2. Install dependencies.
3. Run the script:

```bash
python transformer_attention_viz.py
```

4. The script will output:

   * Tokenized sentence.
   * Attention heatmap.
   * 2D PCA visualization of MLP output embeddings.

---

## Example

Input sentence:

```
"Transformers are amazing for NLP!"
```

Outputs:

1. Tokens:

```
['[CLS]', 'transformers', 'are', 'amazing', 'for', 'nl', '##p', '!', '[SEP]']
```

2. Attention heatmap (token-to-token similarity).
3. 2D embedding plot of MLP outputs.

---

## Model Architecture

```
Input Sentence --> Tokenization --> Embedding Layer --> Attention Block --> Feedforward MLP --> Visualizations (Heatmap & PCA)
```

**Attention Block Details:**

* Q = W\_q \* Embeddings
* K = W\_k \* Embeddings
* V = W\_v \* Embeddings
* Attention Weights = softmax((Q\*K^T)/sqrt(d))
* Context Vector = Attention Weights \* V

---

## Optional Enhancements

* Multi-head attention visualization.
* Residual connections + LayerNorm.
* Interactive visualization with Plotly Dash or Streamlit.

---

## License

MIT License

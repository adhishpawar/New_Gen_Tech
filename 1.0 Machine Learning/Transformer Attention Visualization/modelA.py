import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel
import matplotlib.pyplot as plt
import seaborn as sns

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)


# -------------------------------
# Tokenization
# -------------------------------
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
model = AutoModel.from_pretrained("bert-base-uncased").to(device)
model.eval()  # evaluation mode

sentence = "Transformers are amazing for NLP!"
inputs = tokenizer(sentence, return_tensors="pt").to(device)
input_ids = inputs['input_ids']  # [1, seq_len]

tokens = tokenizer.convert_ids_to_tokens(input_ids[0])
print("Tokens:", tokens)



# -------------------------------
# Embeddings
# -------------------------------
embedding_layer = model.get_input_embeddings()
token_embeddings = embedding_layer(input_ids)  # [1, seq_len, embedding_dim]
seq_len, embedding_dim = token_embeddings.shape[1], token_embeddings.shape[2]


# -------------------------------
# Attention Mechanism
# -------------------------------
# Simple single-head attention
class SimpleAttention(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.embed_dim = embed_dim
        self.W_q = nn.Linear(embed_dim, embed_dim)
        self.W_k = nn.Linear(embed_dim, embed_dim)
        self.W_v = nn.Linear(embed_dim, embed_dim)

    def forward(self, x):
        Q = self.W_q(x)  # [batch, seq_len, embed_dim]
        K = self.W_k(x)
        V = self.W_v(x)
        
        # Scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.embed_dim ** 0.5)  # [batch, seq_len, seq_len]
        attention_weights = torch.softmax(scores, dim=-1)
        output = torch.matmul(attention_weights, V)
        return output, attention_weights

attention_layer = SimpleAttention(embedding_dim).to(device)
attention_output, attention_weights = attention_layer(token_embeddings)

# -------------------------------
# Feedforward MLP
# -------------------------------
mlp = nn.Sequential(
    nn.Linear(embedding_dim, embedding_dim),
    nn.ReLU(),
    nn.Linear(embedding_dim, embedding_dim)
).to(device)

mlp_output = mlp(attention_output)


# -------------------------------
# Visualization: Attention Heatmap
# -------------------------------
# Detach tensors to CPU for plotting
attn_weights_np = attention_weights[0].detach().cpu().numpy()

plt.figure(figsize=(8,6))
sns.heatmap(attn_weights_np, xticklabels=tokens, yticklabels=tokens, annot=True, cmap="viridis")
plt.title("Attention Weights")
plt.show()


# -------------------------------
#  Visualize Embeddings (first 2D projection)
# -------------------------------
from sklearn.decomposition import PCA
import numpy as np

# Flatten embeddings to 2D
embeddings_2d = PCA(n_components=2).fit_transform(mlp_output[0].detach().cpu().numpy())

plt.figure(figsize=(6,6))
plt.scatter(embeddings_2d[:,0], embeddings_2d[:,1])

for i, token in enumerate(tokens):
    plt.text(embeddings_2d[i,0]+0.01, embeddings_2d[i,1]+0.01, token, fontsize=12)

plt.title("MLP Output Embeddings (2D PCA)")
plt.show()
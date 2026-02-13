import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA

# -------------------------------
# 1️⃣ Device
# -------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# -------------------------------
# 2️⃣ Tokenization (Batch Input)
# -------------------------------
batch_size = 128
sentence = "Transformers are amazing for NLP! Transformers are amazing for NLP! Transformers are amazing for NLP! Transformers are amazing for NLP! Transformers are amazing for NLP! Transformers are amazing for NLP! Transformers are amazing for NLP! Transformers are amazing for NLP! These models can capture long-range dependencies, understand context, and generate meaningful text. By using multi-head attention, Transformers can focus on different parts of the input simultaneously, making them extremely powerful for tasks like language translation, summarization, and question answering. Their ability to scale with data and compute has revolutionized natural language processing, enabling models to learn complex patterns and semantics. With careful training and fine-tuning, Transformers can achieve state-of-the-art performance on a wide variety of NLP tasks, from sentiment analysis to text generation and beyond. Their flexibility allows them to handle multiple languages and domains, making them versatile tools in modern AI applications. As research continues, Transformers are being adapted for multimodal tasks, combining text, vision, and audio, further expanding their capabilities and impact in AI." 
sentences = [sentence] * batch_size

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
model = AutoModel.from_pretrained("bert-base-uncased").to(device)
model.eval()

inputs = tokenizer(sentences, return_tensors="pt", padding=True, truncation=True).to(device)
input_ids = inputs['input_ids']  # [batch_size, seq_len]
tokens = tokenizer.convert_ids_to_tokens(input_ids[0])
print("Tokens:", tokens)

# -------------------------------
# 3️⃣ Embeddings
# -------------------------------
embedding_layer = model.get_input_embeddings()
token_embeddings = embedding_layer(input_ids)  # [batch_size, seq_len, embed_dim]
batch_size, seq_len, embed_dim = token_embeddings.shape

# -------------------------------
# 4️⃣ Multi-Head Attention
# -------------------------------
class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads=8):
        super().__init__()
        assert embed_dim % num_heads == 0
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        
        self.W_q = nn.Linear(embed_dim, embed_dim)
        self.W_k = nn.Linear(embed_dim, embed_dim)
        self.W_v = nn.Linear(embed_dim, embed_dim)
        self.out = nn.Linear(embed_dim, embed_dim)
    
    def forward(self, x):
        Q = self.W_q(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1,2)
        K = self.W_k(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1,2)
        V = self.W_v(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1,2)
        
        scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.head_dim ** 0.5)
        attn_weights = torch.softmax(scores, dim=-1)
        output = torch.matmul(attn_weights, V)
        output = output.transpose(1,2).contiguous().view(batch_size, seq_len, embed_dim)
        return self.out(output), attn_weights

attention_layer = MultiHeadAttention(embed_dim, num_heads=8).to(device)
attention_output, attention_weights = attention_layer(token_embeddings)

# -------------------------------
# 5️⃣ Feedforward MLP
# -------------------------------
mlp = nn.Sequential(
    nn.Linear(embed_dim, embed_dim*2),
    nn.ReLU(),
    nn.Linear(embed_dim*2, embed_dim)
).to(device)
mlp_output = mlp(attention_output)

# -------------------------------
# 6️⃣ Visualization: Attention Heatmap (First Head)
# -------------------------------
attn_weights_np = attention_weights[0,0].detach().cpu().numpy()  # First head
plt.figure(figsize=(10,8))
sns.heatmap(attn_weights_np, xticklabels=tokens, yticklabels=tokens, annot=False, cmap="viridis")
plt.title("Attention Weights (Head 0)")
plt.show()

# -------------------------------
# 7️⃣ Visualize Embeddings (PCA)
# -------------------------------
embeddings_2d = PCA(n_components=2).fit_transform(mlp_output[0].detach().cpu().numpy())
plt.figure(figsize=(8,8))
plt.scatter(embeddings_2d[:,0], embeddings_2d[:,1])
for i, token in enumerate(tokens):
    plt.text(embeddings_2d[i,0]+0.01, embeddings_2d[i,1]+0.01, token, fontsize=12)
plt.title("MLP Output Embeddings (2D PCA)")
plt.show()
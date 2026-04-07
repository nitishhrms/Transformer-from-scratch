import torch
import torch.nn.functional as F
import tiktoken

from model import GPT, GPTConfig
from data import DataLoaderLite

# -------------------------------------------------------------------
# Config
# -------------------------------------------------------------------
num_return_sequences = 4
max_length = 50

device = "cpu"
if torch.cuda.is_available():
    device = "cuda"
print(f"using device: {device}")

# -------------------------------------------------------------------
# Model
# -------------------------------------------------------------------
model = GPT(GPTConfig())
model.to(device)
print("model initialized")

# -------------------------------------------------------------------
# Data
# -------------------------------------------------------------------
train_loader = DataLoaderLite(B=4, T=32)
x, y = train_loader.next_batch()
x, y = x.to(device), y.to(device)

# -------------------------------------------------------------------
# Training loop
# -------------------------------------------------------------------
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)

for i in range(50):
    optimizer.zero_grad()
    logits, loss = model(x, y)
    loss.backward()
    optimizer.step()
    print(f"step {i}, loss: {loss.item():.4f}")

# -------------------------------------------------------------------
# Generation
# -------------------------------------------------------------------
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed(42)

model.eval()
enc = tiktoken.get_encoding('gpt2')

while x.size(1) < max_length:
    with torch.no_grad():
        logits, _ = model(x)
        logits = logits[:, -1, :]
        probs = F.softmax(logits, dim=-1)
        topk_probs, topk_indices = torch.topk(probs, 50, dim=-1)
        ix = torch.multinomial(topk_probs, 1)
        xcol = torch.gather(topk_indices, -1, ix)
        x = torch.cat((x, xcol), dim=1)

for i in range(num_return_sequences):
    tokens = x[i, :max_length].tolist()
    decoded = enc.decode(tokens)
    print(">", decoded)

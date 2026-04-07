from  dataclasses import dataclass
import torch
import torch.nn as nn

import torch.nn.functional as F

class CausalSelfAttention(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        # output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        self.c_proj.NANOGPT_SCALE_INIT = 1
        # regularization
        self.n_head = config.n_head
        self.n_embd = config.n_embd

    def forward(self, x):
        B, T, C = x.size() # batch size, sequence length, embedding dimensionality (n_embd)
        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        # nh is "number of heads", hs is "head size", and C (number of channels) = nh * hs
        # e.g. in GPT-2 (124M), n_head=12, hs=64, so nh*hs=C=768 channels in the Transformer
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True) # flash attention
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side
        # output projection
        y = self.c_proj(y)
        return y



class mlp(nn.Module):


    def __init__(self,config):
        super().__init__()
        self.c_fc=nn.Linear(config.n_embd,4*config.n_embd)
        self.gelu=nn.GELU(approximate='tanh')
        self.c_proj=nn.Linear(4*config.n_embd,config.n_embd)
    
    def forward(self,x):
        x=self.c_fc(x)
        x=self.gelu(x)
        x=self.c_proj(x)
        return x


class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = mlp(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x





@dataclass
class GPTConfig:
    block_size: int = 1024 # max sequence length
    vocab_size: int = 50257 # number of tokens: 50,000 BPE merges + 256 bytes tokens + 1 <|endoftext|> token
    n_layer: int = 12 # number of layers
    n_head: int = 12 # number of heads
    n_embd: int = 768 # Embedding dimension



class GPT(nn.Module):


    def __init__(self,config):
        
        super().__init__()
        self.config=config

        self.transformer=nn.ModuleDict(dict(
            ###token Embedding
            wte=nn.Embedding(config.vocab_size,config.n_embd),

            wpe=nn.Embedding(config.block_size,config.n_embd),

            h=nn.ModuleList([Block(config) for _ in range(config.n_layer)]),

            ln_f=nn.LayerNorm(config.n_embd)

        ))

        self.lm_head=nn.Linear(config.n_embd,config.vocab_size,bias=False)

        ###reducess the parameters also

        self.transformer.wte.weight=self.lm_head.weight


    def forward(self,idx,targets=None):

        assert T <= self.config.block_size, "Sequence too long"

        B,T=idx.size()

        pos=torch.arange(0,T,dtype=torch.long,device=idx.device)

        pos_emb=self.transformer.wpe(pos)    ### (T,embed)

        tok_emb=self.transformer.wte(idx)  ###(B,T,embd)

        x=tok_emb+pos_emb   ###x=(batch,T,embedding)

        for block in self.transformer.h:
            x=block(x)
        
        x=self.transformer.ln_f(x)

        logits=self.lm_head(x) ##(b,t,vocab_size)

        loss = None

        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        
        return logits, loss
        


    @classmethod

    def from_pretrained(cls, model_type):
        """Loads pretrained GPT-2 model weights from huggingface"""
        assert model_type in {'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'}

        from transformers import GPT2LMHeadModel

        print("loading weights from pretrained gpt: %s" % model_type)
        
        # n_layer, n_head and n_embd are determined from model_type
        config_args = {
            'gpt2':         dict(n_layer=12, n_head=12, n_embd=768),  # 124M params
            'gpt2-medium':  dict(n_layer=24, n_head=16, n_embd=1024), # 350M params
            'gpt2-large':   dict(n_layer=36, n_head=20, n_embd=1280), # 774M params
            'gpt2-xl':      dict(n_layer=48, n_head=25, n_embd=1600), # 1558M params
        }[model_type]
        config_args['vocab_size'] = 50257 # always 50257 for GPT model checkpoints
        config_args['block_size'] = 1024 # always 1024 for GPT model checkpoints
        # create a from-scratch initialized minGPT model
        config = GPTConfig(**config_args)
        model = GPT(config)
        sd = model.state_dict()
        sd_keys = sd.keys()
        sd_keys = [k for k in sd_keys if not k.endswith('.attn.bias')] # discard this mask / buffer, not a param

        # init a huggingface/transformers model
        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf = model_hf.state_dict()

        # copy while ensuring all of the parameters are aligned and match in names and shapes
        sd_keys_hf = sd_hf.keys()
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.masked_bias')] # ignore these, just a buffer
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.bias')] # same, just the mask (buffer)
        transposed = ['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight']
        # basically the openai checkpoints use a "Conv1D" module, but we only want to use a vanilla Linear
        # this means that we have to transpose these weights when we import them
        assert len(sd_keys_hf) == len(sd_keys), f"mismatched keys: {len(sd_keys_hf)} != {len(sd_keys)}"
        for k in sd_keys_hf:
            if any(k.endswith(w) for w in transposed):
                # special treatment for the Conv1D weights we need to transpose
                assert sd_hf[k].shape[::-1] == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k].t())
            else:
                # vanilla copy over the other parameters
                assert sd_hf[k].shape == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k])

        return model

import tiktoken
import torch




# transformer.wte.weight torch.Size([50257, 768])
# transformer.wpe.weight torch.Size([1024, 768])
# transformer.h.0.ln_1.weight torch.Size([768])
# transformer.h.0.ln_1.bias torch.Size([768])
# transformer.h.0.attn.c_attn.weight torch.Size([768, 2304])
# transformer.h.0.attn.c_attn.bias torch.Size([2304])
# transformer.h.0.attn.c_proj.weight torch.Size([768, 768])
# transformer.h.0.attn.c_proj.bias torch.Size([768])
# transformer.h.0.ln_2.weight torch.Size([768])
# transformer.h.0.ln_2.bias torch.Size([768])
# transformer.h.0.mlp.c_fc.weight torch.Size([768, 3072])
# transformer.h.0.mlp.c_fc.bias torch.Size([3072])
# transformer.h.0.mlp.c_proj.weight torch.Size([3072, 768])
# transformer.h.0.mlp.c_proj.bias torch.Size([768])
# transformer.h.1.ln_1.weight torch.Size([768])
# transformer.h.1.ln_1.bias torch.Size([768])
# transformer.h.1.attn.c_attn.weight torch.Size([768, 2304])
# transformer.h.1.attn.c_attn.bias torch.Size([2304])
# transformer.h.1.attn.c_proj.weight torch.Size([768, 768])
# transformer.h.1.attn.c_proj.bias torch.Size([768])
# transformer.h.1.ln_2.weight torch.Size([768])
# transformer.h.1.ln_2.bias torch.Size([768])
# transformer.h.1.mlp.c_fc.weight torch.Size([768, 3072])
# transformer.h.1.mlp.c_fc.bias torch.Size([3072])
# transformer.h.1.mlp.c_proj.weight torch.Size([3072, 768])
# transformer.h.1.mlp.c_proj.bias torch.Size([768])
# transformer.h.2.ln_1.weight torch.Size([768])
# transformer.h.2.ln_1.bias torch.Size([768])
# transformer.h.2.attn.c_attn.weight torch.Size([768, 2304])
# transformer.h.2.attn.c_attn.bias torch.Size([2304])
# transformer.h.2.attn.c_proj.weight torch.Size([768, 768])
# transformer.h.2.attn.c_proj.bias torch.Size([768])
# transformer.h.2.ln_2.weight torch.Size([768])
# transformer.h.2.ln_2.bias torch.Size([768])
# transformer.h.2.mlp.c_fc.weight torch.Size([768, 3072])
# transformer.h.2.mlp.c_fc.bias torch.Size([3072])
# transformer.h.2.mlp.c_proj.weight torch.Size([3072, 768])
# transformer.h.2.mlp.c_proj.bias torch.Size([768])
# transformer.h.3.ln_1.weight torch.Size([768])
# transformer.h.3.ln_1.bias torch.Size([768])
# transformer.h.3.attn.c_attn.weight torch.Size([768, 2304])
# transformer.h.3.attn.c_attn.bias torch.Size([2304])
# transformer.h.3.attn.c_proj.weight torch.Size([768, 768])
# transformer.h.3.attn.c_proj.bias torch.Size([768])
# transformer.h.3.ln_2.weight torch.Size([768])
# transformer.h.3.ln_2.bias torch.Size([768])
# transformer.h.3.mlp.c_fc.weight torch.Size([768, 3072])
# transformer.h.3.mlp.c_fc.bias torch.Size([3072])
# transformer.h.3.mlp.c_proj.weight torch.Size([3072, 768])
# transformer.h.3.mlp.c_proj.bias torch.Size([768])
# transformer.h.4.ln_1.weight torch.Size([768])
# transformer.h.4.ln_1.bias torch.Size([768])
# transformer.h.4.attn.c_attn.weight torch.Size([768, 2304])
# transformer.h.4.attn.c_attn.bias torch.Size([2304])
# transformer.h.4.attn.c_proj.weight torch.Size([768, 768])
# transformer.h.4.attn.c_proj.bias torch.Size([768])
# transformer.h.4.ln_2.weight torch.Size([768])
# transformer.h.4.ln_2.bias torch.Size([768])
# transformer.h.4.mlp.c_fc.weight torch.Size([768, 3072])
# transformer.h.4.mlp.c_fc.bias torch.Size([3072])
# transformer.h.4.mlp.c_proj.weight torch.Size([3072, 768])
# transformer.h.4.mlp.c_proj.bias torch.Size([768])
# transformer.h.5.ln_1.weight torch.Size([768])
# transformer.h.5.ln_1.bias torch.Size([768])
# transformer.h.5.attn.c_attn.weight torch.Size([768, 2304])
# transformer.h.5.attn.c_attn.bias torch.Size([2304])
# transformer.h.5.attn.c_proj.weight torch.Size([768, 768])
# transformer.h.5.attn.c_proj.bias torch.Size([768])
# transformer.h.5.ln_2.weight torch.Size([768])
# transformer.h.5.ln_2.bias torch.Size([768])
# transformer.h.5.mlp.c_fc.weight torch.Size([768, 3072])
# transformer.h.5.mlp.c_fc.bias torch.Size([3072])
# transformer.h.5.mlp.c_proj.weight torch.Size([3072, 768])
# transformer.h.5.mlp.c_proj.bias torch.Size([768])
# transformer.h.6.ln_1.weight torch.Size([768])
# transformer.h.6.ln_1.bias torch.Size([768])
# transformer.h.6.attn.c_attn.weight torch.Size([768, 2304])
# transformer.h.6.attn.c_attn.bias torch.Size([2304])
# transformer.h.6.attn.c_proj.weight torch.Size([768, 768])
# transformer.h.6.attn.c_proj.bias torch.Size([768])
# transformer.h.6.ln_2.weight torch.Size([768])
# transformer.h.6.ln_2.bias torch.Size([768])
# transformer.h.6.mlp.c_fc.weight torch.Size([768, 3072])
# transformer.h.6.mlp.c_fc.bias torch.Size([3072])
# transformer.h.6.mlp.c_proj.weight torch.Size([3072, 768])
# transformer.h.6.mlp.c_proj.bias torch.Size([768])
# transformer.h.7.ln_1.weight torch.Size([768])
# transformer.h.7.ln_1.bias torch.Size([768])
# transformer.h.7.attn.c_attn.weight torch.Size([768, 2304])
# transformer.h.7.attn.c_attn.bias torch.Size([2304])
# transformer.h.7.attn.c_proj.weight torch.Size([768, 768])
# transformer.h.7.attn.c_proj.bias torch.Size([768])
# transformer.h.7.ln_2.weight torch.Size([768])
# transformer.h.7.ln_2.bias torch.Size([768])
# transformer.h.7.mlp.c_fc.weight torch.Size([768, 3072])
# transformer.h.7.mlp.c_fc.bias torch.Size([3072])
# transformer.h.7.mlp.c_proj.weight torch.Size([3072, 768])
# transformer.h.7.mlp.c_proj.bias torch.Size([768])
# transformer.h.8.ln_1.weight torch.Size([768])
# transformer.h.8.ln_1.bias torch.Size([768])
# transformer.h.8.attn.c_attn.weight torch.Size([768, 2304])
# transformer.h.8.attn.c_attn.bias torch.Size([2304])
# transformer.h.8.attn.c_proj.weight torch.Size([768, 768])
# transformer.h.8.attn.c_proj.bias torch.Size([768])
# transformer.h.8.ln_2.weight torch.Size([768])
# transformer.h.8.ln_2.bias torch.Size([768])
# transformer.h.8.mlp.c_fc.weight torch.Size([768, 3072])
# transformer.h.8.mlp.c_fc.bias torch.Size([3072])
# transformer.h.8.mlp.c_proj.weight torch.Size([3072, 768])
# transformer.h.8.mlp.c_proj.bias torch.Size([768])
# transformer.h.9.ln_1.weight torch.Size([768])
# transformer.h.9.ln_1.bias torch.Size([768])
# transformer.h.9.attn.c_attn.weight torch.Size([768, 2304])
# transformer.h.9.attn.c_attn.bias torch.Size([2304])
# transformer.h.9.attn.c_proj.weight torch.Size([768, 768])
# transformer.h.9.attn.c_proj.bias torch.Size([768])
# transformer.h.9.ln_2.weight torch.Size([768])
# transformer.h.9.ln_2.bias torch.Size([768])
# transformer.h.9.mlp.c_fc.weight torch.Size([768, 3072])
# transformer.h.9.mlp.c_fc.bias torch.Size([3072])
# transformer.h.9.mlp.c_proj.weight torch.Size([3072, 768])
# transformer.h.9.mlp.c_proj.bias torch.Size([768])
# transformer.h.10.ln_1.weight torch.Size([768])
# transformer.h.10.ln_1.bias torch.Size([768])
# transformer.h.10.attn.c_attn.weight torch.Size([768, 2304])
# transformer.h.10.attn.c_attn.bias torch.Size([2304])
# transformer.h.10.attn.c_proj.weight torch.Size([768, 768])
# transformer.h.10.attn.c_proj.bias torch.Size([768])
# transformer.h.10.ln_2.weight torch.Size([768])
# transformer.h.10.ln_2.bias torch.Size([768])
# transformer.h.10.mlp.c_fc.weight torch.Size([768, 3072])
# transformer.h.10.mlp.c_fc.bias torch.Size([3072])
# transformer.h.10.mlp.c_proj.weight torch.Size([3072, 768])
# transformer.h.10.mlp.c_proj.bias torch.Size([768])
# transformer.h.11.ln_1.weight torch.Size([768])
# transformer.h.11.ln_1.bias torch.Size([768])
# transformer.h.11.attn.c_attn.weight torch.Size([768, 2304])
# transformer.h.11.attn.c_attn.bias torch.Size([2304])
# transformer.h.11.attn.c_proj.weight torch.Size([768, 768])
# transformer.h.11.attn.c_proj.bias torch.Size([768])
# transformer.h.11.ln_2.weight torch.Size([768])
# transformer.h.11.ln_2.bias torch.Size([768])
# transformer.h.11.mlp.c_fc.weight torch.Size([768, 3072])
# transformer.h.11.mlp.c_fc.bias torch.Size([3072])
# transformer.h.11.mlp.c_proj.weight torch.Size([3072, 768])
# transformer.h.11.mlp.c_proj.bias torch.Size([768])
# transformer.ln_f.weight torch.Size([768])
# transformer.ln_f.bias torch.Size([768])
# lm_head.weight torch.Size([50257, 768])



# Start coding or generate with AI.

num_return_sequences=4
max_length=50


device="cpu"

if torch.cuda.is_available():
    device="cuda"

print(f'using_device:{device}')

####model=GPT.from_pretrained("gpt2")

###model.eval()

###model.to(device)

print("didn't crash")


# import tiktoken

# enc=tiktoken.get_encoding("gpt2")

# tokens=enc.encode("Hello! i am a language model")

# tokens=torch.tensor(tokens,dtype=torch.long)

# tokens=tokens.unsqueeze(0).repeat(num_return_sequences,1)

# x=tokens.to(device)

import torch
import tiktoken

# get a data batch
enc = tiktoken.get_encoding('gpt2')

with open("input.txt", "r") as f:
    text = f.read()

# take a small chunk
text = text[:1000]

# tokenize
tokens = enc.encode(text)


import tiktoken
import torch


class DataLoaderLite:
    def __init__(self, B, T):
        self.B = B
        self.T = T

        # at init load tokens from disk and store them in memory
        with open('input.txt', 'r') as f:
            text = f.read()
        enc = tiktoken.get_encoding('gpt2')
        tokens = enc.encode(text)
        self.tokens = torch.tensor(tokens)
        print(f"loaded {len(self.tokens)} tokens")
        print(f"1 epoch = {len(self.tokens) // (B * T)} batches")

        # state
        self.current_position = 0

    def next_batch(self):
        B, T = self.B, self.T
        buf = self.tokens[self.current_position : self.current_position + B * T + 1]
        x = buf[:-1].view(B, T)  # inputs
        y = buf[1:].view(B, T)   # targets

        # advance the position in the tensor
        self.current_position += B * T

        # if loading the next batch would be out of bounds, reset
        if self.current_position + (B * T + 1) > len(self.tokens):
            self.current_position = 0

        return x, y
# get logits

model=GPT(GPTConfig())

model.to(device)

logits,loss = model(x)

optimizer=torch.optim.AdamW(model.parameters(),lr=3e-4)

for i in range(50):
    optimizer.zero_grad()
    logits,loss=model(x,y)
    loss.backward()
    optimizer.step()
    print(f"step {i}, loss: {loss.item()}")


print("logits shape:", logits.shape)  # (B, T, vocab_size)


torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed(42)
##torch.cuda.manual_seed(42)

# prompt_tokens = enc.encode("Hello, I am a language model,")
# x_gen = torch.tensor(prompt_tokens, dtype=torch.long, device=device)
# x_gen = x_gen.unsqueeze(0).repeat(num_return_sequences, 1)  # (num_return_sequences, prompt_len)
 
# FIX 5: Switch to eval mode before generation (disables dropout etc.)
model.eval()

while x.size(1)<max_length:
    with torch.no_grad():

        logits,_=model(x)

        logits=logits[:,-1,:]

        probs=F.softmax(logits,dim=-1)

        # #we will take top 50 proabablities
        topk_probs,topk_indices=torch.topk(probs,50,dim=-1)

        ix=torch.multinomial(topk_probs,1)

        xcol=torch.gather(topk_indices,-1,ix)

        x=torch.cat((x,xcol),dim=1)


for i in range(num_return_sequences):
    tokens=x[i,:max_length].tolist()
    decoded=enc.decode(tokens)
    print(">",decoded)






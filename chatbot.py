import torch
import torch.nn as nn 
from torch.nn import functional as F 
import mmap
import random
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-batch_size', type=str, help="Please  provide a batch size", default=64)

args = parser.parse_args()

print(f"batch_size: {args.batch_size}")

device = "cuda" if torch.cuda.is_available() else "cpu"
# device = "cpu"

block_size = args.batch_size
batch_size = 32
max_iters = 3000
learning_rate = 3e-4
eval_iters = 100
n_embd = 384
n_head = 8
n_layers = 8
dropout = 0.2



# ========= 读入词典 ============
with open('./openwebtext/vocab.txt', 'r', encoding='utf-8') as f:
    text = f.read()
    
chars = sorted(set(text))
vocab_size = len(chars)

# 字符到索引的映射
string_to_int = {ch:i for i, ch in enumerate(chars)}
int_to_string = {i:ch for i, ch in enumerate(chars)}
# encoder 和 decoder
encode = lambda s: [string_to_int[c] for c in s]
decode = lambda l: ''.join([int_to_string[i] for i in l])  # 接受一个整数列表 l 转换为字符串





### ============ 模型 ==========
class Head(nn.Module):
    """ one head of self-attention 
        head_size: 这个头所捕获的特征数量
    """
    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        # 掩码层, 在模型中注册 no look ahead masking
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        # 输入维度 (batch_size, time-step, channels)
        # 输出维度 (batch_size, time-step, head_size)
        B, T, C = x.shape
        k = self.key(x)     # (B, T, hs)
        q = self.query(x)   # (B, T, hs)
        # 计算注意力分数 （affinities）
        # (B, T, hs) * (B, hs, T) -> (B, T, T)
        wei = q @ k.transpose(-2, -1) * k.shape[-1] ** -0.5
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf')) # (B, T, T)
        wei = F.softmax(wei, dim=-1) # (B, T, T)
        wei = self.dropout(wei)
        # 执行值的加权聚合
        v = self.value(x)
        out = wei @ v # (B, T, T) @ (B, T, hs) -> (B, T, hs)
        return out
    
    
    
class FeedFoward(nn.Module):
    """ 一个简单的前馈神经网络 """
    def __init__(self,n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            # 防止过拟合
            nn.Dropout(dropout),
        )
    def forward(self, x):
        return self.net(x)

class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        # 投影 projection 将头大小*头数量投影到嵌入中
        # 这样写只是为了便于以后修改，可以复用
        # 且能够帮助网络更多的了解这个文本，添加更多可学习的参数
        self.proj = nn.Linear(head_size * num_heads, n_embd)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        # 拼接词向量维度
        out = torch.cat([h(x) for h in self.heads], dim=-1) #(B, T, C) 最后一个维度C
        out = self.dropout(self.proj(out))
        return out

class Block(nn.Module):
    """ Transformer block """
    def __init__(self, n_embd, n_head):
        # n_embd 嵌入维度  n_head:头数量
        super().__init__()
        # head_size 每个头在多头注意力中要捕获的特征数量
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedFoward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)
        
    def forward(self, x):
        y = self.sa(x)
        ## add-norm
        x = self.ln1(x + y)
        y = self.ffwd(x)
        x = self.ln2(x+y)
        return x

class GPTLanguageModel(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        # 嵌入表 vocab_size * n_embd
        self.token_embeddings_table = nn.Embedding(vocab_size, n_embd)
        # 位置编码 block_size：序列长度
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        # 添加decoder blocks
        self.blocks = nn.Sequential(*[Block(n_embd, n_head=n_head) for _ in range(n_layers) ])
        # layer norm final 帮助模型更好的收敛（层规范化）
        self.ln_f = nn.LayerNorm(n_embd)
        # language model head 最终投影（变换）
        self.lm_head = nn.Linear(n_embd, vocab_size)
        
        # 初始化参数
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        
    # index: (B, T)
    def forward(self, index, targets=None):
        B, T = index.shape
        # index 和 targets 都是 (B, T)形状的整形张量 B:batch_size T:seqLen
        # (B, T, C)
        tok_emb = self.token_embeddings_table(index)
        # (T, C)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device))
        # 将这两个词向量相加
        x = tok_emb + pos_emb # 广播 (B, T, C)
        x = self.blocks(x) # (B, T, C)
        x = self.ln_f(x)   # (B, T, C)
        logits = self.lm_head(x) # (B, T, vocab_size)
        
        
        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)
        
        return logits, loss
    
    # 生成token的函数
    def generate(self, index, max_new_tokens):
        # index is (B, T) array of indices in the current context
        # 根据需要生成的token数进行循环
        for _ in range(max_new_tokens):
            # 剪裁tokens， 不要超过block size
            index_cond = index[:, -block_size:]
            # 获得预测， 这里没有给target，返回的logits是三维的 B*T*C
            logits, loss = self.forward(index_cond)
            # print("shape", logits.shape)
            # 只关心最后一个时间步(time step)
            logits = logits[:, -1, :] # become (B, C)
            # 应用Softmax，得到概率分布 只应用最后一维
            probs = F.softmax(logits, dim=-1) # (B, C)
            # 从分布中取样
            index_next = torch.multinomial(probs, num_samples=1) # (B, 1)
            # 增加取样的索引到当前序列
            index = torch.cat((index, index_next), dim=1)  # (B, T+1)
        return index
    
model = GPTLanguageModel(vocab_size)
print("loading parms")
model.load_state_dict(torch.load('./checkpoints/model_500.pt'))
m = model.to(device)
print("loading done")

while True:
    prompt = input("Prompt:\n")
    context = torch.tensor(encode(prompt), dtype=torch.long, device=device)
    generated_chars = decode(m.generate(context.unsqueeze(0), max_new_tokens=500)[0].tolist())
    print(f"Completion:\n{generated_chars}")       
"""
GPT model (rewrite, a lot simpler)
Notable features:
- rotary embeddings (and no positional embeddings)
- QK norm
- untied weights for token embedding and lm_head
- relu^2 activation in MLP
- norm after token embedding
- no learnable params in rmsnorm
- no bias in linear layers
- Multi-Query Attention (MQA) support for more efficient inference
"""

import math
from functools import partial
from dataclasses import dataclass
from collections.abc import Iterable, Iterator

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from nanochat.common import get_dist_info
from nanochat.muon import Muon, DistMuon
from nanochat.adamw import DistAdamW


@dataclass
class GPTConfig:
    """GPT模型配置类

    包含GPT模型的所有超参数配置，用于定义模型的结构和大小。
    """
    sequence_len: int = 1024
    vocab_size: int = 50304
    n_layer: int = 12
    n_head: int = 6 # number of query heads
    n_kv_head: int = 6 # number of key/value heads (MQA)
    n_embd: int = 768


def norm(x: Tensor) -> Tensor:
    """Purely functional rmsnorm with no learnable params"""
    return F.rms_norm(x, (x.size(-1),))


def apply_rotary_emb(
    x: Tensor, cos: Tensor, sin: Tensor
) -> Tensor:
    """旋转位置编码"""
    assert x.ndim == 4  # multihead attention
    d = x.shape[3] // 2
    x1, x2 = x[..., :d], x[..., d:] # split up last time into two halves
    y1 = x1 * cos + x2 * sin # rotate pairs of dims
    y2 = x1 * (-sin) + x2 * cos
    out = torch.cat([y1, y2], 3) # re-assemble
    out = out.to(x.dtype) # ensure input/output dtypes match
    return out


def repeat_kv(x: Tensor, n_rep):
    """torch.repeat_interleave(x, dim=1, repeats=n_rep)"""
    if n_rep == 1:
        return x
    bs, n_kv_heads, slen, head_dim = x.shape
    return (
        x[:, :, None, :, :]
        .expand(bs, n_kv_heads, n_rep, slen, head_dim)
        .reshape(bs, n_kv_heads * n_rep, slen, head_dim)
    )


class CausalSelfAttention(nn.Module):
    """因果多头自注意力机制

    实现了GPT模型中的因果多头自注意力层，支持以下特性：
    - 多头注意力机制，将输入分割到多个头并行处理
    - 因果掩码，确保每个位置只能关注到之前的位置（自回归特性）
    - 旋转位置编码（Rotary Embeddings），提供相对位置信息
    - 键值缓存（KV Cache），加速推理过程
    - 多查询注意力（MQA），通过共享键值头减少内存使用和计算量

    该类是Transformer架构中的核心组件，负责捕捉序列中的长距离依赖关系。
    """

    def __init__(self, config: GPTConfig, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        self.n_head = config.n_head
        self.n_kv_head = config.n_kv_head
        self.n_embd = config.n_embd
        self.head_dim = self.n_embd // self.n_head

        assert self.n_embd % self.n_head == 0
        assert self.n_kv_head <= self.n_head and self.n_head % self.n_kv_head == 0

        self.c_q = nn.Linear(self.n_embd, self.n_head * self.head_dim, bias=False)
        self.c_k = nn.Linear(self.n_embd, self.n_kv_head * self.head_dim, bias=False)
        self.c_v = nn.Linear(self.n_embd, self.n_kv_head * self.head_dim, bias=False)
        self.c_proj = nn.Linear(self.n_embd, self.n_embd, bias=False)

    def forward(
        self,
        x: Tensor,
        cos_sin: tuple[Tensor, Tensor],
        kv_cache,
    ) -> Tensor:
        """
        前向传播函数

        参数:
            x: 输入张量，形状为 [batch_size, seq_len, n_embd]
            cos_sin: 余弦和正弦位置编码的元组
            kv_cache: 键值缓存，用于加速推理

        返回:
            输出张量
        """
        batch_size, seq_len, _ = x.size()

        # Project the input to get queries, keys, and values
        q: Tensor = self.c_q(x).view(batch_size, seq_len, self.n_head, self.head_dim)
        k: Tensor = self.c_k(x).view(batch_size, seq_len, self.n_kv_head, self.head_dim)
        v: Tensor = self.c_v(x).view(batch_size, seq_len, self.n_kv_head, self.head_dim)

        # Apply Rotary Embeddings to queries and keys to get relative positional encoding
        cos, sin = cos_sin
        q, k = apply_rotary_emb(q, cos, sin), apply_rotary_emb(
            k, cos, sin
        )  # QK rotary embedding
        q, k = norm(q), norm(k)  # QK norm
        # make head be batch dicm,
        # i.e. (batch_size, seq_len, num_heads, head_dim)
        #   -> (batch_size, num_heads, seq_len, head_dim)
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)

        # Apply KV cache: insert current k,v into cache, get the full view so far
        if kv_cache is not None:
            k, v = kv_cache.insert_kv(self.layer_idx, k, v)
        num_queries = q.size(2)  # number of queries in this forward pass
        num_keys = k.size(
            2
        )  # number of keys/values in total (in the cache + current forward pass)

        # Apply MQA: replicate the key/value heads for each query head
        nrep = self.n_head // self.n_kv_head
        k, v = repeat_kv(k, nrep), repeat_kv(v, nrep)

        # Attention: queries attend to keys/values autoregressively. A few cases to handle:
        if kv_cache is None or num_queries == num_keys:
            # During training (no KV cache), attend as usual with causal attention
            # And even if there is KV cache, we can still use this simple version when Tq == Tk
            y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        elif num_queries == 1:
            # During inference but with a single query in this forward pass:
            # The query has to attend to all the keys/values in the cache
            y = F.scaled_dot_product_attention(q, k, v, is_causal=False)
        else:
            # During inference AND we have a chunk of queries in this forward pass:
            # First, each query attends to all the cached keys/values (i.e. full prefix)
            attn_mask = torch.zeros(
                (num_queries, num_keys), dtype=torch.bool, device=q.device
            )  # True = keep, False = mask
            prefix_len = num_keys - num_queries
            if prefix_len > 0:  # can't be negative but could be zero
                attn_mask[:, :prefix_len] = True
            # Then, causal attention within this chunk
            attn_mask[:, prefix_len:] = torch.tril(
                torch.ones(
                    (num_queries, num_queries), dtype=torch.bool, device=q.device
                )
            )
            y = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)

        # Re-assemble the heads side by side and project back to residual stream
        y = y.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)
        y = self.c_proj(y)
        return y


class MLP(nn.Module):
    """Multi-Layer Perceptron"""

    def __init__(self, config: GPTConfig):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=False)
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        """
        MLP前向传播函数

        Args:
            x: 输入张量，形状为 [batch_size, seq_len, n_embd]

        Returns:
            输出张量，形状为 [batch_size, seq_len, n_embd]
        """
        x = self.c_fc(x)
        x = F.relu(x).square()
        x = self.c_proj(x)
        return x


class Block(nn.Module):
    """Transformer 块，包含自注意力机制和前馈神经网络

    这是 GPT 模型中的基本构建块，每个块包含：
    1. 因果自注意力层（CausalSelfAttention）
    2. 多层感知机（MLP）

    使用残差连接和层归一化来稳定训练过程。
    """

    def __init__(self, config: GPTConfig, layer_idx: int):
        """
        初始化 Transformer 块

        Args:
            config: 模型配置对象，包含模型超参数
            layer_idx: 当前块的层索引，用于位置编码等
        """
        super().__init__()
        self.attn = CausalSelfAttention(config, layer_idx)
        self.mlp = MLP(config)

    def forward(self, x, cos_sin: tuple[Tensor, Tensor], kv_cache) -> Tensor:
        """
        前向传播

        Args:
            x: 输入张量，形状为 [batch_size, seq_len, hidden_size]
            cos_sin: 余弦和正弦位置编码元组，用于旋转位置编码
            kv_cache: 键值缓存，用于加速推理

        Returns:
            输出张量，形状与输入相同 [batch_size, seq_len, hidden_size]
        """
        x = x + self.attn(norm(x), cos_sin, kv_cache)
        x = x + self.mlp(norm(x))
        return x


class GPT(nn.Module):
    """GPT (Generative Pre-trained Transformer) 模型

    实现了一个基于 Transformer 的生成式预训练语言模型，具有以下特点：
    - 使用旋转位置编码（rotary embeddings）替代传统位置编码
    - QK 归一化（QK norm）
    - 词嵌入和输出层权重不共享（untied weights）
    - MLP 中使用 ReLU^2 激活函数
    - 词嵌入后进行归一化
    - 线性层无偏置
    - 支持多查询注意力（Multi-Query Attention, MQA）以提高推理效率
    """

    def __init__(self, config: GPTConfig):
        """初始化 GPT 模型"""
        super().__init__()

        self.config = config

        self.wte = nn.Embedding(
            config.vocab_size, config.n_embd
        )  # Word Token Embeddings
        self.trf_blocks: Iterable[Block] = nn.ModuleList(
            [Block(config, layer_idx) for layer_idx in range(config.n_layer)]
        )
        self.lm_head = nn.Linear(
            config.n_embd, config.vocab_size, bias=False
        )  # Language Model Head

        # To support meta device initialization, we init the rotary embeddings here, but it's fake
        # As for rotary_seq_len, these rotary embeddings are pretty small/cheap in memory,
        # so let's just over-compute them, but assert fail if we ever reach that amount.
        # In the future we can dynamically grow the cache, for now it's fine.
        self.rotary_seq_len = (
            config.sequence_len * 10
        )  # 10X over-compute should be enough, TODO make nicer?
        head_dim = config.n_embd // config.n_head
        cos, sin = self._precompute_rotary_embeddings(self.rotary_seq_len, head_dim)
        self.register_buffer(
            "cos", cos, persistent=False
        )  # persistent=False means it's not saved to the checkpoint
        self.register_buffer("sin", sin, persistent=False)

        # Cast the embeddings from fp32 to bf16: optim can tolerate it and it saves memory:
        # both in the model and the activations
        self.wte.to(dtype=torch.bfloat16)

    def init_weights(self):
        """初始化模型权重

        对模型的所有权重进行初始化，包括：
        - 应用 _init_weights 方法初始化所有模块的权重
        - 将语言模型头的权重置零
        - 将所有块中的 MLP 和注意力层的输出投影权重置零
        - 初始化旋转位置编码
        """
        self.apply(self._init_weights)

        # zero out classifier weights
        torch.nn.init.zeros_(self.lm_head.weight)

        # zero out c_proj weights in all blocks
        for block in self.trf_blocks:
            torch.nn.init.zeros_(block.mlp.c_proj.weight)
            torch.nn.init.zeros_(block.attn.c_proj.weight)

        # init the rotary embeddings
        head_dim = self.config.n_embd // self.config.n_head
        cos, sin = self._precompute_rotary_embeddings(self.rotary_seq_len, head_dim)
        self.cos, self.sin = cos, sin

    def _init_weights(self, module: nn.Module):
        """初始化单个模块的权重

        Args:
            module: 需要初始化权重的神经网络模块

        根据模块类型应用不同的初始化策略：
        - Linear 层：使用正态分布初始化权重，偏置置零
        - Embedding 层：使用标准正态分布初始化权重
        """
        if isinstance(module, nn.Linear):
            # https://arxiv.org/pdf/2310.17813
            fan_out = module.weight.size(0)
            fan_in = module.weight.size(1)
            std = 1.0 / math.sqrt(fan_in) * min(1.0, math.sqrt(fan_out / fan_in))
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=1.0)

    # TODO: bump base theta more, e.g. 100K is more common more recently
    def _precompute_rotary_embeddings(
        self, seq_len: int, head_dim: int, base: int = 10000, device: str = None
    ) -> tuple[Tensor, Tensor]:
        # autodetect the device from model embeddings
        if device is None:
            device = self.wte.weight.device
        # stride the channels
        channel_range = torch.arange(0, head_dim, 2, dtype=torch.float32, device=device)
        inv_freq = 1.0 / (base ** (channel_range / head_dim))
        # stride the time steps
        t = torch.arange(seq_len, dtype=torch.float32, device=device)
        # calculate the rotation frequencies at each (time, channel) pair
        freqs = torch.outer(t, inv_freq)
        cos, sin = freqs.cos(), freqs.sin()
        cos, sin = cos.bfloat16(), sin.bfloat16()  # keep them in bfloat16
        cos, sin = (
            cos[None, :, None, :],
            sin[None, :, None, :],
        )  # add batch and head dims for later broadcasting
        return cos, sin

    def get_device(self):
        """获取模型所在的设备

        Returns:
            torch.device: 模型参数所在的设备（CPU 或 CUDA）
        """
        return self.wte.weight.device

    def estimate_flops(self):
        """Return the estimated FLOPs per token for the model.
        Ref: https://arxiv.org/abs/2204.02311"""
        nparams = sum(p.numel() for p in self.parameters())
        nparams_embedding = self.wte.weight.numel()
        l, h, q, t = (
            self.config.n_layer,
            self.config.n_head,
            self.config.n_embd // self.config.n_head,
            self.config.sequence_len,
        )
        num_flops_per_token = 6 * (nparams - nparams_embedding) + 12 * l * h * q * t
        return num_flops_per_token

    def setup_optimizers(
        self, unembedding_lr=0.004, embedding_lr=0.2, matrix_lr=0.02, weight_decay=0.0
    ) -> list[torch.optim.Optimizer]:
        """设置模型优化器

        将模型参数分为三组（矩阵参数、嵌入参数、语言模型头参数），
        并为每组参数使用不同的优化器和学习率：
        - 矩阵参数（Transformer 块中的参数）：使用 Muon 优化器
        - 嵌入参数和语言模型头参数：使用 AdamW 优化器

        Args:
            unembedding_lr: 语言模型头参数的学习率
            embedding_lr: 嵌入参数的学习率
            matrix_lr: 矩阵参数的学习率
            weight_decay: 权重衰减系数

        Returns:
            list[torch.optim.Optimizer]: 优化器列表，包含 AdamW 和 Muon 优化器
        """
        model_dim = self.config.n_embd
        ddp, rank, local_rank, world_size = get_dist_info()

        # Separate out all parameters into 3 groups (matrix, embedding, lm_head)
        matrix_params = list(self.trf_blocks.parameters())
        embedding_params = list(self.wte.parameters())
        lm_head_params = list(self.lm_head.parameters())
        assert len(list(self.parameters())) == len(matrix_params) + len(
            embedding_params
        ) + len(lm_head_params)

        # Create the AdamW optimizer for the embedding and lm_head
        # Scale the LR for the AdamW parameters by ∝1/√dmodel (having tuned the LRs for 768 dim model)
        dmodel_lr_scale = (model_dim / 768) ** -0.5
        if rank == 0:
            hint_msg = (
                "Scaling the LR for the AdamW parameters ∝1/√"
                + f"({model_dim}/768) = {dmodel_lr_scale:.6f}"
            )
            print(hint_msg)
        adam_groups = [
            dict(params=lm_head_params, lr=unembedding_lr * dmodel_lr_scale),
            dict(params=embedding_params, lr=embedding_lr * dmodel_lr_scale),
        ]
        adamw_kwargs = dict(betas=(0.8, 0.95), eps=1e-10, weight_decay=weight_decay)
        AdamWFactory = DistAdamW if ddp else partial(torch.optim.AdamW, fused=True)
        adamw_optimizer = AdamWFactory(adam_groups, **adamw_kwargs)
        # Create the Muon optimizer for the linear layers
        muon_kwargs = dict(lr=matrix_lr, momentum=0.95)
        MuonFactory = DistMuon if ddp else Muon
        muon_optimizer = MuonFactory(matrix_params, **muon_kwargs)

        # Combine them the two optimizers into one list
        optimizers: list[torch.optim.Optimizer] = [adamw_optimizer, muon_optimizer]
        for opt in optimizers:
            for group in opt.param_groups:
                group["initial_lr"] = group["lr"]
        return optimizers

    def forward(
        self,
        idx: Tensor,
        targets: Tensor = None,
        kv_cache = None,
        loss_reduction: str = "mean",
    ) -> Tensor:
        """GPT 模型的前向传播

        Args:
            idx: 输入的 token 索引张量，形状为 (B, T)
            targets: 目标 token 索引张量，形状为 (B, T)，如果为 None 则返回 logits
            kv_cache: KV 缓存对象，用于加速推理
            loss_reduction: 损失缩减方式，可以是 "mean" 或 "sum"

        Returns:
            Tensor: 如果 targets 为 None，返回 logits 张量；否则返回损失值
        """
        _, seq_len = idx.size()

        # Grab the rotary embeddings for the current sequence length
        # (they are of shape (1, seq_len, 1, head_dim))
        assert_msg = (
            "Sequence length grew beyond the rotary embeddings cache: "
            + f"{seq_len} > {self.cos.size(1)}"
        )
        assert seq_len <= self.cos.size(1), assert_msg
        assert_msg = (
            "Rotary embeddings and idx are on different devices:"
            + f" {idx.device} != {self.cos.device}"
        )
        assert idx.device == self.cos.device, assert_msg
        assert self.cos.dtype == torch.bfloat16, "Rotary embeddings must be in bfloat16"

        # if kv cache exists, we need to offset the rotary embeddings to the current position in the cache
        T0 = 0 if kv_cache is None else kv_cache.get_pos()
        cos_sin = (
            self.cos[:, T0 : T0 + seq_len],
            self.sin[:, T0 : T0 + seq_len],
        )  # truncate cache to current sequence length

        # Forward the trunk of the Transformer
        x = self.wte(idx)
        x = norm(x)
        for block in self.trf_blocks:
            x = block(x, cos_sin, kv_cache)
        x = norm(x)

        # Forward the lm_head (compute logits)
        softcap = 15
        if targets is None:
            # training mode: compute and return the loss
            # TODO: experiment with Liger Kernels / chunked cross-entropy etc.
            logits = self.lm_head(x)
            logits = softcap * torch.tanh(logits / softcap)  # logits softcap
            logits = logits.float()  # use tf32/fp32 for logits
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                ignore_index=-1,
                reduction=loss_reduction,
            )
            return loss

        # inference mode: compute and return the logits
        logits = self.lm_head(x)
        logits = softcap * torch.tanh(logits / softcap)  # logits softcap
        return logits

    @torch.inference_mode()
    def generate(
        self,
        tokens: list[int],
        max_tokens: int,
        temperature=1.0,
        top_k: int = None,
        seed=42,
    ) -> Iterator[int]:
        """
        Naive autoregressive streaming inference.
        To make it super simple, let's assume:
        - batch size is 1
        - ids and the yielded tokens are simple Python lists and ints
        """
        assert isinstance(tokens, list)
        device = self.get_device()
        rng = None
        if temperature > 0:
            rng = torch.Generator(device=device)
            rng.manual_seed(seed)

        ids = torch.tensor([tokens], dtype=torch.long, device=device)  # (1, seq_len)
        for _ in range(max_tokens):
            logits = self.forward(ids)  # (1, seq_len, vocab_size)
            logits = logits[:, -1, :]  # (1, vocab_size)

            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))  # (1, top_k)
                logits[logits < v[:, [-1]]] = -float("Inf")

            if temperature > 0:
                logits = logits / temperature
                probs = F.softmax(logits, dim=-1)
                next_ids = torch.multinomial(probs, num_samples=1, generator=rng)
            else:
                next_ids = torch.argmax(logits, dim=-1, keepdim=True)

            ids = torch.cat((ids, next_ids), dim=1)
            token = next_ids.item()
            yield token

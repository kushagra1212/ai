#!/usr/bin/env python3
"""
====================================================================================
PROGRAM 38: HETEROGENEOUS 4-LAYER CHAT BRAIN (PYTHON SIDE-BY-SIDE)
====================================================================================
Companion script to 38_heterogeneous_chat_brain.cpp

Key Innovations Implemented:
1. Layer 1: Local Window Causal Filter (W=3, Linear O(T) compute)
2. Layer 2: Two-Tier Selective Memory (Working Memory + 8 Permanent Fact Slots)
3. Layer 3: Shared-KV Multi-Query Attention (MQA) with cross-attention to Memory
4. Layer 4: Deep Expansion MLP with Leaky ReLU (alpha = 0.02)
5. Pure 32-bit Floating-Point (FP32) math throughout
====================================================================================
"""

import math
import os
import random
import time
from typing import List, Tuple, Dict

def sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))

def leaky_relu(x: float, alpha: float = 0.02) -> float:
    return x if x > 0.0 else alpha * x

def softmax(logits: List[float]) -> List[float]:
    max_val = max(logits)
    exp_l = [math.exp(x - max_val) for x in logits]
    sum_exp = sum(exp_l)
    inv_sum = 1.0 / (sum_exp + 1e-12)
    return [e * inv_sum for e in exp_l]

class HeterogeneousBrainPy:
    def __init__(self, vocab_size: int, embed_dim: int = 64, num_heads: int = 4, context_len: int = 32, seed: int = 42):
        self.V = vocab_size
        self.D = embed_dim
        self.H = num_heads
        self.d_k = embed_dim // num_heads # 16
        self.D_ff = 4 * embed_dim          # 256
        self.max_T = context_len
        self.num_slots = 8                 # Brain 2 permanent memory slots
        self.scale = 1.0 / math.sqrt(self.d_k)
        self.leaky_alpha = 0.02

        random.seed(seed)
        def alloc_2d(r, c, std):
            return [[random.gauss(0.0, std) for _ in range(c)] for _ in range(r)]
        def alloc_3d(h, r, c, std):
            return [[[random.gauss(0.0, std) for _ in range(c)] for _ in range(r)] for _ in range(h)]

        # Layer 0: Embedding & Position
        self.E = alloc_2d(self.V, self.D, 1.0 / math.sqrt(self.D))
        self.P = alloc_2d(self.max_T, self.D, 1.0 / math.sqrt(self.D))

        # Layer 1: Local Window Filter (W = 3)
        self.W_local = alloc_2d(3, self.D, 0.1)

        # Layer 2: Two-Tier Selective Memory Gating
        self.W_gate = [random.gauss(0.0, 0.1) for _ in range(self.D)]
        self.b_gate = 0.0

        # Layer 3: Shared-KV Multi-Query Attention (MQA)
        self.Wq = alloc_3d(self.H, self.D, self.d_k, math.sqrt(2.0 / (self.D + self.d_k)))
        self.Wk_shared = alloc_2d(self.D, self.d_k, math.sqrt(2.0 / (self.D + self.d_k)))
        self.Wv_shared = alloc_2d(self.D, self.d_k, math.sqrt(2.0 / (self.D + self.d_k)))
        self.Wo = alloc_2d(self.D, self.D, math.sqrt(1.0 / self.D))

        # Layer 4: Expansion MLP (FFN)
        self.Wf1 = alloc_2d(self.D, self.D_ff, math.sqrt(2.0 / (self.D + self.D_ff)))
        self.Wf2 = alloc_2d(self.D_ff, self.D, math.sqrt(2.0 / (self.D_ff + self.D)))

        # Output Head
        self.W_out = alloc_2d(self.D, self.V, math.sqrt(2.0 / (self.D + self.V)))

    def forward(self, tokens: List[int]) -> List[List[float]]:
        T = min(len(tokens), self.max_T)

        # 0. Embeddings + Position
        H0 = [[self.E[tokens[t]][d] + self.P[t][d] for d in range(self.D)] for t in range(T)]

        # Layer 1: Local Window Filter (W=3)
        H1 = []
        for t in range(T):
            row = []
            for d in range(self.D):
                val = sum(self.W_local[k][d] * H0[t - k][d] for k in range(min(2, t) + 1))
                row.append(H0[t][d] + leaky_relu(val, self.leaky_alpha))
            H1.append(row)

        # Layer 2: Two-Tier Selective Memory
        perm_slots = [[0.0] * self.D for _ in range(self.num_slots)]
        curr_working = [0.0] * self.D
        H2 = []
        for t in range(T):
            g_logit = self.b_gate + sum(H1[t][d] * self.W_gate[d] for d in range(self.D))
            g = sigmoid(g_logit)

            # Working memory update
            decay = 1.0 - (g * 0.25)
            for d in range(self.D):
                curr_working[d] = decay * curr_working[d] + g * H1[t][d]

            # Permanent slot update
            if g > 0.65:
                slot_idx = t % self.num_slots
                perm_slots[slot_idx] = list(H1[t])

            # Output with residual highway bypass
            row = [H1[t][d] + (0.5 * g * curr_working[d]) for d in range(self.D)]
            H2.append(row)

        # Layer 3: Shared-KV Multi-Query Attention (MQA)
        total_ctx = T + self.num_slots
        full_ctx = [list(r) for r in H2] + [list(r) for r in perm_slots]

        # 1 Shared Key and 1 Shared Value across all context!
        K_shared = [[sum(full_ctx[j][d] * self.Wk_shared[d][k] for d in range(self.D)) for k in range(self.d_k)] for j in range(total_ctx)]
        V_shared = [[sum(full_ctx[j][d] * self.Wv_shared[d][k] for d in range(self.D)) for k in range(self.d_k)] for j in range(total_ctx)]

        Concat = [[0.0] * self.D for _ in range(T)]
        for h in range(self.H):
            Q_h = [[sum(H2[t][d] * self.Wq[h][d][k] for d in range(self.D)) for k in range(self.d_k)] for t in range(T)]
            for t in range(T):
                scores = []
                for j in range(total_ctx):
                    if j < T and j > t:
                        scores.append(-1e9) # Causal mask
                    else:
                        scores.append(sum(Q_h[t][k] * K_shared[j][k] for k in range(self.d_k)) * self.scale)
                attn = softmax(scores)
                for k in range(self.d_k):
                    Concat[t][h * self.d_k + k] = sum(attn[j] * V_shared[j][k] for j in range(total_ctx) if attn[j] > 0)

        # Output projection Wo + Residual Highway
        H3 = [[H2[t][d] + sum(Concat[t][k] * self.Wo[k][d] for k in range(self.D)) for d in range(self.D)] for t in range(T)]

        # Layer 4: Expansion MLP (64 -> 256 -> 64) with Leaky ReLU
        H4 = []
        for t in range(T):
            F1 = [leaky_relu(sum(H3[t][d] * self.Wf1[d][f] for d in range(self.D)), self.leaky_alpha) for f in range(self.D_ff)]
            row = [H3[t][d] + sum(F1[f] * self.Wf2[f][d] for f in range(self.D_ff)) for d in range(self.D)]
            H4.append(row)

        # Output Head
        probs = []
        for t in range(T):
            logits = [sum(H4[t][d] * self.W_out[d][v] for d in range(self.D)) for v in range(self.V)]
            probs.append(softmax(logits))

        return probs

if __name__ == "__main__":
    print("=" * 80)
    print(" PROGRAM 38: HETEROGENEOUS CHAT BRAIN (PYTHON COMPANION)")
    print("=" * 80)

    model = HeterogeneousBrainPy(vocab_size=4054, embed_dim=64, num_heads=4, context_len=32)

    emb_p = (4054 * 64) + (32 * 64)
    l1_p = 3 * 64
    l2_p = 64 + 1
    l3_p = (4 * 64 * 16) + (2 * 64 * 16) + (64 * 64)
    l4_p = (64 * 256) + (256 * 64)
    out_p = 64 * 4054
    total_p = emb_p + l1_p + l2_p + l3_p + l4_p + out_p

    print(f"[+] Total Parameters: {total_p:,} ({total_p * 4 / 1024:.1f} KB in FP32)")
    print("    - Layer 1 (Local Filter):", l1_p)
    print("    - Layer 2 (Two-Tier Memory):", l2_p)
    print("    - Layer 3 (Shared-KV MQA):", l3_p)
    print("    - Layer 4 (Expansion FFN):", l4_p)
    print("    - Head & Embeddings:", emb_p + out_p)
    print("\n[+] Testing Forward Pass with dummy token sequence...")
    dummy_input = [1, 5, 23, 89, 44]
    probs = model.forward(dummy_input)
    print(f"[+] Forward pass successful! Output sequence length: {len(probs)}, Vocab: {len(probs[0])}")

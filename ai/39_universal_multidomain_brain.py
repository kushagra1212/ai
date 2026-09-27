#!/usr/bin/env python3
"""
====================================================================================
PROGRAM 39: UNIVERSAL MULTI-DOMAIN BRAIN (PYTHON COMPANION)
====================================================================================
Companion script to 39_universal_multidomain_brain.cpp

Key Innovations:
1. Scaled Architecture (D = 128, H = 8 Heads, D_ff = 512, Context = 64)
2. Layer 1: Local Window Filter (W = 3)
3. Layer 2: Content-Addressable Selective Memory (Cosine similarity routing across 8 slots)
4. Layer 3: Shared-KV Multi-Query Attention (MQA) with cross-attention to 8 slots
5. Layer 4: Deep Expansion MLP with Leaky ReLU (alpha = 0.02)
6. Pure FP32 representation (1.85M parameters, ~7.4 MB)
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

class UniversalBrainPy:
    def __init__(self, vocab_size: int, embed_dim: int = 128, num_heads: int = 8, context_len: int = 64, seed: int = 42):
        self.V = vocab_size
        self.D = embed_dim
        self.H = num_heads
        self.d_k = embed_dim // num_heads # 16
        self.D_ff = 4 * embed_dim          # 512
        self.max_T = context_len
        self.num_slots = 8
        self.scale = 1.0 / math.sqrt(self.d_k)
        self.leaky_alpha = 0.02

        random.seed(seed)
        def alloc_2d(r, c, std):
            return [[random.gauss(0.0, std) for _ in range(c)] for _ in range(r)]
        def alloc_3d(h, r, c, std):
            return [[[random.gauss(0.0, std) for _ in range(c)] for _ in range(r)] for _ in range(h)]

        self.E = alloc_2d(self.V, self.D, 1.0 / math.sqrt(self.D))
        self.P = alloc_2d(self.max_T, self.D, 1.0 / math.sqrt(self.D))

        # Layer 1: Local Window Filter (W = 3)
        self.W_local = alloc_2d(3, self.D, 0.05)

        # Layer 2: Content-Addressable Memory
        self.W_gate = [random.gauss(0.0, 0.05) for _ in range(self.D)]
        self.b_gate = 0.0

        # Layer 3: Shared-KV MQA (8 Q heads, 1 K, 1 V)
        self.Wq = alloc_3d(self.H, self.D, self.d_k, math.sqrt(2.0 / (self.D + self.d_k)))
        self.Wk_shared = alloc_2d(self.D, self.d_k, math.sqrt(2.0 / (self.D + self.d_k)))
        self.Wv_shared = alloc_2d(self.D, self.d_k, math.sqrt(2.0 / (self.D + self.d_k)))
        self.Wo = alloc_2d(self.D, self.D, math.sqrt(1.0 / self.D))

        # Layer 4: Expansion MLP (128 -> 512 -> 128)
        self.Wf1 = alloc_2d(self.D, self.D_ff, math.sqrt(2.0 / (self.D + self.D_ff)))
        self.Wf2 = alloc_2d(self.D_ff, self.D, math.sqrt(2.0 / (self.D_ff + self.D)))

        # Output Head
        self.W_out = alloc_2d(self.D, self.V, math.sqrt(2.0 / (self.D + self.V)))

    def forward(self, tokens: List[int]) -> List[List[float]]:
        T = min(len(tokens), self.max_T)

        # 0. Embeddings + Position
        H0 = [[self.E[tokens[t]][d] + self.P[t][d] for d in range(self.D)] for t in range(T)]

        # Layer 1: Local Window Filter (W = 3)
        H1 = []
        for t in range(T):
            row = []
            for d in range(self.D):
                val = sum(self.W_local[k][d] * H0[t - k][d] for k in range(min(2, t) + 1))
                row.append(H0[t][d] + leaky_relu(val, self.leaky_alpha))
            H1.append(row)

        # Layer 2: Content-Addressable Memory
        perm_slots = [[0.0] * self.D for _ in range(self.num_slots)]
        curr_working = [0.0] * self.D
        slot_activity = [0] * self.num_slots
        H2 = []

        for t in range(T):
            g_logit = self.b_gate + sum(H1[t][d] * self.W_gate[d] for d in range(self.D))
            g = sigmoid(g_logit)

            # Working memory decay
            decay = 1.0 - (g * 0.20)
            for d in range(self.D):
                curr_working[d] = decay * curr_working[d] + g * H1[t][d]

            # Content-Addressable Write:
            if g > 0.60:
                max_sim = -1e9
                best_slot = -1
                for s in range(self.num_slots):
                    dot = sum(H1[t][d] * perm_slots[s][d] for d in range(self.D))
                    norm_s = sum(x * x for x in perm_slots[s])
                    norm_w = sum(x * x for x in H1[t])
                    sim = dot / (math.sqrt(norm_s * norm_w) + 1e-8)
                    if sim > max_sim:
                        max_sim = sim
                        best_slot = s

                target_slot = best_slot
                if max_sim < 0.40 or best_slot == -1:
                    target_slot = min(range(self.num_slots), key=lambda s: slot_activity[s])

                for d in range(self.D):
                    perm_slots[target_slot][d] = 0.5 * perm_slots[target_slot][d] + 0.5 * H1[t][d]
                slot_activity[target_slot] += 1

            row = [H1[t][d] + (0.5 * g * curr_working[d]) for d in range(self.D)]
            H2.append(row)

        # Layer 3: Shared-KV MQA Attention
        total_ctx = T + self.num_slots
        full_ctx = [list(r) for r in H2] + [list(r) for r in perm_slots]

        K_shared = [[sum(full_ctx[j][d] * self.Wk_shared[d][k] for d in range(self.D)) for k in range(self.d_k)] for j in range(total_ctx)]
        V_shared = [[sum(full_ctx[j][d] * self.Wv_shared[d][k] for d in range(self.D)) for k in range(self.d_k)] for j in range(total_ctx)]

        Concat = [[0.0] * self.D for _ in range(T)]
        for h in range(self.H):
            Q_h = [[sum(H2[t][d] * self.Wq[h][d][k] for d in range(self.D)) for k in range(self.d_k)] for t in range(T)]
            for t in range(T):
                scores = []
                for j in range(total_ctx):
                    if j < T and j > t:
                        scores.append(-1e9)
                    else:
                        scores.append(sum(Q_h[t][k] * K_shared[j][k] for k in range(self.d_k)) * self.scale)
                attn = softmax(scores)
                for k in range(self.d_k):
                    Concat[t][h * self.d_k + k] = sum(attn[j] * V_shared[j][k] for j in range(total_ctx) if attn[j] > 0)

        # Mixer Wo + Residual Highway
        H3 = [[H2[t][d] + sum(Concat[t][k] * self.Wo[k][d] for k in range(self.D)) for d in range(self.D)] for t in range(T)]

        # Layer 4: Expansion MLP (128 -> 512 -> 128)
        H4 = []
        for t in range(T):
            F1 = [leaky_relu(sum(H3[t][d] * self.Wf1[d][f] for d in range(self.D)), self.leaky_alpha) for f in range(self.D_ff)]
            row = [H3[t][d] + sum(F1[f] * self.Wf2[f][d] for f in range(self.D_ff)) for d in range(self.D)]
            H4.append(row)

        probs = []
        for t in range(T):
            logits = [sum(H4[t][d] * self.W_out[d][v] for d in range(self.D)) for v in range(self.V)]
            probs.append(softmax(logits))

        return probs

if __name__ == "__main__":
    print("=" * 80)
    print(" PROGRAM 39: UNIVERSAL MULTI-DOMAIN BRAIN (PYTHON COMPANION)")
    print("=" * 80)

    model = UniversalBrainPy(vocab_size=4500, embed_dim=128, num_heads=8, context_len=64)

    emb_p = (4500 * 128) + (64 * 128)
    l1_p = 3 * 128
    l2_p = 128 + 1
    l3_p = (8 * 128 * 16) + (2 * 128 * 16) + (128 * 128)
    l4_p = (128 * 512) + (512 * 128)
    out_p = 128 * 4500
    total_p = emb_p + l1_p + l2_p + l3_p + l4_p + out_p

    print(f"[+] Total Parameters: {total_p:,} ({total_p * 4 / 1024:.1f} KB in FP32)")
    print(f"[+] Memory Footprint: Fits inside 16 MB CPU L3 Cache!")
    print("    - Layer 1 (Local Filter):", l1_p)
    print("    - Layer 2 (Content-Addressable Memory):", l2_p)
    print("    - Layer 3 (Shared-KV MQA 8 Heads):", l3_p)
    print("    - Layer 4 (Expansion FFN 128->512):", l4_p)
    print("    - Head & Embeddings:", emb_p + out_p)

    dummy_input = [1, 10, 45, 120, 300]
    probs = model.forward(dummy_input)
    print(f"\n[+] Forward pass verified! Sequence length: {len(probs)}, Vocab: {len(probs[0])}")

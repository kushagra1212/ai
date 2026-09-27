#!/usr/bin/env python3
"""
====================================================================================
PROGRAM 35C: SCALING CAUSAL SELF-ATTENTION TO REAL DATASETS (PURE PYTHON)
====================================================================================
Side-by-Side Comparison with 35c_self_attention_real_data.cpp

Demonstrates:
1. Real-world dataset streaming & vocabulary construction from raw text files.
2. 100% First-Principles Causal Self-Attention in Pure Python (zero external dependencies).
3. Dynamic Learning Rate Schedule (Linear Warmup + Half-Cosine Decay).
4. Auto-regressive text generation from prompts with temperature & sampling.
====================================================================================
"""

import math
import os
import random
import time
from typing import List, Tuple, Dict

# ====================================================================================
# STEP 1: TOKENIZER & DATASET STREAMING
# ====================================================================================

class RealTokenizer:
    def __init__(self, target_vocab: int = 512):
        self.target_vocab = target_vocab
        self.word_to_id: Dict[str, int] = {
            "<PAD>": 0, "<UNK>": 1, "<BOS>": 2, "<EOS>": 3
        }
        self.id_to_word: List[str] = ["<PAD>", "<UNK>", "<BOS>", "<EOS>"]

    def build_vocab(self, filepath: str, max_chars: int = 1000000):
        print(f"[*] Scanning {filepath} to build frequency dictionary...")
        freq: Dict[str, int] = {}
        with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
            text = f.read(max_chars)

        # Normalize text and extract words/punctuation
        words = text.lower().replace(".", " . ").replace(",", " , ").replace("!", " ! ").replace("?", " ? ").split()
        for w in words:
            freq[w] = freq.get(w, 0) + 1

        sorted_words = sorted(freq.items(), key=lambda x: x[1], reverse=True)
        for w, _ in sorted_words[:self.target_vocab - 4]:
            self.word_to_id[w] = len(self.id_to_word)
            self.id_to_word.append(w)

        print(f"[+] Top-{len(self.id_to_word)} vocabulary compiled successfully.")

    def encode(self, text: str) -> List[int]:
        words = text.lower().replace(".", " . ").replace(",", " , ").replace("!", " ! ").replace("?", " ? ").split()
        return [self.word_to_id.get(w, 1) for w in words]

    def decode(self, token_ids: List[int]) -> str:
        words = [self.id_to_word[i] if 0 <= i < len(self.id_to_word) else "<UNK>" for i in token_ids]
        out = []
        for w in words:
            if w in [".", ",", "!", "?"] and out:
                out[-1] += w
            else:
                out.append(w)
        return " ".join(out)

# ====================================================================================
# STEP 2: MATHEMATICAL PRIMITIVES & DYNAMIC LEARNING RATE
# ====================================================================================

def softmax(logits: List[float]) -> List[float]:
    max_val = max(logits)
    exp_l = [math.exp(x - max_val) for x in logits]
    sum_exp = sum(exp_l)
    return [e / sum_exp for e in exp_l]

def get_dynamic_lr(step: int, total_steps: int, max_lr: float = 0.015, min_lr: float = 0.0005, warmup_pct: float = 0.05) -> float:
    warmup_steps = max(1, int(total_steps * warmup_pct))
    if step <= warmup_steps:
        return 1e-4 + (float(step) / warmup_steps) * (max_lr - 1e-4)
    else:
        decay_pct = float(step - warmup_steps) / float(total_steps - warmup_steps)
        return min_lr + 0.5 * (max_lr - min_lr) * (1.0 + math.cos(decay_pct * math.pi))

# ====================================================================================
# STEP 3: FIRST-PRINCIPLES MODEL
# ====================================================================================

class PurePythonSelfAttention:
    def __init__(self, vocab_size: int, embed_dim: int = 16, context_len: int = 16, seed: int = 42):
        self.V = vocab_size
        self.D = embed_dim
        self.max_T = context_len
        self.scale = 1.0 / math.sqrt(self.D)

        random.seed(seed)
        def alloc_2d(r, c, std):
            return [[random.gauss(0.0, std) for _ in range(c)] for _ in range(r)]

        self.E = alloc_2d(self.V, self.D, 1.0 / math.sqrt(self.D))
        self.P = alloc_2d(self.max_T, self.D, 1.0 / math.sqrt(self.D))
        self.Wq = alloc_2d(self.D, self.D, math.sqrt(2.0 / self.D))
        self.Wk = alloc_2d(self.D, self.D, math.sqrt(2.0 / self.D))
        self.Wv = alloc_2d(self.D, self.D, math.sqrt(2.0 / self.D))
        self.Wout = alloc_2d(self.D, self.V, math.sqrt(2.0 / self.D))

        # Adam momentum buffers
        def zero_2d(r, c): return [[0.0] * c for _ in range(r)]
        self.m_E, self.v_E = zero_2d(self.V, self.D), zero_2d(self.V, self.D)
        self.m_P, self.v_P = zero_2d(self.max_T, self.D), zero_2d(self.max_T, self.D)
        self.m_Wq, self.v_Wq = zero_2d(self.D, self.D), zero_2d(self.D, self.D)
        self.m_Wk, self.v_Wk = zero_2d(self.D, self.D), zero_2d(self.D, self.D)
        self.m_Wv, self.v_Wv = zero_2d(self.D, self.D), zero_2d(self.D, self.D)
        self.m_Wout, self.v_Wout = zero_2d(self.D, self.V), zero_2d(self.D, self.V)
        self.t = 0

    def step_adam(self, w, g, m, v, lr):
        beta1, beta2, eps = 0.9, 0.999, 1e-8
        m_new = beta1 * m + (1.0 - beta1) * g
        v_new = beta2 * v + (1.0 - beta2) * (g * g)
        m_hat = m_new / (1.0 - math.pow(beta1, self.t))
        v_hat = v_new / (1.0 - math.pow(beta2, self.t))
        w_new = w - lr * (m_hat / (math.sqrt(v_hat) + eps))
        return w_new, m_new, v_new

    def forward(self, tokens: List[int]):
        T = min(len(tokens), self.max_T)
        X = [[self.E[tokens[t]][d] + self.P[t][d] for d in range(self.D)] for t in range(T)]

        Q = [[sum(X[t][k] * self.Wq[k][d] for k in range(self.D)) for d in range(self.D)] for t in range(T)]
        K = [[sum(X[t][k] * self.Wk[k][d] for k in range(self.D)) for d in range(self.D)] for t in range(T)]
        V_mat = [[sum(X[t][k] * self.Wv[k][d] for k in range(self.D)) for d in range(self.D)] for t in range(T)]

        A = []
        C = []
        for t in range(T):
            scores = []
            for i in range(t + 1):
                dot = sum(Q[t][d] * K[i][d] for d in range(self.D))
                scores.append(dot * self.scale)
            weights = softmax(scores)
            A.append(weights)

            c_t = [sum(weights[i] * V_mat[i][d] for i in range(t + 1)) for d in range(self.D)]
            C.append(c_t)

        probs = []
        for t in range(T):
            logits = [sum(C[t][d] * self.Wout[d][v] for d in range(self.D)) for v in range(self.V)]
            probs.append(softmax(logits))

        return probs, A, (T, tokens, X, Q, K, V_mat, A, C, probs)

    def train_step(self, x_tokens: List[int], y_targets: List[int], lr: float) -> float:
        self.t += 1
        probs, _, cache = self.forward(x_tokens)
        T, tokens, X, Q, K, V_mat, A, C, _ = cache

        loss = 0.0
        dZ = [[0.0] * self.V for _ in range(T)]
        dC = [[0.0] * self.D for _ in range(T)]
        dWout = [[0.0] * self.V for _ in range(self.D)]

        for t in range(T):
            target = y_targets[t]
            p = max(probs[t][target], 1e-12)
            loss += -math.log(p)

            for v in range(self.V):
                dZ[t][v] = (probs[t][v] - (1.0 if v == target else 0.0)) / T
                for d in range(self.D):
                    dWout[d][v] += C[t][d] * dZ[t][v]
                    dC[t][d] += dZ[t][v] * self.Wout[d][v]

        # Update output weights
        for d in range(self.D):
            for v in range(self.V):
                self.Wout[d][v], self.m_Wout[d][v], self.v_Wout[d][v] = \
                    self.step_adam(self.Wout[d][v], dWout[d][v], self.m_Wout[d][v], self.v_Wout[d][v], lr)

        return loss / T

def sample_token_dynamic(probs: List[float], nucleus_p: float = 0.90) -> Tuple[int, int, float]:
    # RULE: Zero out <UNK>, <PAD>, <BOS>
    clean_probs = list(probs)
    if len(clean_probs) > 1: clean_probs[1] = 0.0  # <UNK> = 0!
    if len(clean_probs) > 0: clean_probs[0] = 0.0  # <PAD> = 0!
    if len(clean_probs) > 2: clean_probs[2] = 0.0  # <BOS> = 0!
    
    total = sum(clean_probs)
    if total > 1e-12:
        clean_probs = [p / total for p in clean_probs]

    # 1. Rank words
    ranked = sorted(enumerate(clean_probs), key=lambda x: x[1], reverse=True)
    top_prob = ranked[0][1]

    # 2. Dynamic Temperature:
    # High confidence -> Cools to 0.20; Low confidence -> Warms to 0.80
    dynamic_temp = 0.20 + 0.60 * (1.0 - min(1.0, top_prob ** 1.2))

    # 3. Dynamic K (Top-P / Nucleus):
    cum_prob = 0.0
    dynamic_pool = []
    for idx, p in ranked:
        dynamic_pool.append((idx, p))
        cum_prob += p
        if cum_prob >= nucleus_p or len(dynamic_pool) >= 30:
            break

    # 4. Scale by dynamic temp
    scaled = [p ** (1.0 / dynamic_temp) for _, p in dynamic_pool]
    scale_sum = sum(scaled)
    norm_probs = [s / scale_sum for s in scaled]

    # 5. Roll weighted dice
    indices = [idx for idx, _ in dynamic_pool]
    chosen = random.choices(indices, weights=norm_probs)[0]
    return chosen, len(dynamic_pool), dynamic_temp

# ====================================================================================
# STEP 4: TRAINING & GENERATION DEMO
# ====================================================================================

def main():
    dataset_path = "data/tinystories_5mb.txt"
    if not os.path.exists(dataset_path):
        dataset_path = "data/tinyshakespeare.txt"

    tok = RealTokenizer(target_vocab=1024)
    tok.build_vocab(dataset_path, max_chars=1000000)

    with open(dataset_path, "r", encoding="utf-8", errors="ignore") as f:
        stream_tokens = tok.encode(f.read(1000000))

    model = PurePythonSelfAttention(len(tok.id_to_word), embed_dim=16, context_len=16, seed=42)

    print("\n--- Training Pure Python Causal Self-Attention on Real Data ---")
    start = time.time()
    total_steps = 150

    for step in range(1, total_steps + 1):
        lr = get_dynamic_lr(step, total_steps, max_lr=0.015, min_lr=0.001)
        idx = random.randint(0, len(stream_tokens) - 16 - 2)
        x_tok = stream_tokens[idx:idx + 16]
        y_tok = stream_tokens[idx + 1:idx + 17]

        loss = model.train_step(x_tok, y_tok, lr)

        if step % 30 == 0 or step == 1:
            print(f">>> [Step {step:3d}/{total_steps}] Loss: {loss:.4f} | LR: {lr:.5f} | Time: {time.time() - start:.1f}s")

    # Generate sample output with Dynamic K and Dynamic Temp
    prompt = "once upon a time there was"
    tokens = tok.encode(prompt)
    for _ in range(12):
        probs, _, _ = model.forward(tokens[-16:])
        next_tok, k_used, t_used = sample_token_dynamic(probs[-1], nucleus_p=0.90)
        tokens.append(next_tok)

    print(f"\n[Generated Output (Zero UNK, Dynamic K/Temp)]:\n  \"{tok.decode(tokens)}\"")

if __name__ == "__main__":
    main()

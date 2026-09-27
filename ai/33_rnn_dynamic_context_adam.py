#!/usr/bin/env python3
"""
====================================================================================
PROGRAM 33: DYNAMIC-CONTEXT RECURRENT NEURAL NETWORK (RNN) & ADAM OPTIMIZER (PYTHON)
====================================================================================
Python side-by-side implementation of:
1. Dynamic arbitrary-length prompt input using a recurrent memory loop (Whh).
2. First-principles Adam optimizer (Dynamic Momentum with beta1=0.9, beta2=0.999).
3. Backpropagation Through Time (BPTT).
====================================================================================
"""

import math
import random
import time
from typing import List, Dict, Tuple

def sigmoid(z: float) -> float:
    if z > 20.0: return 1.0
    if z < -20.0: return 0.0
    return 1.0 / (1.0 + math.exp(-z))

def smooth_dynamic_leaky(z: float, alpha: float) -> float:
    s = sigmoid(z)
    return alpha * z + (1.0 - alpha) * z * s

def smooth_dynamic_leaky_derivative(z: float, alpha: float) -> float:
    s = sigmoid(z)
    return alpha + (1.0 - alpha) * (s + z * s * (1.0 - s))

def smooth_dynamic_leaky_d_alpha(z: float) -> float:
    s = sigmoid(z)
    return z * (1.0 - s)

def softmax(logits: List[float]) -> List[float]:
    max_val = max(logits)
    exp_l = [math.exp(x - max_val) for x in logits]
    sum_exp = sum(exp_l)
    return [e / sum_exp for e in exp_l]

class AdamOptimizer:
    def __init__(self, beta1: float = 0.9, beta2: float = 0.999, eps: float = 1e-8):
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.t = 0

    def step(self, weight: float, grad: float, m: float, v: float, lr: float) -> Tuple[float, float, float]:
        m = self.beta1 * m + (1.0 - self.beta1) * grad
        v = self.beta2 * v + (1.0 - self.beta2) * (grad * grad)
        m_hat = m / (1.0 - math.pow(self.beta1, self.t))
        v_hat = v / (1.0 - math.pow(self.beta2, self.t))
        new_weight = weight - lr * (m_hat / (math.sqrt(v_hat) + self.eps))
        return new_weight, m, v

class RecurrentLanguageModel:
    def __init__(self, vocab_size: int, coord_dim: int = 12, hidden_dim: int = 32):
        self.V = vocab_size
        self.D = coord_dim
        self.H = hidden_dim
        self.adam = AdamOptimizer()

        # Dynamic trainable leak rate alpha
        self.alpha = 0.03
        self.m_alpha = 0.0
        self.v_alpha = 0.0

        random.seed(42)
        scale_wxh = math.sqrt(2.0 / self.D)
        scale_whh = math.sqrt(2.0 / self.H)
        scale_why = math.sqrt(2.0 / self.H)

        # Word coordinates
        self.C = [[random.gauss(0.0, 0.1) for _ in range(self.D)] for _ in range(self.V)]
        self.mC = [[0.0] * self.D for _ in range(self.V)]
        self.vC = [[0.0] * self.D for _ in range(self.V)]

        # Wxh: Input to hidden
        self.Wxh = [[random.gauss(0.0, scale_wxh) for _ in range(self.H)] for _ in range(self.D)]
        self.mWxh = [[0.0] * self.H for _ in range(self.D)]
        self.vWxh = [[0.0] * self.H for _ in range(self.D)]

        # Whh: Recurrent memory loop
        self.Whh = [[random.gauss(0.0, scale_whh) for _ in range(self.H)] for _ in range(self.H)]
        self.mWhh = [[0.0] * self.H for _ in range(self.H)]
        self.vWhh = [[0.0] * self.H for _ in range(self.H)]

        self.Bh = [0.0] * self.H
        self.mBh = [0.0] * self.H
        self.vBh = [0.0] * self.H

        # Why: Hidden to output
        self.Why = [[random.gauss(0.0, scale_why) for _ in range(self.V)] for _ in range(self.H)]
        self.mWhy = [[0.0] * self.V for _ in range(self.H)]
        self.vWhy = [[0.0] * self.V for _ in range(self.H)]

        self.By = [0.0] * self.V
        self.mBy = [0.0] * self.V
        self.vBy = [0.0] * self.V

    def train_sentence(self, tokens: List[int], lr: float) -> Tuple[float, int, int]:
        T = len(tokens) - 1
        if T <= 0: return 0.0, 0, 0
        self.adam.t += 1

        h_pre = []
        h = []
        logits = []
        probs = []
        loss = 0.0
        correct = 0

        # Forward pass across time steps
        for t in range(T):
            in_word = tokens[t]
            target_word = tokens[t + 1]

            cur_h_pre = [self.Bh[j] for j in range(self.H)]
            for j in range(self.H):
                for d in range(self.D):
                    cur_h_pre[j] += self.C[in_word][d] * self.Wxh[d][j]
                if t > 0:
                    for prev_j in range(self.H):
                        cur_h_pre[j] += h[t - 1][prev_j] * self.Whh[prev_j][j]

            cur_h = [smooth_dynamic_leaky(val, self.alpha) for val in cur_h_pre]
            h_pre.append(cur_h_pre)
            h.append(cur_h)

            cur_logits = [self.By[k] + sum(cur_h[j] * self.Why[j][k] for j in range(self.H)) for k in range(self.V)]
            logits.append(cur_logits)

            cur_probs = softmax(cur_logits)
            probs.append(cur_probs)

            p_targ = max(cur_probs[target_word], 1e-12)
            loss += -math.log(p_targ)
            if cur_probs.index(max(cur_probs)) == target_word:
                correct += 1

        # Backpropagation Through Time (BPTT)
        dWxh = [[0.0] * self.H for _ in range(self.D)]
        dWhh = [[0.0] * self.H for _ in range(self.H)]
        dBh = [0.0] * self.H
        dWhy = [[0.0] * self.V for _ in range(self.H)]
        dBy = [0.0] * self.V
        d_alpha = 0.0
        dh_next = [0.0] * self.H

        for t in range(T - 1, -1, -1):
            in_word = tokens[t]
            targ_word = tokens[t + 1]

            dLogits = list(probs[t])
            dLogits[targ_word] -= 1.0

            for k in range(self.V):
                dBy[k] += dLogits[k]
                for j in range(self.H):
                    dWhy[j][k] += dLogits[k] * h[t][j]

            dh = [sum(dLogits[k] * self.Why[j][k] for k in range(self.V)) + dh_next[j] for j in range(self.H)]
            dh_pre_t = [0.0] * self.H
            for j in range(self.H):
                dh_pre_t[j] = dh[j] * smooth_dynamic_leaky_derivative(h_pre[t][j], self.alpha)
                dBh[j] += dh_pre_t[j]
                d_alpha += dh[j] * smooth_dynamic_leaky_d_alpha(h_pre[t][j])

            for d in range(self.D):
                d_coord = 0.0
                for j in range(self.H):
                    dWxh[d][j] += dh_pre_t[j] * self.C[in_word][d]
                    d_coord += dh_pre_t[j] * self.Wxh[d][j]
                self.C[in_word][d], self.mC[in_word][d], self.vC[in_word][d] = \
                    self.adam.step(self.C[in_word][d], d_coord, self.mC[in_word][d], self.vC[in_word][d], lr)

            dh_next = [0.0] * self.H
            if t > 0:
                for prev_j in range(self.H):
                    for j in range(self.H):
                        dWhh[prev_j][j] += dh_pre_t[j] * h[t - 1][prev_j]
                        dh_next[prev_j] += dh_pre_t[j] * self.Whh[prev_j][j]

        # Apply Adam step to matrices
        for d in range(self.D):
            for j in range(self.H):
                self.Wxh[d][j], self.mWxh[d][j], self.vWxh[d][j] = \
                    self.adam.step(self.Wxh[d][j], dWxh[d][j], self.mWxh[d][j], self.vWxh[d][j], lr)

        for i in range(self.H):
            for j in range(self.H):
                self.Whh[i][j], self.mWhh[i][j], self.vWhh[i][j] = \
                    self.adam.step(self.Whh[i][j], dWhh[i][j], self.mWhh[i][j], self.vWhh[i][j], lr)

        for j in range(self.H):
            self.Bh[j], self.mBh[j], self.vBh[j] = \
                self.adam.step(self.Bh[j], dBh[j], self.mBh[j], self.vBh[j], lr)

        for j in range(self.H):
            for k in range(self.V):
                self.Why[j][k], self.mWhy[j][k], self.vWhy[j][k] = \
                    self.adam.step(self.Why[j][k], dWhy[j][k], self.mWhy[j][k], self.vWhy[j][k], lr)

        for k in range(self.V):
            self.By[k], self.mBy[k], self.vBy[k] = \
                self.adam.step(self.By[k], dBy[k], self.mBy[k], self.vBy[k], lr)

        # Dynamic leak alpha Adam step
        self.alpha, self.m_alpha, self.v_alpha = \
            self.adam.step(self.alpha, d_alpha, self.m_alpha, self.v_alpha, lr * 0.05)
        self.alpha = max(0.005, min(0.250, self.alpha))

        return loss / T, correct, T

    def predict_next(self, prompt_tokens: List[int]) -> List[float]:
        if not prompt_tokens: return [1.0 / self.V] * self.V
        h_curr = [0.0] * self.H

        for wid in prompt_tokens:
            h_new = []
            for j in range(self.H):
                val = self.Bh[j] + sum(self.C[wid][d] * self.Wxh[d][j] for d in range(self.D)) + \
                      sum(h_curr[pj] * self.Whh[pj][j] for pj in range(self.H))
                h_new.append(smooth_dynamic_leaky(val, self.alpha))
            h_curr = h_new

        logits = [self.By[k] + sum(h_curr[j] * self.Why[j][k] for j in range(self.H)) for k in range(self.V)]
        return softmax(logits)

def main():
    sentences = [
        "the king sits on the golden throne .",
        "the queen sits on the golden throne .",
        "the king rules the royal castle .",
        "the queen rules the royal castle .",
        "the monkey eats a sweet apple .",
        "the wild wolf hunts in the deep forest ."
    ]
    vocab = []
    word_to_id = {}
    id_to_word = {}
    for s in sentences:
        for w in s.split():
            if w not in word_to_id:
                wid = len(vocab)
                word_to_id[w] = wid
                id_to_word[wid] = w
                vocab.append(w)

    model = RecurrentLanguageModel(len(vocab), coord_dim=8, hidden_dim=24)
    corpus = [[word_to_id[w] for w in s.split()] for s in sentences]

    print("\n--- Training RNN with Adam in Python ---")
    for epoch in range(1, 151):
        loss = 0.0
        corr = 0
        tot = 0
        for sent in corpus:
            l, c, t = model.train_sentence(sent, lr=0.01)
            loss += l
            corr += c
            tot += t
        if epoch % 50 == 0:
            print(f"Epoch {epoch:3d} | Loss: {loss/len(corpus):.4f} | Acc: {corr*100.0/tot:.1f}% | Dynamic Leak (alpha): {model.alpha:.4f}")

    print("\n--- Testing Arbitrary-Length Prompt Generation ---")
    prompt = ["the", "wild", "wolf"]
    tokens = [word_to_id[w] for w in prompt]
    print(f"Prompt: '{' '.join(prompt)}' (Length: {len(prompt)})")
    for _ in range(10):
        probs = model.predict_next(tokens)
        next_id = probs.index(max(probs))
        next_word = id_to_word[next_id]
        print(f"Predicted next: '{next_word}' ({max(probs)*100.0:.1f}%)")
        tokens.append(next_id)
        if next_word == ".": break

if __name__ == "__main__":
    main()

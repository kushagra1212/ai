#!/usr/bin/env python3
"""
====================================================================================
PROGRAM 35: THE CAUSAL SELF-ATTENTION ENGINE & FIRST-PRINCIPLES ADAM (PYTHON)
====================================================================================
Python side-by-side implementation of:
1. Causal Self-Attention (Weighted Prefix Sum using Q, K, V).
2. Learned Position Embeddings.
3. First-principles Adam optimizer.
4. Backpropagation through Attention Softmax and Projections.
5. Terminal Attention Spotlight Heatmap.
====================================================================================
"""

import math
import random
import time
from typing import List, Dict, Tuple

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

class SelfAttentionLanguageModel:
    def __init__(self, vocab_size: int, coord_dim: int = 16, max_seq_len: int = 32):
        self.V = vocab_size
        self.D = coord_dim
        self.max_len = max_seq_len
        self.adam = AdamOptimizer()

        random.seed(42)
        scale_proj = math.sqrt(2.0 / self.D)
        scale_out = math.sqrt(2.0 / self.D)

        def alloc_2d(r, c, init_fn):
            return [[init_fn() for _ in range(c)] for _ in range(r)]

        # Word coordinates C & Position coordinates P
        self.C = alloc_2d(self.V, self.D, lambda: random.gauss(0.0, 0.1))
        self.mC = [[0.0] * self.D for _ in range(self.V)]
        self.vC = [[0.0] * self.D for _ in range(self.V)]

        self.P = alloc_2d(self.max_len, self.D, lambda: random.gauss(0.0, 0.1))
        self.mP = [[0.0] * self.D for _ in range(self.max_len)]
        self.vP = [[0.0] * self.D for _ in range(self.max_len)]

        # Q, K, V Projections
        def init_proj():
            W = alloc_2d(self.D, self.D, lambda: random.gauss(0.0, scale_proj))
            mW = [[0.0] * self.D for _ in range(self.D)]
            vW = [[0.0] * self.D for _ in range(self.D)]
            return W, mW, vW

        self.Wq, self.mWq, self.vWq = init_proj()
        self.Wk, self.mWk, self.vWk = init_proj()
        self.Wv, self.mWv, self.vWv = init_proj()

        # Output judges
        self.Wy = alloc_2d(self.D, self.V, lambda: random.gauss(0.0, scale_out))
        self.mWy = [[0.0] * self.V for _ in range(self.D)]
        self.vWy = [[0.0] * self.V for _ in range(self.D)]
        self.By = [0.0] * self.V
        self.mBy = [0.0] * self.V
        self.vBy = [0.0] * self.V

    def train_sentence(self, tokens: List[int], lr: float) -> Tuple[float, int, int]:
        T = len(tokens) - 1
        if T <= 0 or T >= self.max_len: return 0.0, 0, 0
        self.adam.t += 1

        def alloc_d(r, c): return [[0.0] * c for _ in range(r)]

        # 1. Embeddings + Position: X[t] = C[token] + P[t]
        X = alloc_d(T, self.D)
        for t in range(T):
            for d in range(self.D):
                X[t][d] = self.C[tokens[t]][d] + self.P[t][d]

        # 2. Q, K, V
        Q = alloc_d(T, self.D)
        K = alloc_d(T, self.D)
        V_mat = alloc_d(T, self.D)

        for t in range(T):
            for j in range(self.D):
                Q[t][j] = sum(X[t][d] * self.Wq[d][j] for d in range(self.D))
                K[t][j] = sum(X[t][d] * self.Wk[d][j] for d in range(self.D))
                V_mat[t][j] = sum(X[t][d] * self.Wv[d][j] for d in range(self.D))

        # 3. Attention Weights A (Causal Mask: only i <= t)
        scale = math.sqrt(self.D)
        scores = alloc_d(T, T)
        A = alloc_d(T, T)

        for t in range(T):
            cur_scores = [sum(Q[t][d] * K[i][d] for d in range(self.D)) / scale for i in range(t + 1)]
            max_s = max(cur_scores)
            exp_s = [math.exp(s - max_s) for s in cur_scores]
            sum_e = sum(exp_s)
            for i in range(t + 1):
                A[t][i] = exp_s[i] / sum_e
                scores[t][i] = cur_scores[i]

        # 4. Weighted Prefix Sum (Context) & Residual Rep
        Context = alloc_d(T, self.D)
        Rep = alloc_d(T, self.D)
        for t in range(T):
            for d in range(self.D):
                Context[t][d] = sum(A[t][i] * V_mat[i][d] for i in range(t + 1))
                Rep[t][d] = X[t][d] + Context[t][d]

        # 5. Output Judges & Loss
        logits = alloc_d(T, self.V)
        probs = alloc_d(T, self.V)
        loss = 0.0
        correct = 0

        for t in range(T):
            targ = tokens[t + 1]
            cur_logits = [self.By[k] + sum(Rep[t][d] * self.Wy[d][k] for d in range(self.D)) for k in range(self.V)]
            logits[t] = cur_logits
            cur_probs = softmax(cur_logits)
            probs[t] = cur_probs

            p_targ = max(cur_probs[targ], 1e-12)
            loss += -math.log(p_targ)
            if cur_probs.index(max(cur_probs)) == targ:
                correct += 1

        # Backpropagation
        dWy = alloc_d(self.D, self.V)
        dBy = [0.0] * self.V
        dRep = alloc_d(T, self.D)

        for t in range(T):
            targ = tokens[t + 1]
            dLogits = list(probs[t])
            dLogits[targ] -= 1.0

            for k in range(self.V):
                dBy[k] += dLogits[k]
                for d in range(self.D):
                    dWy[d][k] += dLogits[k] * Rep[t][d]
                    dRep[t][d] += dLogits[k] * self.Wy[d][k]

        dContext = dRep
        dX = [[dRep[t][d] for d in range(self.D)] for t in range(T)]

        dV_mat = alloc_d(T, self.D)
        dA = alloc_d(T, T)

        for t in range(T):
            for i in range(t + 1):
                for d in range(self.D):
                    dV_mat[i][d] += dContext[t][d] * A[t][i]
                    dA[t][i] += dContext[t][d] * V_mat[i][d]

        dScores = alloc_d(T, T)
        for t in range(T):
            sum_A_dA = sum(A[t][i] * dA[t][i] for i in range(t + 1))
            for i in range(t + 1):
                dScores[t][i] = A[t][i] * (dA[t][i] - sum_A_dA)

        dQ = alloc_d(T, self.D)
        dK = alloc_d(T, self.D)

        for t in range(T):
            for i in range(t + 1):
                ds = dScores[t][i] / scale
                for d in range(self.D):
                    dQ[t][d] += ds * K[i][d]
                    dK[i][d] += ds * Q[t][d]

        dWq = alloc_d(self.D, self.D)
        dWk = alloc_d(self.D, self.D)
        dWv = alloc_d(self.D, self.D)

        for t in range(T):
            for d in range(self.D):
                for j in range(self.D):
                    dWq[d][j] += dQ[t][j] * X[t][d]
                    dWk[d][j] += dK[t][j] * X[t][d]
                    dWv[d][j] += dV_mat[t][j] * X[t][d]

                    dX[t][d] += dQ[t][j] * self.Wq[d][j] + dK[t][j] * self.Wk[d][j] + dV_mat[t][j] * self.Wv[d][j]

        # Update Embeddings & Coordinates
        for t in range(T):
            wid = tokens[t]
            for d in range(self.D):
                self.C[wid][d], self.mC[wid][d], self.vC[wid][d] = \
                    self.adam.step(self.C[wid][d], dX[t][d], self.mC[wid][d], self.vC[wid][d], lr)
                self.P[t][d], self.mP[t][d], self.vP[t][d] = \
                    self.adam.step(self.P[t][d], dX[t][d], self.mP[t][d], self.vP[t][d], lr)

        # Update Projections
        for r in range(self.D):
            for c in range(self.D):
                self.Wq[r][c], self.mWq[r][c], self.vWq[r][c] = self.adam.step(self.Wq[r][c], dWq[r][c], self.mWq[r][c], self.vWq[r][c], lr)
                self.Wk[r][c], self.mWk[r][c], self.vWk[r][c] = self.adam.step(self.Wk[r][c], dWk[r][c], self.mWk[r][c], self.vWk[r][c], lr)
                self.Wv[r][c], self.mWv[r][c], self.vWv[r][c] = self.adam.step(self.Wv[r][c], dWv[r][c], self.mWv[r][c], self.vWv[r][c], lr)

        for r in range(self.D):
            for c in range(self.V):
                self.Wy[r][c], self.mWy[r][c], self.vWy[r][c] = self.adam.step(self.Wy[r][c], dWy[r][c], self.mWy[r][c], self.vWy[r][c], lr)
        for k in range(self.V):
            self.By[k], self.mBy[k], self.vBy[k] = self.adam.step(self.By[k], dBy[k], self.mBy[k], self.vBy[k], lr)

        return loss / T, correct, T

    def predict_next(self, prompt_tokens: List[int]) -> Tuple[List[float], List[List[float]]]:
        T = len(prompt_tokens)
        if T == 0: return [1.0 / self.V] * self.V, []

        def alloc_d(r, c): return [[0.0] * c for _ in range(r)]
        X = alloc_d(T, self.D)
        for t in range(T):
            for d in range(self.D): X[t][d] = self.C[prompt_tokens[t]][d] + self.P[t][d]

        Q = alloc_d(T, self.D)
        K = alloc_d(T, self.D)
        V_mat = alloc_d(T, self.D)

        for t in range(T):
            for j in range(self.D):
                Q[t][j] = sum(X[t][d] * self.Wq[d][j] for d in range(self.D))
                K[t][j] = sum(X[t][d] * self.Wk[d][j] for d in range(self.D))
                V_mat[t][j] = sum(X[t][d] * self.Wv[d][j] for d in range(self.D))

        scale = math.sqrt(self.D)
        A = alloc_d(T, T)
        for t in range(T):
            cur_scores = [sum(Q[t][d] * K[i][d] for d in range(self.D)) / scale for i in range(t + 1)]
            max_s = max(cur_scores)
            exp_s = [math.exp(s - max_s) for s in cur_scores]
            sum_e = sum(exp_s)
            for i in range(t + 1): A[t][i] = exp_s[i] / sum_e

        Rep = alloc_d(T, self.D)
        for t in range(T):
            for d in range(self.D):
                context_d = sum(A[t][i] * V_mat[i][d] for i in range(t + 1))
                Rep[t][d] = X[t][d] + context_d

        last_t = T - 1
        logits = [self.By[k] + sum(Rep[last_t][d] * self.Wy[d][k] for d in range(self.D)) for k in range(self.V)]
        return softmax(logits), A

def main():
    sentences = [
        "the king who lived in the royal palace sits on the golden throne .",
        "the queen who lived in the royal palace sits on the golden throne .",
        "the monkey who climbed the tall green tree eats a sweet apple .",
        "the wild wolf that hunted across the deep forest runs in the night ."
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

    model = SelfAttentionLanguageModel(len(vocab), coord_dim=16, max_seq_len=32)
    corpus = [[word_to_id[w] for w in s.split()] for s in sentences]

    print("\n--- Training Self-Attention Engine in Python ---")
    for epoch in range(1, 151):
        loss = 0.0
        corr = 0
        tot = 0
        for sent in corpus:
            l, c, t = model.train_sentence(sent, lr=0.015)
            loss += l
            corr += c
            tot += t
        if epoch % 50 == 0:
            print(f"Epoch {epoch:3d} | Loss: {loss/len(corpus):.4f} | Acc: {corr*100.0/tot:.1f}%")

    prompt = ["the", "king", "who", "lived", "in", "the", "royal", "palace"]
    tokens = [word_to_id[w] for w in prompt]
    probs, A = model.predict_next(tokens)
    next_id = probs.index(max(probs))
    print(f"\nPrompt: '{' '.join(prompt)}'")
    print(f"Predicted next: '{id_to_word[next_id]}' ({max(probs)*100.0:.1f}%)")

    last_pos = len(prompt) - 1
    print(f"Attention Spotlight from '{prompt[last_pos]}' back to prefix:")
    for i in range(len(prompt)):
        pct = A[last_pos][i] * 100.0
        print(f"  [{prompt[i]:8s}] {pct:5.1f}% {'#' * int(pct / 5)}")

if __name__ == "__main__":
    main()

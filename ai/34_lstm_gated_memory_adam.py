#!/usr/bin/env python3
"""
====================================================================================
PROGRAM 34: GATED MEMORY (LSTM) & FIRST-PRINCIPLES ADAM OPTIMIZER (PYTHON)
====================================================================================
Python side-by-side implementation of:
1. Long Short-Term Memory (LSTM) with Conveyor Belt (C_t) and 3 Security Doors (f, i, o).
2. The Additive Highway preventing vanishing gradients over long sentences.
3. First-principles Adam optimizer per gate parameter.
4. Backpropagation Through Time (BPTT) for LSTM.
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

def dtanh(tanh_val: float) -> float:
    return 1.0 - tanh_val * tanh_val

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

class LSTMLanguageModel:
    def __init__(self, vocab_size: int, coord_dim: int = 12, hidden_dim: int = 32):
        self.V = vocab_size
        self.D = coord_dim
        self.H = hidden_dim
        self.adam = AdamOptimizer()

        random.seed(42)
        scale_x = math.sqrt(2.0 / self.D)
        scale_h = math.sqrt(2.0 / self.H)
        scale_y = math.sqrt(2.0 / self.H)

        # Word coordinates
        self.C = [[random.gauss(0.0, 0.1) for _ in range(self.D)] for _ in range(self.V)]
        self.mC = [[0.0] * self.D for _ in range(self.V)]
        self.vC = [[0.0] * self.D for _ in range(self.V)]

        # Helper to init gates
        def init_gate(b_init: float = 0.0):
            Wx = [[random.gauss(0.0, scale_x) for _ in range(self.H)] for _ in range(self.D)]
            mWx = [[0.0] * self.H for _ in range(self.D)]
            vWx = [[0.0] * self.H for _ in range(self.D)]

            Wh = [[random.gauss(0.0, scale_h) for _ in range(self.H)] for _ in range(self.H)]
            mWh = [[0.0] * self.H for _ in range(self.H)]
            vWh = [[0.0] * self.H for _ in range(self.H)]

            B = [b_init] * self.H
            mB = [0.0] * self.H
            vB = [0.0] * self.H

            return Wx, mWx, vWx, Wh, mWh, vWh, B, mB, vB

        # 1. Forget gate (Bf initialized to 1.0)
        self.Wxf, self.mWxf, self.vWxf, self.Whf, self.mWhf, self.vWhf, self.Bf, self.mBf, self.vBf = init_gate(1.0)
        # 2. Input gate
        self.Wxi, self.mWxi, self.vWxi, self.Whi, self.mWhi, self.vWhi, self.Bi, self.mBi, self.vBi = init_gate(0.0)
        # 3. Candidate note
        self.Wxc, self.mWxc, self.vWxc, self.Whc, self.mWhc, self.vWhc, self.Bc, self.mBc, self.vBc = init_gate(0.0)
        # 4. Output gate
        self.Wxo, self.mWxo, self.vWxo, self.Who, self.mWho, self.vWho, self.Bo, self.mBo, self.vBo = init_gate(0.0)

        # Output judges
        self.Why = [[random.gauss(0.0, scale_y) for _ in range(self.V)] for _ in range(self.H)]
        self.mWhy = [[0.0] * self.V for _ in range(self.H)]
        self.vWhy = [[0.0] * self.V for _ in range(self.H)]

        self.By = [0.0] * self.V
        self.mBy = [0.0] * self.V
        self.vBy = [0.0] * self.V

    def train_sentence(self, tokens: List[int], lr: float) -> Tuple[float, int, int]:
        T = len(tokens) - 1
        if T <= 0: return 0.0, 0, 0
        self.adam.t += 1

        f_gate, i_gate, c_cand, o_gate = [], [], [], []
        C_cell, tanh_C, h_state = [], [], []
        logits, probs = [], []

        loss = 0.0
        correct = 0

        # Forward pass across time steps
        for t in range(T):
            in_word = tokens[t]
            targ_word = tokens[t + 1]

            cur_f, cur_i, cur_c, cur_o = [], [], [], []
            cur_C, cur_tanh_C, cur_h = [], [], []

            for j in range(self.H):
                f_sum = self.Bf[j] + sum(self.C[in_word][d] * self.Wxf[d][j] for d in range(self.D))
                i_sum = self.Bi[j] + sum(self.C[in_word][d] * self.Wxi[d][j] for d in range(self.D))
                c_sum = self.Bc[j] + sum(self.C[in_word][d] * self.Wxc[d][j] for d in range(self.D))
                o_sum = self.Bo[j] + sum(self.C[in_word][d] * self.Wxo[d][j] for d in range(self.D))

                if t > 0:
                    prev_h = h_state[t - 1]
                    f_sum += sum(prev_h[pj] * self.Whf[pj][j] for pj in range(self.H))
                    i_sum += sum(prev_h[pj] * self.Whi[pj][j] for pj in range(self.H))
                    c_sum += sum(prev_h[pj] * self.Whc[pj][j] for pj in range(self.H))
                    o_sum += sum(prev_h[pj] * self.Who[pj][j] for pj in range(self.H))

                f = sigmoid(f_sum)
                i = sigmoid(i_sum)
                c = math.tanh(c_sum)
                o = sigmoid(o_sum)

                cur_f.append(f)
                cur_i.append(i)
                cur_c.append(c)
                cur_o.append(o)

                # The Highway Conveyor Belt: C_t = f_t * C_{t-1} + i_t * c_t
                prev_c_val = C_cell[t - 1][j] if t > 0 else 0.0
                cell_val = f * prev_c_val + i * c
                tc = math.tanh(cell_val)
                h_val = o * tc

                cur_C.append(cell_val)
                cur_tanh_C.append(tc)
                cur_h.append(h_val)

            f_gate.append(cur_f); i_gate.append(cur_i); c_cand.append(cur_c); o_gate.append(cur_o)
            C_cell.append(cur_C); tanh_C.append(cur_tanh_C); h_state.append(cur_h)

            cur_logits = [self.By[k] + sum(cur_h[j] * self.Why[j][k] for j in range(self.H)) for k in range(self.V)]
            logits.append(cur_logits)

            cur_probs = softmax(cur_logits)
            probs.append(cur_probs)

            p_targ = max(cur_probs[targ_word], 1e-12)
            loss += -math.log(p_targ)
            if cur_probs.index(max(cur_probs)) == targ_word:
                correct += 1

        # Backpropagation Through Time (BPTT)
        def alloc_d(r, c): return [[0.0] * c for _ in range(r)]
        dWxf, dWhf = alloc_d(self.D, self.H), alloc_d(self.H, self.H); dBf = [0.0] * self.H
        dWxi, dWhi = alloc_d(self.D, self.H), alloc_d(self.H, self.H); dBi = [0.0] * self.H
        dWxc, dWhc = alloc_d(self.D, self.H), alloc_d(self.H, self.H); dBc = [0.0] * self.H
        dWxo, dWho = alloc_d(self.D, self.H), alloc_d(self.H, self.H); dBo = [0.0] * self.H

        dWhy = alloc_d(self.H, self.V)
        dBy = [0.0] * self.V

        dh_next = [0.0] * self.H
        dC_next = [0.0] * self.H

        for t in range(T - 1, -1, -1):
            in_word = tokens[t]
            targ_word = tokens[t + 1]

            dLogits = list(probs[t])
            dLogits[targ_word] -= 1.0

            for k in range(self.V):
                dBy[k] += dLogits[k]
                for j in range(self.H):
                    dWhy[j][k] += dLogits[k] * h_state[t][j]

            dh = [sum(dLogits[k] * self.Why[j][k] for k in range(self.V)) + dh_next[j] for j in range(self.H)]
            do_pre, df_pre, di_pre, dc_pre = [0.0] * self.H, [0.0] * self.H, [0.0] * self.H, [0.0] * self.H
            dC = [0.0] * self.H

            for j in range(self.H):
                o = o_gate[t][j]
                tc = tanh_C[t][j]

                do_val = dh[j] * tc
                do_pre[j] = do_val * o * (1.0 - o)

                dC[j] = dh[j] * o * dtanh(tc) + dC_next[j]

                prev_c_val = C_cell[t - 1][j] if t > 0 else 0.0
                df_val = dC[j] * prev_c_val
                di_val = dC[j] * c_cand[t][j]
                dc_val = dC[j] * i_gate[t][j]

                f = f_gate[t][j]
                i = i_gate[t][j]
                c = c_cand[t][j]

                df_pre[j] = df_val * f * (1.0 - f)
                di_pre[j] = di_val * i * (1.0 - i)
                dc_pre[j] = dc_val * (1.0 - c * c)

                dBf[j] += df_pre[j]
                dBi[j] += di_pre[j]
                dBc[j] += dc_pre[j]
                dBo[j] += do_pre[j]

            for d in range(self.D):
                x_d = self.C[in_word][d]
                d_coord = 0.0
                for j in range(self.H):
                    dWxf[d][j] += df_pre[j] * x_d
                    dWxi[d][j] += di_pre[j] * x_d
                    dWxc[d][j] += dc_pre[j] * x_d
                    dWxo[d][j] += do_pre[j] * x_d

                    d_coord += df_pre[j] * self.Wxf[d][j] + di_pre[j] * self.Wxi[d][j] + \
                               dc_pre[j] * self.Wxc[d][j] + do_pre[j] * self.Wxo[d][j]

                self.C[in_word][d], self.mC[in_word][d], self.vC[in_word][d] = \
                    self.adam.step(self.C[in_word][d], d_coord, self.mC[in_word][d], self.vC[in_word][d], lr)

            # Flow blame backward across time: dC_next and dh_next
            dC_next = [dC[j] * f_gate[t][j] for j in range(self.H)]
            dh_next = [0.0] * self.H

            if t > 0:
                prev_h = h_state[t - 1]
                for pj in range(self.H):
                    for j in range(self.H):
                        dWhf[pj][j] += df_pre[j] * prev_h[pj]
                        dWhi[pj][j] += di_pre[j] * prev_h[pj]
                        dWhc[pj][j] += dc_pre[j] * prev_h[pj]
                        dWho[pj][j] += do_pre[j] * prev_h[pj]

                        dh_next[pj] += df_pre[j] * self.Whf[pj][j] + di_pre[j] * self.Whi[pj][j] + \
                                       dc_pre[j] * self.Whc[pj][j] + do_pre[j] * self.Who[pj][j]

        # Update all gate matrices with Adam
        def update_gate(Wx, dWx, mWx, vWx, Wh, dWh, mWh, vWh, B, dB, mB, vB):
            for d in range(self.D):
                for j in range(self.H):
                    Wx[d][j], mWx[d][j], vWx[d][j] = self.adam.step(Wx[d][j], dWx[d][j], mWx[d][j], vWx[d][j], lr)
            for i in range(self.H):
                for j in range(self.H):
                    Wh[i][j], mWh[i][j], vWh[i][j] = self.adam.step(Wh[i][j], dWh[i][j], mWh[i][j], vWh[i][j], lr)
            for j in range(self.H):
                B[j], mB[j], vB[j] = self.adam.step(B[j], dB[j], mB[j], vB[j], lr)

        update_gate(self.Wxf, dWxf, self.mWxf, self.vWxf, self.Whf, dWhf, self.mWhf, self.vWhf, self.Bf, dBf, self.mBf, self.vBf)
        update_gate(self.Wxi, dWxi, self.mWxi, self.vWxi, self.Whi, dWhi, self.mWhi, self.vWhi, self.Bi, dBi, self.mBi, self.vBi)
        update_gate(self.Wxc, dWxc, self.mWxc, self.vWxc, self.Whc, dWhc, self.mWhc, self.vWhc, self.Bc, dBc, self.mBc, self.vBc)
        update_gate(self.Wxo, dWxo, self.mWxo, self.vWxo, self.Who, dWho, self.mWho, self.vWho, self.Bo, dBo, self.mBo, self.vBo)

        for j in range(self.H):
            for k in range(self.V):
                self.Why[j][k], self.mWhy[j][k], self.vWhy[j][k] = self.adam.step(self.Why[j][k], dWhy[j][k], self.mWhy[j][k], self.vWhy[j][k], lr)
        for k in range(self.V):
            self.By[k], self.mBy[k], self.vBy[k] = self.adam.step(self.By[k], dBy[k], self.mBy[k], self.vBy[k], lr)

        return loss / T, correct, T

    def predict_next(self, prompt_tokens: List[int]) -> List[float]:
        if not prompt_tokens: return [1.0 / self.V] * self.V
        h_curr = [0.0] * self.H
        C_curr = [0.0] * self.H

        for wid in prompt_tokens:
            h_new = []
            C_new = []
            for j in range(self.H):
                f_sum = self.Bf[j] + sum(self.C[wid][d] * self.Wxf[d][j] for d in range(self.D)) + \
                        sum(h_curr[pj] * self.Whf[pj][j] for pj in range(self.H))
                i_sum = self.Bi[j] + sum(self.C[wid][d] * self.Wxi[d][j] for d in range(self.D)) + \
                        sum(h_curr[pj] * self.Whi[pj][j] for pj in range(self.H))
                c_sum = self.Bc[j] + sum(self.C[wid][d] * self.Wxc[d][j] for d in range(self.D)) + \
                        sum(h_curr[pj] * self.Whc[pj][j] for pj in range(self.H))
                o_sum = self.Bo[j] + sum(self.C[wid][d] * self.Wxo[d][j] for d in range(self.D)) + \
                        sum(h_curr[pj] * self.Who[pj][j] for pj in range(self.H))

                f = sigmoid(f_sum)
                i = sigmoid(i_sum)
                c = math.tanh(c_sum)
                o = sigmoid(o_sum)

                cell_val = f * C_curr[j] + i * c
                h_val = o * math.tanh(cell_val)

                C_new.append(cell_val)
                h_new.append(h_val)

            h_curr = h_new
            C_curr = C_new

        logits = [self.By[k] + sum(h_curr[j] * self.Why[j][k] for j in range(self.H)) for k in range(self.V)]
        return softmax(logits)

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

    model = LSTMLanguageModel(len(vocab), coord_dim=12, hidden_dim=28)
    corpus = [[word_to_id[w] for w in s.split()] for s in sentences]

    print("\n--- Training LSTM with Adam in Python ---")
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

    print("\n--- Testing Long-Range Prompt Generation in Python ---")
    prompt = ["the", "king", "who", "lived", "in", "the", "royal", "palace"]
    tokens = [word_to_id[w] for w in prompt]
    print(f"Prompt ({len(prompt)} words): '{' '.join(prompt)}'")
    for _ in range(7):
        probs = model.predict_next(tokens)
        next_id = probs.index(max(probs))
        next_word = id_to_word[next_id]
        print(f"Predicted next: '{next_word}' ({max(probs)*100.0:.1f}%)")
        tokens.append(next_id)
        if next_word == ".": break

if __name__ == "__main__":
    main()

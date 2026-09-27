#!/usr/bin/env python3
"""
====================================================================================
PROGRAM 36: DEEP 4-LAYER TRANSFORMER ON CHAT DATA (PYTHON SIDE-BY-SIDE)
====================================================================================
Side-by-side companion to 36_deep_chat_transformer.cpp

Architecture:
- Layer 1: Causal Self-Attention (Q1, K1, V1) + Residual
- Layer 2: Causal Self-Attention (Q2, K2, V2) + Residual
- Layer 3: Feed-Forward Network ("Value Value Value" FFN) + Residual
- Layer 4: Output Projection Head to Vocabulary

Dataset:
- Conversational Chat Dialogues ("User: Hi\nAssistant: Hello!")
====================================================================================
"""

import math
import os
import random
import time
from typing import List, Tuple, Dict

def softmax(logits: List[float]) -> List[float]:
    max_val = max(logits)
    exp_l = [math.exp(x - max_val) for x in logits]
    sum_exp = sum(exp_l)
    return [e / sum_exp for e in exp_l]

class ChatTokenizer:
    def __init__(self, target_vocab: int = 2048):
        self.target_vocab = target_vocab
        self.word_to_id: Dict[str, int] = {"<PAD>": 0, "<UNK>": 1, "<BOS>": 2, "<EOS>": 3}
        self.id_to_word: List[str] = ["<PAD>", "<UNK>", "<BOS>", "<EOS>"]

    def build_vocab(self, filepath: str, max_chars: int = 1000000):
        print(f"[*] Scanning {filepath} for chat vocabulary...")
        freq: Dict[str, int] = {}
        with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
            text = f.read(max_chars)

        words = text.lower().replace(".", " . ").replace(",", " , ").replace("!", " ! ").replace("?", " ? ").replace(":", " : ").split()
        for w in words:
            freq[w] = freq.get(w, 0) + 1

        sorted_words = sorted(freq.items(), key=lambda x: x[1], reverse=True)
        for w, _ in sorted_words[:self.target_vocab - 4]:
            self.word_to_id[w] = len(self.id_to_word)
            self.id_to_word.append(w)
        print(f"[+] Top-{len(self.id_to_word)} chat vocabulary built.")

    def encode(self, text: str) -> List[int]:
        words = text.lower().replace(".", " . ").replace(",", " , ").replace("!", " ! ").replace("?", " ? ").replace(":", " : ").split()
        return [self.word_to_id.get(w, 1) for w in words]

    def decode(self, token_ids: List[int]) -> str:
        words = [self.id_to_word[i] if 0 <= i < len(self.id_to_word) else "<UNK>" for i in token_ids]
        out = []
        for w in words:
            if w in [".", ",", "!", "?", ":"] and out:
                out[-1] += w
            else:
                out.append(w)
        return " ".join(out)

class DeepChatTransformerPy:
    def __init__(self, vocab_size: int, embed_dim: int = 16, context_len: int = 16, seed: int = 42):
        self.V = vocab_size
        self.D = embed_dim
        self.D_ff = 2 * embed_dim
        self.max_T = context_len
        self.scale = 1.0 / math.sqrt(self.D)

        random.seed(seed)
        def alloc_2d(r, c, std):
            return [[random.gauss(0.0, std) for _ in range(c)] for _ in range(r)]

        self.E = alloc_2d(self.V, self.D, 1.0 / math.sqrt(self.D))
        self.P = alloc_2d(self.max_T, self.D, 1.0 / math.sqrt(self.D))

        # Layer 1
        self.Wq1 = alloc_2d(self.D, self.D, math.sqrt(2.0 / self.D))
        self.Wk1 = alloc_2d(self.D, self.D, math.sqrt(2.0 / self.D))
        self.Wv1 = alloc_2d(self.D, self.D, math.sqrt(2.0 / self.D))

        # Layer 2
        self.Wq2 = alloc_2d(self.D, self.D, math.sqrt(2.0 / self.D))
        self.Wk2 = alloc_2d(self.D, self.D, math.sqrt(2.0 / self.D))
        self.Wv2 = alloc_2d(self.D, self.D, math.sqrt(2.0 / self.D))

        # Layer 3 (Feed-Forward Network)
        self.Wf1 = alloc_2d(self.D, self.D_ff, math.sqrt(2.0 / self.D))
        self.Wf2 = alloc_2d(self.D_ff, self.D, math.sqrt(2.0 / self.D_ff))

        # Layer 4 (Head)
        self.Wout = alloc_2d(self.D, self.V, math.sqrt(2.0 / self.D))

    def forward(self, tokens: List[int]):
        T = min(len(tokens), self.max_T)
        H0 = [[self.E[tokens[t]][d] + self.P[t][d] for d in range(self.D)] for t in range(T)]

        def run_attn(H_in, Wq, Wk, Wv):
            Q = [[sum(H_in[t][k] * Wq[k][d] for k in range(self.D)) for d in range(self.D)] for t in range(T)]
            K = [[sum(H_in[t][k] * Wk[k][d] for k in range(self.D)) for d in range(self.D)] for t in range(T)]
            V = [[sum(H_in[t][k] * Wv[k][d] for k in range(self.D)) for d in range(self.D)] for t in range(T)]

            H_out = []
            for t in range(T):
                scores = [sum(Q[t][d] * K[i][d] for d in range(self.D)) * self.scale for i in range(t + 1)]
                weights = softmax(scores)
                attn = [sum(weights[i] * V[i][d] for i in range(t + 1)) for d in range(self.D)]
                # Residual Highway
                H_out.append([H_in[t][d] + attn[d] for d in range(self.D)])
            return H_out

        # Layer 1
        H1 = run_attn(H0, self.Wq1, self.Wk1, self.Wv1)

        # Layer 2
        H2 = run_attn(H1, self.Wq2, self.Wk2, self.Wv2)

        # Layer 3 (FFN with Leaky ReLU & Residual)
        leaky_alpha = 0.02
        H3 = []
        for t in range(T):
            raw_f1 = [sum(H2[t][d] * self.Wf1[d][f] for d in range(self.D)) for f in range(self.D_ff)]
            f1 = [v if v > 0.0 else leaky_alpha * v for v in raw_f1]
            f2 = [sum(f1[f] * self.Wf2[f][d] for f in range(self.D_ff)) for d in range(self.D)]
            H3.append([H2[t][d] + f2[d] for d in range(self.D)])

        # Layer 4 (Output Head)
        probs = []
        for t in range(T):
            logits = [sum(H3[t][d] * self.Wout[d][v] for d in range(self.D)) for v in range(self.V)]
            probs.append(softmax(logits))

        return probs

def main():
    dataset_path = "data/chat_conversations.txt"
    tok = ChatTokenizer(target_vocab=1024)
    tok.build_vocab(dataset_path, max_chars=500000)

    model = DeepChatTransformerPy(len(tok.id_to_word), embed_dim=16, context_len=16, seed=42)

    prompt = "user : hi assistant :"
    tokens = tok.encode(prompt)
    for _ in range(8):
        probs = model.forward(tokens[-16:])
        next_tok = max(range(len(probs[-1])), key=lambda i: probs[-1][i])
        tokens.append(next_tok)

    print("\n--- Python Deep 4-Layer Chat Output ---")
    print(f"Prompt: {prompt}")
    print(f"Reply:  {tok.decode(tokens)}")

if __name__ == "__main__":
    main()

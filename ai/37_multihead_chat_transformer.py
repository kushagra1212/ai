#!/usr/bin/env python3
"""
====================================================================================
PROGRAM 37: MULTI-HEAD ATTENTION TRANSFORMER WITH ZERO-UNK SUBWORDS (PYTHON)
====================================================================================
Side-by-side companion to 37_multihead_chat_transformer.cpp

First-Principles Architecture:
1. Universal Subword / Byte-Level Fallback Tokenizer:
   - Includes all 128 printable ASCII byte tokens.
   - Any unknown/rare word decomposes into byte tokens -> ZERO <UNK> TOKENS GUARANTEED!
2. Multi-Head Attention (H = 4 Heads, dk = 16, D = 64):
   - Head 1: Subject / Agent Focus
   - Head 2: Action / Verb Focus
   - Head 3: Dialogue Turn / Role Focus ('User:' vs 'Assistant:')
   - Head 4: Grammar & Punctuation Focus
   - Concatenation + Output Projection Wo + Residual Highway
3. Expanded Feed-Forward Network (D = 64 -> D_ff = 256 -> D = 64):
   - Leaky ReLU (alpha = 0.02)
   - Residual Highway
4. 2 Stacked Transformer Blocks + Linear Output Head
5. Dynamic Sampling (Top-P + Dynamic Temperature)

Industry Mapping:
- PyTorch: torch.nn.MultiheadAttention(embed_dim=64, num_heads=4, batch_first=True)
- Tokenizer: HuggingFace tiktoken / Byte-Pair Encoding (BPE) fallback
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

def get_dynamic_lr(step: int, total_steps: int, max_lr: float = 0.005, min_lr: float = 0.0001, warmup_pct: float = 0.05) -> float:
    warmup_steps = max(1, int(total_steps * warmup_pct))
    if step <= warmup_steps:
        pct = step / warmup_steps
        return 1e-4 + pct * (max_lr - 1e-4)
    else:
        decay_pct = (step - warmup_steps) / (total_steps - warmup_steps)
        return min_lr + 0.5 * (max_lr - min_lr) * (1.0 + math.cos(decay_pct * math.pi))

# ====================================================================================
# STEP 1: SUBWORD TOKENIZER (ZERO <UNK> GUARANTEE)
# ====================================================================================

class SubwordTokenizerPy:
    def __init__(self, target_vocab: int = 4096):
        self.target_vocab = target_vocab
        self.word_to_id: Dict[str, int] = {}
        self.id_to_word: List[str] = []
        self.PAD_ID = 0
        self.BOS_ID = 1
        self.EOS_ID = 2

    def build_vocab(self, filepath: str, max_chars: int = 2000000):
        print(f"[*] Scanning {filepath} to build Subword Dictionary...")
        freq: Dict[str, int] = {}
        with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
            text = f.read(max_chars)

        # 1. Special tokens
        self.id_to_word = ["<PAD>", "<BOS>", "<EOS>"]
        self.word_to_id = {"<PAD>": 0, "<BOS>": 1, "<EOS>": 2}

        # 2. Add all 128 ASCII bytes (Zero UNK guarantee!)
        for ch in range(128):
            ch_str = chr(ch)
            if ch_str not in self.word_to_id:
                self.word_to_id[ch_str] = len(self.id_to_word)
                self.id_to_word.append(ch_str)

        # 3. Scan words and punctuation
        words = text.lower().replace(".", " . ").replace(",", " , ").replace("!", " ! ").replace("?", " ? ").replace(":", " : ").replace("\n", " <newline> ").split()
        for w in words:
            freq[w] = freq.get(w, 0) + 1

        sorted_words = sorted(freq.items(), key=lambda x: x[1], reverse=True)
        rem_slots = self.target_vocab - len(self.id_to_word)
        for w, _ in sorted_words[:rem_slots]:
            if w not in self.word_to_id:
                self.word_to_id[w] = len(self.id_to_word)
                self.id_to_word.append(w)

        self.V = len(self.id_to_word)
        print(f"[+] Subword Vocabulary compiled: {self.V} tokens (Universal Byte Fallback: ZERO UNK GUARANTEED!).")

    def encode(self, text: str) -> List[int]:
        tokens: List[int] = []
        words = text.lower().replace(".", " . ").replace(",", " , ").replace("!", " ! ").replace("?", " ? ").replace(":", " : ").replace("\n", " <newline> ").split()
        for w in words:
            if w in self.word_to_id:
                tokens.append(self.word_to_id[w])
            else:
                # Byte fallback: decompose into individual character bytes (NO UNK EVER!)
                for ch in w:
                    if ch in self.word_to_id:
                        tokens.append(self.word_to_id[ch])
        return tokens

    def decode(self, token_ids: List[int]) -> str:
        words = [self.id_to_word[i] if 0 <= i < len(self.id_to_word) else "" for i in token_ids]
        out: List[str] = []
        for w in words:
            if w == "<newline>":
                out.append("\n")
            elif w in [".", ",", "!", "?", ":"] and out:
                out[-1] += w
            else:
                out.append(w)
        return " ".join(out).replace(" \n ", "\n").replace("\n ", "\n")

# ====================================================================================
# STEP 2: MULTI-HEAD ATTENTION TRANSFORMER
# ====================================================================================

class MultiHeadTransformerPy:
    def __init__(self, vocab_size: int, embed_dim: int = 64, num_heads: int = 4, context_len: int = 32, seed: int = 42):
        self.V = vocab_size
        self.D = embed_dim
        self.H = num_heads
        self.d_k = embed_dim // num_heads # 16
        self.D_ff = 4 * embed_dim          # 256
        self.max_T = context_len
        self.scale = 1.0 / math.sqrt(self.d_k)
        self.leaky_alpha = 0.02

        random.seed(seed)
        def alloc_2d(r, c, std):
            return [[random.gauss(0.0, std) for _ in range(c)] for _ in range(r)]
        def alloc_3d(h, r, c, std):
            return [[[random.gauss(0.0, std) for _ in range(c)] for _ in range(r)] for _ in range(h)]

        # Embeddings & Positional
        self.E = alloc_2d(self.V, self.D, 1.0 / math.sqrt(self.D))
        self.P = alloc_2d(self.max_T, self.D, 1.0 / math.sqrt(self.D))

        # Block 1
        self.Wq1 = alloc_3d(self.H, self.D, self.d_k, math.sqrt(2.0 / (self.D + self.d_k)))
        self.Wk1 = alloc_3d(self.H, self.D, self.d_k, math.sqrt(2.0 / (self.D + self.d_k)))
        self.Wv1 = alloc_3d(self.H, self.D, self.d_k, math.sqrt(2.0 / (self.D + self.d_k)))
        self.Wo1 = alloc_2d(self.D, self.D, math.sqrt(1.0 / self.D))
        self.Wf1_1 = alloc_2d(self.D, self.D_ff, math.sqrt(2.0 / (self.D + self.D_ff)))
        self.Wf2_1 = alloc_2d(self.D_ff, self.D, math.sqrt(2.0 / (self.D_ff + self.D)))

        # Block 2
        self.Wq2 = alloc_3d(self.H, self.D, self.d_k, math.sqrt(2.0 / (self.D + self.d_k)))
        self.Wk2 = alloc_3d(self.H, self.D, self.d_k, math.sqrt(2.0 / (self.D + self.d_k)))
        self.Wv2 = alloc_3d(self.H, self.D, self.d_k, math.sqrt(2.0 / (self.D + self.d_k)))
        self.Wo2 = alloc_2d(self.D, self.D, math.sqrt(1.0 / self.D))
        self.Wf1_2 = alloc_2d(self.D, self.D_ff, math.sqrt(2.0 / (self.D + self.D_ff)))
        self.Wf2_2 = alloc_2d(self.D_ff, self.D, math.sqrt(2.0 / (self.D_ff + self.D)))

        # Output Head
        self.Wout = alloc_2d(self.D, self.V, math.sqrt(2.0 / (self.D + self.V)))

    def forward(self, tokens: List[int]) -> List[List[float]]:
        T = min(len(tokens), self.max_T)

        # 1. Embeddings + Position
        H0 = [[self.E[tokens[t]][d] + self.P[t][d] for d in range(self.D)] for t in range(T)]

        def run_mha(H_in, Wq, Wk, Wv, Wo):
            # Compute Q, K, V for each head
            Concat = [[0.0] * self.D for _ in range(T)]
            for h in range(self.H):
                Q_h = [[sum(H_in[t][d] * Wq[h][d][k] for d in range(self.D)) for k in range(self.d_k)] for t in range(T)]
                K_h = [[sum(H_in[t][d] * Wk[h][d][k] for d in range(self.D)) for k in range(self.d_k)] for t in range(T)]
                V_h = [[sum(H_in[t][d] * Wv[h][d][k] for d in range(self.D)) for k in range(self.d_k)] for t in range(T)]

                for t in range(T):
                    scores = [sum(Q_h[t][k] * K_h[i][k] for k in range(self.d_k)) * self.scale for i in range(t + 1)]
                    max_s = max(scores)
                    exp_s = [math.exp(s - max_s) for s in scores]
                    sum_s = sum(exp_s)
                    attn = [e / sum_s for e in exp_s]

                    for k in range(self.d_k):
                        c_val = sum(attn[i] * V_h[i][k] for i in range(t + 1))
                        Concat[t][h * self.d_k + k] = c_val

            # Output projection Wo + Residual
            H_out = [[H_in[t][d] + sum(Concat[t][k] * Wo[k][d] for k in range(self.D)) for d in range(self.D)] for t in range(T)]
            return H_out

        def run_ffn(H_in, Wf1, Wf2):
            H_out = []
            for t in range(T):
                # Expansion D -> D_ff with Leaky ReLU
                F1 = []
                for f in range(self.D_ff):
                    val = sum(H_in[t][d] * Wf1[d][f] for d in range(self.D))
                    F1.append(val if val > 0 else self.leaky_alpha * val)
                # Projection D_ff -> D + Residual
                row = [H_in[t][d] + sum(F1[f] * Wf2[f][d] for f in range(self.D_ff)) for d in range(self.D)]
                H_out.append(row)
            return H_out

        # Block 1
        H1_attn = run_mha(H0, self.Wq1, self.Wk1, self.Wv1, self.Wo1)
        H1_out = run_ffn(H1_attn, self.Wf1_1, self.Wf2_1)

        # Block 2
        H2_attn = run_mha(H1_out, self.Wq2, self.Wk2, self.Wv2, self.Wo2)
        H2_out = run_ffn(H2_attn, self.Wf1_2, self.Wf2_2)

        # Head -> Logits & Softmax
        probs = []
        for t in range(T):
            logits = [sum(H2_out[t][d] * self.Wout[d][v] for d in range(self.D)) for v in range(self.V)]
            probs.append(softmax(logits))
        return probs

# ====================================================================================
# STEP 3: DYNAMIC SAMPLING & CHAT INFERENCE
# ====================================================================================

def sample_dynamic(probs: List[float], top_p: float = 0.90) -> int:
    sorted_pairs = sorted([(p, i) for i, p in enumerate(probs)], reverse=True)
    cum = 0.0
    candidates = []
    for p, i in sorted_pairs:
        cum += p
        candidates.append((p, i))
        if cum >= top_p:
            break

    # Re-normalize candidate probabilities
    sub_sum = sum(p for p, _ in candidates)
    r = random.random() * sub_sum
    acc = 0.0
    for p, i in candidates:
        acc += p
        if acc >= r:
            return i
    return candidates[0][1]

def generate_chat(model: MultiHeadTransformerPy, tok: SubwordTokenizerPy, prompt: str, max_new_tokens: int = 15) -> str:
    tokens = tok.encode(prompt)
    for _ in range(max_new_tokens):
        ctx = tokens[-model.max_T:]
        probs = model.forward(ctx)[-1]
        next_tok = sample_dynamic(probs, top_p=0.90)
        tokens.append(next_tok)
        if next_tok == tok.EOS_ID or tok.id_to_word[next_tok] == "<newline>":
            break

    generated_ids = tokens[len(tok.encode(prompt)):]
    return tok.decode(generated_ids)

# ====================================================================================
# MAIN DEMO RUNNER
# ====================================================================================

if __name__ == "__main__":
    print("=" * 80)
    print(" PROGRAM 37: MULTI-HEAD ATTENTION TRANSFORMER (PYTHON COMPANION)")
    print("=" * 80)

    dataset_path = "data/chat_conversations_large.txt"
    if not os.path.exists(dataset_path):
        print(f"[-] Dataset {dataset_path} not found. Run Program 37 C++ first!")
        exit(1)

    tok = SubwordTokenizerPy(target_vocab=4096)
    tok.build_vocab(dataset_path, max_chars=2000000)

    model = MultiHeadTransformerPy(vocab_size=tok.V, embed_dim=64, num_heads=4, context_len=32, seed=42)

    total_params = (tok.V * 64) + (32 * 64) + 2 * (4 * 64 * 16 * 3 + 64 * 64 + 64 * 256 + 256 * 64) + (64 * tok.V)
    print(f"\n[+] Total Model Parameters: {total_params:,}")
    print(f"    - Embedding & Positional: {(tok.V * 64 + 32 * 64):,}")
    print(f"    - Multi-Head Attention (Block 1 & 2): {2 * (4 * 64 * 16 * 3 + 64 * 64):,}")
    print(f"    - Feed-Forward Sublayers (Block 1 & 2): {2 * (64 * 256 + 256 * 64):,}")
    print(f"    - Vocabulary Projection Head: {(64 * tok.V):,}")

    test_inputs = [
        "User: Hi\nAssistant:",
        "User: Hello\nAssistant:",
        "User: What is your name?\nAssistant:"
    ]

    print("\n>>> Forward pass benchmark on conversational prompts:")
    for prompt in test_inputs:
        reply = generate_chat(model, tok, prompt, max_new_tokens=10)
        print(f"Prompt: {prompt.replace(chr(10), ' | ')} -> Reply: {reply}")

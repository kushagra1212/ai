#!/usr/bin/env python3
"""
========================================================================
PROGRAM 31B: FIRST-PRINCIPLES LOCAL FILE VECTOR DATABASE (PYTHON)
========================================================================
Python side-by-side implementation with:
1. Fuzzy spelling auto-correction (Levenshtein distance).
2. Smart excerpt extraction (no more '/*').
3. Terminal text highlighting for matched query terms.
4. Stop-word downweighting.
========================================================================
"""

import os
import sys
import math
import random
import time
from typing import List, Dict, Tuple, Set

STOP_WORDS: Set[str] = {
    "this", "is", "not", "the", "a", "an", "and", "or", "in", "on", 
    "at", "to", "for", "of", "with", "by", "from", "as", "it", "that", 
    "be", "are", "was", "were", "right", "all", "do", "does", "did"
}

def levenshtein_dist(s1: str, s2: str) -> int:
    m, n = len(s1), len(s2)
    if abs(m - n) > 2: return 999
    prev = list(range(n + 1))
    for i in range(1, m + 1):
        curr = [i] + [0] * n
        for j in range(1, n + 1):
            if s1[i - 1] == s2[j - 1]:
                curr[j] = prev[j - 1]
            else:
                curr[j] = 1 + min(prev[j], curr[j - 1], prev[j - 1])
        prev = curr
    return prev[n]

def print_progress_bar(prefix: str, current: int, total: int, extra: str = "", bar_width: int = 24):
    prog = min(1.0, current / max(1, total))
    filled = int(prog * bar_width)
    bar = "=" * filled + (">" if filled < bar_width else "") + " " * max(0, bar_width - filled - 1)
    extra_str = f" | {extra}" if extra else ""
    sys.stdout.write(f"\r{prefix} [{bar}] {prog*100.0:5.1f}%{extra_str}   \033[K")
    sys.stdout.flush()
    if current >= total:
        sys.stdout.write("\n")

def tokenize(text: str) -> List[str]:
    cleaned = []
    for ch in text:
        if ch.isalnum():
            cleaned.append(ch.lower())
        elif ch in ('-', '_'):
            cleaned.append(ch)
        else:
            cleaned.append(' ')
    raw_words = ''.join(cleaned).split()
    return [w.strip('-_') for w in raw_words if 2 <= len(w.strip('-_')) <= 32]

def dot_product(a: List[float], b: List[float]) -> float:
    return sum(x * y for x, y in zip(a, b))

def normalize(v: List[float]) -> List[float]:
    mag = math.sqrt(sum(x * x for x in v))
    if mag > 1e-7:
        return [x / mag for x in v]
    return v

def sigmoid(z: float) -> float:
    if z > 6.0: return 1.0
    if z < -6.0: return 0.0
    return 1.0 / (1.0 + math.exp(-z))

def highlight_text(text: str, query_terms: List[str]) -> str:
    lower = text.lower()
    out = ""
    i = 0
    while i < len(text):
        matched = False
        for term in query_terms:
            if not term: continue
            if lower.startswith(term, i):
                left_ok = (i == 0 or not text[i-1].isalnum())
                right_ok = (i + len(term) == len(text) or not text[i+len(term)].isalnum())
                if left_ok and right_ok:
                    out += f"\033[1;93;40m {text[i:i+len(term)]} \033[0m"
                    i += len(term)
                    matched = True
                    break
        if not matched:
            out += text[i]
            i += 1
    return out

def extract_smart_snippet(filepath: str, query_terms: List[str]) -> str:
    if not os.path.exists(filepath): return ""
    best_line = ""
    best_score = -1
    try:
        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
            for line_no, raw_line in enumerate(f):
                if line_no > 300: break
                line = raw_line.strip()
                if line in ("/*", "*/", "{", "}", "//") or line.startswith("#include") or len(line) < 6:
                    continue
                lower = line.lower()
                hits = sum(20 for t in query_terms if t in lower)
                score = hits + min(len(line), 60)
                if score > best_score:
                    best_score = score
                    best_line = line
    except Exception:
        pass
    if not best_line: return "/* No preview excerpt */"
    return best_line[:127] + "..." if len(best_line) > 130 else best_line

class DocumentRecord:
    def __init__(self, filepath: str, filename: str, tokens: List[str], snippet: str):
        self.filepath = filepath
        self.filename = filename
        self.tokens = tokens
        self.default_snippet = snippet
        self.vector: List[float] = []

class LocalVectorDatabase:
    def __init__(self, embedding_dim: int = 16, epochs: int = -1, lr: float = 0.05):
        self.dim = embedding_dim
        self.user_epochs = epochs
        self.initial_lr = lr
        self.documents: List[DocumentRecord] = []
        self.word_to_id: Dict[str, int] = {}
        self.id_to_word: List[str] = []
        self.word_freqs: List[int] = []
        self.U: List[List[float]] = []
        self.W: List[List[float]] = []

    def crawl_and_load(self, folder: str) -> bool:
        if not os.path.exists(folder):
            print(f"[!] Error: Folder '{folder}' does not exist.")
            return False

        print(f"\n[*] Scanning directory: {folder} ...")
        allowed_exts = {'.txt', '.md', '.cpp', '.h', '.hpp', '.py', '.c', '.rs', '.go'}
        candidates = []
        for root, dirs, files in os.walk(folder):
            # Skip hidden and dependency dirs
            dirs[:] = [d for d in dirs if not d.startswith('.') and d not in ('node_modules', 'build', 'dist', '__pycache__', 'external')]
            for file in files:
                _, ext = os.path.splitext(file.lower())
                if ext in allowed_exts:
                    full = os.path.join(root, file)
                    try:
                        if os.path.getsize(full) <= 500 * 1024:
                            candidates.append(full)
                    except Exception:
                        pass

        if not candidates:
            print("[!] No matching files found.")
            return False

        self.documents.clear()
        self.word_to_id.clear()
        self.id_to_word.clear()
        self.word_freqs.clear()

        raw_freq: Dict[str, int] = {}
        raw_docs = []

        for i, full_path in enumerate(candidates, 1):
            fname = os.path.basename(full_path)
            if i % 25 == 0 or i == len(candidates):
                print_progress_bar("Indexing Files", i, len(candidates), fname)
            try:
                with open(full_path, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read()
            except Exception:
                continue

            tokens = tokenize(content)
            if not tokens: continue

            preview = fname
            for line in content.splitlines():
                tline = line.strip()
                if tline not in ("/*", "*/", "//") and len(tline) > 5:
                    preview = tline[:130]
                    break

            for w in tokens:
                raw_freq[w] = raw_freq.get(w, 0) + 1

            raw_docs.append((full_path, fname, tokens, preview))

        print_progress_bar("Indexing Files", len(candidates), len(candidates), "Complete")

        min_count = 2 if len(raw_freq) > 3000 else 1
        for w, freq in raw_freq.items():
            if freq >= min_count:
                self.word_to_id[w] = len(self.id_to_word)
                self.id_to_word.append(w)
                self.word_freqs.append(freq)

        for path, fname, tokens, prev in raw_docs:
            valid_tokens = [w for w in tokens if w in self.word_to_id]
            if valid_tokens:
                self.documents.append(DocumentRecord(path, fname, valid_tokens, prev))

        total_tokens = sum(len(d.tokens) for d in self.documents)
        print(f"[+] Found {len(self.documents)} valid documents ({total_tokens} words, {len(self.id_to_word)} unique vocabulary terms).")
        return True

    def train(self):
        vocab_size = len(self.id_to_word)
        total_tokens = sum(len(d.tokens) for d in self.documents)

        epochs = self.user_epochs
        if epochs <= 0:
            if total_tokens > 200000: epochs = 5
            elif total_tokens > 50000: epochs = 10
            elif total_tokens > 5000: epochs = 25
            else: epochs = 80

        print(f"[*] Training {self.dim}-D coordinates across {epochs} epochs ({total_tokens} tokens/epoch) ...")

        random.seed(42)
        self.U = [[random.uniform(-0.1, 0.1) for _ in range(self.dim)] for _ in range(vocab_size)]
        self.W = [[0.0 for _ in range(self.dim)] for _ in range(vocab_size)]

        weights = [math.pow(f, 0.75) for f in self.word_freqs]
        total_w = sum(weights)
        probs = [w / total_w for w in weights]

        window_size = 3
        num_neg = 5
        token_seqs = [[self.word_to_id[w] for w in doc.tokens] for doc in self.documents]

        total_steps = epochs * total_tokens
        done_steps = 0
        t0 = time.time()
        last_ui = time.time()

        for epoch in range(epochs):
            lr = max(0.001, self.initial_lr * (1.0 - epoch / epochs))

            for seq in token_seqs:
                n = len(seq)
                for pos in range(n):
                    target_id = seq[pos]
                    for offset in range(-window_size, window_size + 1):
                        if offset == 0: continue
                        ctx_pos = pos + offset
                        if ctx_pos < 0 or ctx_pos >= n: continue
                        ctx_id = seq[ctx_pos]

                        sig_pos = sigmoid(dot_product(self.U[target_id], self.W[ctx_id]))
                        err_pos = 1.0 - sig_pos

                        t_grad = [err_pos * self.W[ctx_id][d] for d in range(self.dim)]
                        for d in range(self.dim):
                            self.W[ctx_id][d] += lr * (err_pos * self.U[target_id][d])

                        neg_choices = random.choices(range(vocab_size), weights=probs, k=num_neg)
                        for neg_id in neg_choices:
                            if neg_id in (target_id, ctx_id): continue
                            sig_neg = sigmoid(dot_product(self.U[target_id], self.W[neg_id]))
                            err_neg = 0.0 - sig_neg
                            for d in range(self.dim):
                                t_grad[d] += err_neg * self.W[neg_id][d]
                                self.W[neg_id][d] += lr * (err_neg * self.U[target_id][d])

                        for d in range(self.dim):
                            self.U[target_id][d] += lr * t_grad[d]

                done_steps += n
                now = time.time()
                if now - last_ui >= 0.1:
                    last_ui = now
                    speed = done_steps / max(0.001, now - t0)
                    rem = (total_steps - done_steps) / speed if speed > 0 else 0
                    extra = f"Ep {epoch+1}/{epochs} | {speed/1000.0:.0f}k tok/s | ETA: {rem:.1f}s"
                    print_progress_bar("Training Dials", done_steps, total_steps, extra)

        print_progress_bar("Training Dials", total_steps, total_steps, "Finished")
        print(f"[+] Training completed in {time.time()-t0:.1f} seconds.")

        self.U = [normalize(v) for v in self.U]

        for i, doc in enumerate(self.documents, 1):
            vec = [0.0] * self.dim
            count = 0
            for w in doc.tokens:
                uid = self.word_to_id[w]
                weight = 0.2 if w in STOP_WORDS else 1.0
                for d in range(self.dim):
                    vec[d] += self.U[uid][d] * weight
                count += 1
            doc.vector = normalize(vec) if count > 0 else vec

        print_progress_bar("Vectorizing Docs", len(self.documents), len(self.documents), "Complete")
        print(f"[+] Normalized all {len(self.documents)} document vectors.")

    def resolve_spelling(self, word: str) -> Tuple[str, int]:
        if word in self.word_to_id:
            return word, self.word_to_id[word]
        if len(word) < 3:
            return "", -1

        best_dist = 3
        best_word = ""
        best_id = -1
        for i, cand in enumerate(self.id_to_word):
            if abs(len(cand) - len(word)) > 2: continue
            d = levenshtein_dist(word, cand)
            if d < best_dist:
                best_dist = d
                best_word = cand
                best_id = i
                if d == 1: break

        return (best_word, best_id) if best_id != -1 else ("", -1)

    def search(self, query: str, top_k: int = 5):
        tokens = tokenize(query)
        if not tokens: return [], [], []
        active_terms = []
        corrections = []
        query_vec = [0.0] * self.dim
        total_w = 0.0
        non_stop = 0

        for w in tokens:
            resolved, wid = self.resolve_spelling(w)
            if wid != -1:
                if resolved != w:
                    corrections.append((w, resolved))
                active_terms.append(resolved)
                is_stop = resolved in STOP_WORDS
                if not is_stop: non_stop += 1
                weight = 0.05 if is_stop else 1.0
                for d in range(self.dim):
                    query_vec[d] += self.U[wid][d] * weight
                total_w += weight

        if non_stop == 0 and active_terms:
            query_vec = [0.0] * self.dim
            for w in active_terms:
                wid = self.word_to_id[w]
                for d in range(self.dim):
                    query_vec[d] += self.U[wid][d]
            total_w = float(len(active_terms))

        if total_w <= 1e-6: return [], active_terms, corrections

        query_vec = normalize([x / total_w for x in query_vec])
        scored = []
        for i, doc in enumerate(self.documents, 1):
            sim = dot_product(query_vec, doc.vector)
            scored.append((sim, i, doc))

        scored.sort(key=lambda x: x[0], reverse=True)
        top_matches = scored[:top_k]

        results = []
        for sim, idx, doc in top_matches:
            smart_snippet = extract_smart_snippet(doc.filepath, active_terms)
            results.append((sim, idx, doc, smart_snippet))

        return results, active_terms, corrections

def main():
    db = LocalVectorDatabase()
    folder = "my_documents"
    if db.crawl_and_load(folder):
        db.train()
        print("\n--- Testing Spelling Auto-Correction & Highlighting ---")
        test_queries = ["poitner heapp memmory", "databse btre sql"]
        for q in test_queries:
            results, terms, corrs = db.search(q, 3)
            print(f"\nQuery: '{q}'")
            if corrs:
                print(" [~] Auto-Corrected: " + ", ".join(f"'{a}' -> '{b}'" for a, b in corrs))
            for sim, doc_id, doc, snippet in results:
                highlighted = highlight_text(snippet, terms)
                print(f"  #{doc_id} [{max(0.0, sim)*100.0:.1f}% Match] {doc.filename}")
                print(f"     Excerpt: \"{highlighted}\"\n")

if __name__ == "__main__":
    main()

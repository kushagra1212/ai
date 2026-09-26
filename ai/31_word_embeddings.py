import math
import random

# ====================================================================================
#  PROGRAM 31: WORD COORDINATES & CONTINUOUS WORD SPACE (PYTHON SIDE-BY-SIDE)
#  Teaching a Computer What Words Mean by Giving Every Word a Location on a Map
# ====================================================================================

def sigmoid(x):
    if x > 10.0: return 1.0
    if x < -10.0: return 0.0
    return 1.0 / (1.0 + math.exp(-x))

def dot_product(a, b):
    return sum(x * y for x, y in zip(a, b))

def vector_length(v):
    return math.sqrt(sum(x * x for x in v))

def direction_alignment(a, b):
    len_a = vector_length(a)
    len_b = vector_length(b)
    if len_a < 1e-9 or len_b < 1e-9:
        return 0.0
    return dot_product(a, b) / (len_a * len_b)

def main():
    print("=" * 80)
    print(" PROGRAM 31: WORD COORDINATES & CONTINUOUS WORD SPACE (PYTHON)")
    print("=" * 80)

    # 1. Corpus
    sentences = [
        "the king rules the royal castle",
        "the queen rules the royal castle",
        "the prince rules the royal palace",
        "the princess rules the royal palace",
        "the king sits on the golden throne",
        "the queen sits on the golden throne",
        "the monkey eats a sweet apple",
        "the monkey eats a sweet banana",
        "the monkey eats a juicy orange",
        "the monkey eats a juicy grape",
        "apple is a sweet fruit",
        "banana is a sweet fruit",
        "orange is a juicy fruit",
        "grape is a juicy fruit",
        "the wild wolf hunts in the deep forest",
        "the wild lion hunts in the deep forest",
        "the friendly dog runs in the green park",
        "the friendly cat runs in the green park"
    ]

    words = []
    for s in sentences:
        words.extend(s.split())
    vocab = sorted(list(set(words)))
    word_to_id = {w: i for i, w in enumerate(vocab)}
    id_to_word = {i: w for i, w in enumerate(vocab)}
    V = len(vocab)
    print(f"\n>>> Vocabulary: {V} unique words.")

    # 2. Extract context pairs
    window_size = 2
    pairs = []
    for s in sentences:
        tokens = [word_to_id[w] for w in s.split()]
        for i, target in enumerate(tokens):
            for j in range(max(0, i - window_size), min(len(tokens), i + window_size + 1)):
                if i != j:
                    pairs.append((target, tokens[j]))

    # 3. Initialize coordinates (D = 2 for 2D map)
    D = 2
    random.seed(42)
    U = [[(random.random() - 0.5) * 0.2 for _ in range(D)] for _ in range(V)]
    W = [[(random.random() - 0.5) * 0.2 for _ in range(D)] for _ in range(V)]

    # 4. Train coordinates
    learning_rate = 0.05
    epochs = 1500
    print("\nTraining coordinates (pull neighbors, push strangers)...")
    for epoch in range(1, epochs + 1):
        total_loss = 0.0
        for target, context in pairs:
            # Positive neighbor
            score_pos = dot_product(U[target], W[context])
            prob_pos = sigmoid(score_pos)
            err_pos = 1.0 - prob_pos
            total_loss += -math.log(max(prob_pos, 1e-9))

            # Negative stranger
            neg = random.randint(0, V - 1)
            while neg == context:
                neg = random.randint(0, V - 1)
            score_neg = dot_product(U[target], W[neg])
            prob_neg = sigmoid(score_neg)
            err_neg = 0.0 - prob_neg
            total_loss += -math.log(max(1.0 - prob_neg, 1e-9))

            # Updates
            for d in range(D):
                u_val = U[target][d]
                w_ctx = W[context][d]
                w_neg = W[neg][d]
                U[target][d] += learning_rate * (err_pos * w_ctx + err_neg * w_neg)
                W[context][d] += learning_rate * (err_pos * u_val)
                W[neg][d] += learning_rate * (err_neg * u_val)

        if epoch % 500 == 0:
            print(f"   Epoch {epoch:4d} / {epochs} | Avg Loss: {total_loss / len(pairs):.4f}")

    # Combine coordinates
    coords = {w: [(U[word_to_id[w]][d] + W[word_to_id[w]][d]) / 2.0 for d in range(D)] for w in vocab}

    # 5. Visual 2D Grid
    xs = [c[0] for c in coords.values()]
    ys = [c[1] for c in coords.values()]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    grid_w, grid_h = 60, 20
    grid = [[" " for _ in range(grid_w)] for _ in range(grid_h)]

    for w, (x, y) in coords.items():
        gx = int((x - min_x) / (max_x - min_x + 1e-9) * (grid_w - 7))
        gy = int((y - min_y) / (max_y - min_y + 1e-9) * (grid_h - 1))
        gy = grid_h - 1 - gy
        label = w[:5]
        for k, ch in enumerate(label):
            if gx + k < grid_w and grid[gy][gx + k] == " ":
                grid[gy][gx + k] = ch

    print("\n" + "+" + "-" * grid_w + "+")
    for row in grid:
        print("|" + "".join(row) + "|")
    print("+" + "-" * grid_w + "+")

    # 6. Similarity Search
    print("\n>>> Closest words to 'king':")
    king_v = coords["king"]
    matches = sorted([(direction_alignment(king_v, coords[w]), w) for w in vocab if w != "king"], reverse=True)[:5]
    for sim, w in matches:
        print(f"    {w:12s} (Alignment: {sim:.4f})")

    print("\n>>> Closest words to 'apple':")
    apple_v = coords["apple"]
    matches = sorted([(direction_alignment(apple_v, coords[w]), w) for w in vocab if w != "apple"], reverse=True)[:5]
    for sim, w in matches:
        print(f"    {w:12s} (Alignment: {sim:.4f})")

    # 7. Analogy
    print("\n>>> Analogy Arithmetic: ('wolf' - 'wild' + 'friendly'):")
    target = [coords["wolf"][d] - coords["wild"][d] + coords["friendly"][d] for d in range(D)]
    matches = sorted([(direction_alignment(target, coords[w]), w) for w in vocab if w not in ["wolf", "wild", "friendly"]], reverse=True)[:4]
    for sim, w in matches:
        print(f"    {w:12s} (Alignment: {sim:.4f})")

    print("\n" + "=" * 80)
    print(" INDUSTRY MAPPING: In PyTorch, this is 'torch.nn.Embedding(V, D)'.")
    print(" In 2013, Google released this as 'Word2Vec'. Modern LLMs use D=4096.")
    print("=" * 80)

if __name__ == "__main__":
    main()

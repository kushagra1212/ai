import math
import random

# ====================================================================================
#  PROGRAM 32: THE FIRST NEURAL NEXT-WORD PREDICTOR (PYTHON SIDE-BY-SIDE)
#  Bengio 2003 Architecture: Embeddings -> Hidden Neurons -> Softmax Judges
# ====================================================================================

def leaky_relu(x):
    return x if x > 0.0 else 0.01 * x

def leaky_relu_deriv(x):
    return 1.0 if x > 0.0 else 0.01

def softmax(logits):
    max_val = max(logits)
    exp_vals = [math.exp(x - max_val) for x in logits]
    total = sum(exp_vals)
    return [x / total for x in exp_vals]

def main():
    print("=" * 80)
    print(" PROGRAM 32: THE FIRST NEURAL NEXT-WORD PREDICTOR (PYTHON)")
    print(" Bengio (2003) Neural Language Model from Scratch")
    print("=" * 80)

    sentences = [
        "the king sits on the golden throne .",
        "the queen sits on the golden throne .",
        "the king rules the royal castle .",
        "the queen rules the royal castle .",
        "the prince lives in the royal palace .",
        "the princess lives in the royal palace .",
        "the monkey eats a sweet apple .",
        "the monkey eats a sweet banana .",
        "the monkey eats a juicy orange .",
        "the monkey eats a juicy grape .",
        "apple is a sweet delicious fruit .",
        "banana is a sweet delicious fruit .",
        "orange is a juicy delicious fruit .",
        "grape is a juicy delicious fruit .",
        "the wild wolf hunts in the deep forest .",
        "the wild lion hunts in the deep forest .",
        "the friendly dog runs in the green park .",
        "the friendly cat runs in the green park .",
        "the little girl was happy in the castle .",
        "the brave knight rode through the deep forest ."
    ]

    words = []
    for s in sentences:
        words.extend(s.split())
    vocab = sorted(list(set(words)))
    w2i = {w: i for i, w in enumerate(vocab)}
    i2w = {i: w for i, w in enumerate(vocab)}
    V = len(vocab)
    print(f"\nCorpus: {len(sentences)} sentences | Vocabulary: {V} unique words")

    # Training trigram dataset: (w1, w2) -> target
    dataset = []
    for s in sentences:
        tokens = [w2i[w] for w in s.split()]
        for i in range(len(tokens) - 2):
            dataset.append((tokens[i], tokens[i+1], tokens[i+2]))

    # Architecture dimensions
    D = 8        # Coordinate dials per word
    H = 32       # Hidden neurons
    IN_DIM = 2 * D  # 2 words * 8 = 16

    random.seed(42)
    # Weights initialization
    C = [[random.gauss(0.0, 0.1) for _ in range(D)] for _ in range(V)]
    W1 = [[random.gauss(0.0, math.sqrt(2.0 / IN_DIM)) for _ in range(H)] for _ in range(IN_DIM)]
    B1 = [0.0] * H
    W2 = [[random.gauss(0.0, math.sqrt(2.0 / H)) for _ in range(V)] for _ in range(H)]
    B2 = [0.0] * V

    # Momentum buffers
    vC = [[0.0] * D for _ in range(V)]
    vW1 = [[0.0] * H for _ in range(IN_DIM)]
    vB1 = [0.0] * H
    vW2 = [[0.0] * V for _ in range(H)]
    vB2 = [0.0] * V

    lr = 0.03
    momentum = 0.85
    epochs = 600

    print("\nTraining Neural Language Model...")
    for epoch in range(1, epochs + 1):
        total_loss = 0.0
        correct = 0

        for w1, w2, target in dataset:
            # 1. Forward Pass
            x = C[w1] + C[w2]

            h_pre = [B1[j] + sum(x[i] * W1[i][j] for i in range(IN_DIM)) for j in range(H)]
            h = [leaky_relu(val) for val in h_pre]

            logits = [B2[k] + sum(h[j] * W2[j][k] for j in range(H)) for k in range(V)]
            probs = softmax(logits)

            total_loss += -math.log(max(probs[target], 1e-12))
            if probs.index(max(probs)) == target:
                correct += 1

            # 2. Backpropagation
            dLogits = list(probs)
            dLogits[target] -= 1.0

            dh = [sum(dLogits[k] * W2[j][k] for k in range(V)) for j in range(H)]
            dh_pre = [dh[j] * leaky_relu_deriv(h_pre[j]) for j in range(H)]
            dx = [sum(dh_pre[j] * W1[i][j] for j in range(H)) for i in range(IN_DIM)]

            # 3. Parameter Updates with Momentum
            for j in range(H):
                for k in range(V):
                    vW2[j][k] = momentum * vW2[j][k] - lr * (dLogits[k] * h[j])
                    W2[j][k] += vW2[j][k]
            for k in range(V):
                vB2[k] = momentum * vB2[k] - lr * dLogits[k]
                B2[k] += vB2[k]

            for i in range(IN_DIM):
                for j in range(H):
                    vW1[i][j] = momentum * vW1[i][j] - lr * (dh_pre[j] * x[i])
                    W1[i][j] += vW1[i][j]
            for j in range(H):
                vB1[j] = momentum * vB1[j] - lr * dh_pre[j]
                B1[j] += vB1[j]

            for d in range(D):
                vC[w1][d] = momentum * vC[w1][d] - lr * dx[d]
                C[w1][d] += vC[w1][d]
                vC[w2][d] = momentum * vC[w2][d] - lr * dx[D + d]
                C[w2][d] += vC[w2][d]

        if epoch % 150 == 0 or epoch == epochs:
            acc = 100.0 * correct / len(dataset)
            print(f"   Epoch {epoch:3d} / {epochs} | Loss: {total_loss/len(dataset):.4f} | Accuracy: {acc:.1f}%")

    print("\n" + "=" * 80)
    print(" LIVE PREDICTIONS (What comes next?):")
    print("=" * 80)

    def predict(w1, w2):
        id1, id2 = w2i[w1], w2i[w2]
        x = C[id1] + C[id2]
        h = [leaky_relu(B1[j] + sum(x[i] * W1[i][j] for i in range(IN_DIM))) for j in range(H)]
        logits = [B2[k] + sum(h[j] * W2[j][k] for j in range(H)) for k in range(V)]
        probs = softmax(logits)
        ranked = sorted([(probs[k], i2w[k]) for k in range(V)], reverse=True)[:3]
        print(f"Context: [\"{w1}\", \"{w2}\"] -> Top Next Words: {[(w, f'{p*100:.1f}%') for p, w in ranked]}")

    predict("the", "king")
    predict("sweet", "delicious")
    predict("wild", "wolf")
    predict("friendly", "dog")

    print("\n" + "=" * 80)
    print(" AUTONOMOUS GENERATION (Autoregressive loop):")
    print("=" * 80)
    def generate(w1, w2, steps=10):
        story = [w1, w2]
        for _ in range(steps):
            if w1 not in w2i or w2 not in w2i: break
            x = C[w2i[w1]] + C[w2i[w2]]
            h = [leaky_relu(B1[j] + sum(x[i] * W1[i][j] for i in range(IN_DIM))) for j in range(H)]
            logits = [B2[k] + sum(h[j] * W2[j][k] for j in range(H)) for k in range(V)]
            probs = softmax(logits)
            next_w = i2w[probs.index(max(probs))]
            story.append(next_w)
            w1, w2 = w2, next_w
            if next_w == ".": break
        print("Story:", " ".join(story))

    generate("the", "queen")
    generate("the", "monkey")
    generate("the", "wild")

if __name__ == "__main__":
    main()

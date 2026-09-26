import math
import random
import re

# ====================================================================================
#  PROGRAM 31: SEMANTIC SEARCH ENGINE (PYTHON SIDE-BY-SIDE)
#  Built from first principles (Vector Search vs Keyword Search)
# ====================================================================================

def sigmoid(x):
    if x > 10.0: return 1.0
    if x < -10.0: return 0.0
    return 1.0 / (1.0 + math.exp(-x))

def dot_product(a, b):
    return sum(x * y for x, y in zip(a, b))

def vector_magnitude(v):
    return math.sqrt(sum(x * x for x in v))

def cosine_alignment(a, b):
    mag_a = vector_magnitude(a)
    mag_b = vector_magnitude(b)
    if mag_a < 1e-9 or mag_b < 1e-9:
        return 0.0
    return dot_product(a, b) / (mag_a * mag_b)

def split_words(text):
    return re.findall(r'\b[a-zA-Z0-9]+\b', text.lower())

def main():
    print("=" * 80)
    print(" PROGRAM 31: FIRST-PRINCIPLES SEMANTIC SEARCH ENGINE (PYTHON)")
    print("=" * 80)

    # 1. Document Corpus
    documents = [
        {"id": 0, "text": "the king sits on the golden throne", "cat": "Royalty"},
        {"id": 1, "text": "the queen rules the royal castle", "cat": "Royalty"},
        {"id": 2, "text": "the prince rules the royal palace", "cat": "Royalty"},
        {"id": 3, "text": "the princess lives in the royal palace", "cat": "Royalty"},
        {"id": 4, "text": "the monkey eats a sweet apple in the sun", "cat": "Fruit & Nature"},
        {"id": 5, "text": "the monkey eats a sweet banana in the tree", "cat": "Fruit & Nature"},
        {"id": 6, "text": "the juicy orange is a delicious sweet fruit", "cat": "Fruit & Nature"},
        {"id": 7, "text": "the wild wolf hunts in the deep forest", "cat": "Wild Animals"},
        {"id": 8, "text": "the wild lion hunts in the deep forest", "cat": "Wild Animals"},
        {"id": 9, "text": "the friendly dog runs in the green park", "cat": "Friendly Pets"},
        {"id": 10, "text": "the friendly cat runs in the green park", "cat": "Friendly Pets"}
    ]

    # Extract vocabulary
    vocab = sorted(list(set(w for doc in documents for w in split_words(doc["text"]))))
    w2i = {w: i for i, w in enumerate(vocab)}
    V = len(vocab)
    D = 8

    # Extract neighbor pairs
    window = 2
    pairs = []
    for doc in documents:
        tokens = [w2i[w] for w in split_words(doc["text"])]
        for i, target in enumerate(tokens):
            for j in range(max(0, i - window), min(len(tokens), i + window + 1)):
                if i != j:
                    pairs.append((target, tokens[j]))

    # Train word coordinates (Pull neighbors, push strangers)
    random.seed(42)
    U = [[(random.random() - 0.5) * 0.2 for _ in range(D)] for _ in range(V)]
    W = [[(random.random() - 0.5) * 0.2 for _ in range(D)] for _ in range(V)]

    lr = 0.05
    for epoch in range(1500):
        for target, context in pairs:
            # Positive
            p_pos = sigmoid(dot_product(U[target], W[context]))
            err_pos = 1.0 - p_pos

            # Negative
            neg = random.randint(0, V - 1)
            while neg == context: neg = random.randint(0, V - 1)
            p_neg = sigmoid(dot_product(U[target], W[neg]))
            err_neg = 0.0 - p_neg

            for d in range(D):
                u_val = U[target][d]
                w_c = W[context][d]
                w_n = W[neg][d]
                U[target][d] += lr * (err_pos * w_c + err_neg * w_n)
                W[context][d] += lr * (err_pos * u_val)
                W[neg][d]     += lr * (err_neg * u_val)

    # Word coordinates
    word_coords = {w: [(U[w2i[w]][d] + W[w2i[w]][d]) / 2.0 for d in range(D)] for w in vocab}

    # Document vectors (average of word coordinates)
    for doc in documents:
        tokens = split_words(doc["text"])
        doc_vec = [0.0] * D
        for t in tokens:
            for d in range(D):
                doc_vec[d] += word_coords[t][d]
        doc["vector"] = [x / len(tokens) for x in doc_vec]

    # Search Query function
    def search(query):
        q_tokens = split_words(query)
        q_vec = [0.0] * D
        valid = [t for t in q_tokens if t in word_coords]
        if not valid:
            return []
        for t in valid:
            for d in range(D):
                q_vec[d] += word_coords[t][d]
        q_vec = [x / len(valid) for x in q_vec]

        ranked = []
        for doc in documents:
            sem_score = cosine_alignment(q_vec, doc["vector"])
            # Keyword match check
            kw_match = any(qt in split_words(doc["text"]) for qt in q_tokens)
            ranked.append({
                "text": doc["text"],
                "cat": doc["cat"],
                "score": sem_score,
                "kw_match": kw_match
            })
        ranked.sort(key=lambda x: x["score"], reverse=True)
        return ranked

    print(f"\nDatabase ready with {len(documents)} indexed documents.\n")

    test_queries = ["king throne", "sweet fruit", "wild wolf", "friendly dog"]
    for q in test_queries:
        print(f"SEARCH QUERY: \"{q}\"")
        results = search(q)[:3]
        for rank, r in enumerate(results, 1):
            kw_status = "[KW: MATCHED]" if r["kw_match"] else "[KW: NO MATCH]"
            print(f"   #{rank} | {r['score']*100:5.1f}% Match | {kw_status:15s} | \"{r['text']}\"")
        print()

    print("=" * 80)
    print(" INDUSTRY MAPPING: This is the exact algorithm inside:")
    print(" 1. Vector Databases (Pinecone, ChromaDB, Weaviate)")
    print(" 2. RAG Pipelines (Retrieval-Augmented Generation)")
    print(" 3. Modern Semantic Search Engines")
    print("=" * 80)

if __name__ == "__main__":
    main()

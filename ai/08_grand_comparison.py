# Python side-by-side equivalent: The Grand Arena
import math
import random
from typing import List, Dict, Any

NUM_PIXELS = 64

def squash(z: float) -> float:
    z = max(-50.0, min(50.0, z))
    return 1.0 / (1.0 + math.exp(-z))

def mini_bar(error: float) -> str:
    acc = max(0.0, 1.0 - error * 2.0)
    filled = min(10, int(acc * 10))
    return "[" + "#" * filled + "." * (10 - filled) + "]"

def compute_avg_error(dials: List[float], baseline: float, dataset: List[Dict[str, Any]]) -> float:
    total = sum(abs(squash(baseline + sum(p * d for p, d in zip(s["pixels"], dials))) - s["target"]) for s in dataset)
    return total / len(dataset)

def run_grand_arena():
    random.seed(42)

    # Dataset
    one_center = [0.0] * 64
    for r in range(8): one_center[r * 8 + 3] = 1.0
    zero_box = [0.0] * 64
    for i in range(8):
        zero_box[0 * 8 + i] = 1.0; zero_box[7 * 8 + i] = 1.0
        zero_box[i * 8 + 0] = 1.0; zero_box[i * 8 + 7] = 1.0

    dataset = [
        {"pixels": one_center, "target": 1.0},
        {"pixels": zero_box, "target": 0.0}
    ]

    # Method 1
    m1_d = [0.0] * 64; m1_b = 0.0

    # Method 2
    m2_d = [0.0] * 64; m2_b = 0.0
    m2_vd = [0.0] * 64; m2_vb = 0.0

    # Method 3
    class Org:
        def __init__(self):
            self.d = [random.uniform(-0.5, 0.5) for _ in range(64)]
            self.b = random.uniform(-0.5, 0.5)
            self.err = 999.0
    population = [Org() for _ in range(100)]

    print("-" * 84)
    print(" Round | Method 1: Basic Fixed   | Method 2: Momentum+Decay | Method 3: Evolution   ")
    print("       | Error     Progress      | Error     Progress       | Best Error  Progress  ")
    print("-" * 84)

    for round_num in range(50):
        # M1
        for s in dataset:
            raw = m1_b + sum(p * d for p, d in zip(s["pixels"], m1_d))
            diff = squash(raw) - s["target"]
            m1_b -= 0.15 * diff
            for i in range(64):
                if s["pixels"][i] > 0: m1_d[i] -= 0.15 * diff * s["pixels"][i]

        # M2
        step2 = 0.40 / (1.0 + 0.05 * round_num)
        for s in dataset:
            raw = m2_b + sum(p * d for p, d in zip(s["pixels"], m2_d))
            diff = squash(raw) - s["target"]
            m2_vb = 0.80 * m2_vb - step2 * diff
            m2_b += m2_vb
            for i in range(64):
                if s["pixels"][i] > 0:
                    m2_vd[i] = 0.80 * m2_vd[i] - step2 * diff * s["pixels"][i]
                    m2_d[i] += m2_vd[i]

        # M3
        for org in population:
            org.err = compute_avg_error(org.d, org.b, dataset)
        population.sort(key=lambda o: o.err)
        next_pop = population[:5]
        while len(next_pop) < 100:
            parent = random.choice(population[:5])
            child = Org()
            child.b = parent.b + (random.gauss(0, 0.15) if random.random() < 0.4 else 0.0)
            child.d = [d + (random.gauss(0, 0.15) if random.random() < 0.3 else 0.0) for d in parent.d]
            next_pop.append(child)
        population = next_pop

        if round_num in (0, 5, 10, 20, 35, 49):
            err1 = compute_avg_error(m1_d, m1_b, dataset)
            err2 = compute_avg_error(m2_d, m2_b, dataset)
            err3 = population[0].err
            print(f"  {round_num+1:3d}  | {err1:.4f}  {mini_bar(err1)}   | {err2:.4f}  {mini_bar(err2)}   | {err3:.4f}  {mini_bar(err3)}")

    print("-" * 84)

if __name__ == "__main__":
    run_grand_arena()

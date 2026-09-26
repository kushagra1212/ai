# Python side-by-side equivalent: Darwinian Evolution / Genetic Algorithm
import math
import random
from typing import List, Dict, Any

NUM_PIXELS = 64
POPULATION_SIZE = 100
TOP_SURVIVORS = 5

def squash(z: float) -> float:
    z = max(-50.0, min(50.0, z))
    return 1.0 / (1.0 + math.exp(-z))

def render_progress_bar(error: float) -> str:
    accuracy = max(0.0, 1.0 - error * 2.0)
    filled = min(20, int(accuracy * 20))
    return "[" + "#" * filled + "." * (20 - filled) + "]"

class Individual:
    def __init__(self):
        self.dials = [random.uniform(-0.5, 0.5) for _ in range(NUM_PIXELS)]
        self.baseline = random.uniform(-0.5, 0.5)
        self.error = 999.0

    def predict(self, pixels: List[float]) -> float:
        raw = self.baseline + sum(p * d for p, d in zip(pixels, self.dials))
        return squash(raw)

    def evaluate(self, dataset: List[Dict[str, Any]]):
        total = sum(abs(self.predict(s["pixels"]) - s["target"]) for s in dataset)
        self.error = total / len(dataset)

def train_evolution(dataset: List[Dict[str, Any]], generations=60, mutation_amount=0.15):
    random.seed(42)
    population = [Individual() for _ in range(POPULATION_SIZE)]

    print("-" * 80)
    print(" Generation | Best Error | Avg Error  | Worst Error | Champion Accuracy Bar")
    print("-" * 80)

    for gen in range(generations):
        for ind in population:
            ind.evaluate(dataset)

        population.sort(key=lambda x: x.error)

        best_err = population[0].error
        avg_err = sum(x.error for x in population) / POPULATION_SIZE
        worst_err = population[-1].error

        if gen % 5 == 0 or gen == generations - 1 or best_err < 0.01:
            print(f"   Gen {gen:3d}  |   {best_err:.4f}   |   {avg_err:.4f}   |   {worst_err:.4f}    | {render_progress_bar(best_err)}")

        if best_err < 0.005:
            print("-" * 80)
            print(f">>> Champion found at Generation {gen}! Error near zero without calculus! <<<\n")
            break

        # Next generation
        next_gen = population[:TOP_SURVIVORS] # Elitism
        while len(next_gen) < POPULATION_SIZE:
            parent = random.choice(population[:TOP_SURVIVORS])
            child = Individual()
            child.baseline = parent.baseline + (random.gauss(0, mutation_amount) if random.random() < 0.40 else 0.0)
            child.dials = [
                d + (random.gauss(0, mutation_amount) if random.random() < 0.30 else 0.0)
                for d in parent.dials
            ]
            next_gen.append(child)

        population = next_gen

    return population[0]

if __name__ == "__main__":
    one_center = [0.0] * 64
    for r in range(8): one_center[r * 8 + 3] = 1.0

    zero_box = [0.0] * 64
    for i in range(8):
        zero_box[0 * 8 + i] = 1.0; zero_box[7 * 8 + i] = 1.0
        zero_box[i * 8 + 0] = 1.0; zero_box[i * 8 + 7] = 1.0

    dataset = [
        {"pixels": one_center, "target": 1.0, "label": "1 (center)"},
        {"pixels": zero_box, "target": 0.0, "label": "0 (box)"}
    ]

    champion = train_evolution(dataset)
    print(f"Champion accuracy on 1: {champion.predict(one_center)*100:.1f}%")
    print(f"Champion accuracy on 0: {champion.predict(zero_box)*100:.1f}%")

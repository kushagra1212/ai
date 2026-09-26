# Python side-by-side equivalent: 5 Top Picks -> 19 Clones Each + Probability Mutation
import math
import random
from typing import List, Dict, Any

NUM_PIXELS = 64
TOP_PICKS = 5
COPIES_PER_PICK = 19  # 5 * 19 = 95 copies (+ 5 original = 100)
POPULATION_SIZE = TOP_PICKS + (TOP_PICKS * COPIES_PER_PICK)

def squash(z: float) -> float:
    z = max(-50.0, min(50.0, z))
    return 1.0 / (1.0 + math.exp(-z))

def render_progress_bar(error: float) -> str:
    acc = max(0.0, 1.0 - error * 2.0)
    filled = min(20, int(acc * 20))
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

def train_clone_mutate(dataset: List[Dict[str, Any]], generations=60, mutation_prob=0.25, mutation_amount=0.15):
    random.seed(42)
    population = [Individual() for _ in range(POPULATION_SIZE)]

    print("-" * 84)
    print(" Gen | Best Error | Avg Error  | Worst Error | Champion Accuracy Bar")
    print("-" * 84)

    for gen in range(generations):
        for ind in population:
            ind.evaluate(dataset)

        population.sort(key=lambda ind: ind.error)

        best_err = population[0].error
        avg_err = sum(x.error for x in population) / POPULATION_SIZE
        worst_err = population[-1].error

        if gen % 5 == 0 or gen == generations - 1 or best_err < 0.01:
            print(f" {gen:3d} |   {best_err:.4f}   |   {avg_err:.4f}   |   {worst_err:.4f}    | {render_progress_bar(best_err)}")

        if best_err < 0.005:
            print("-" * 84)
            print(f">>> Champion found at Generation {gen}! Goal achieved successfully! <<<\n")
            break

        # Next generation:
        # A) 5 Champions survive untouched
        next_gen = [population[i] for i in range(TOP_PICKS)]

        # B) 19 Clones per champion with probability mutation
        for p in range(TOP_PICKS):
            for _ in range(COPIES_PER_PICK):
                child = Individual()
                # Mutate baseline with given probability
                child.baseline = population[p].baseline + (random.gauss(0, mutation_amount) if random.random() < mutation_prob else 0.0)
                # Mutate each dial with given probability
                child.dials = [
                    d + (random.gauss(0, mutation_amount) if random.random() < mutation_prob else 0.0)
                    for d in population[p].dials
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

    champ = train_clone_mutate(dataset)
    print(f"Python Champion on 1: {champ.predict(one_center)*100:.1f}%")
    print(f"Python Champion on 0: {champ.predict(zero_box)*100:.1f}%")

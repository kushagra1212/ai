# Python side-by-side equivalent: Hybrid (Evolution 09 + Momentum/Decay 06)
import math
import random
from typing import List, Dict, Any

NUM_PIXELS = 64
TOP_PICKS = 5
COPIES_PER_PICK = 19
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
        self.v_dials = [0.0] * NUM_PIXELS
        self.v_baseline = 0.0
        self.error = 999.0

    def predict(self, pixels: List[float]) -> float:
        raw = self.baseline + sum(p * d for p, d in zip(pixels, self.dials))
        return squash(raw)

    def evaluate(self, dataset: List[Dict[str, Any]]):
        self.error = sum(abs(self.predict(s["pixels"]) - s["target"]) for s in dataset) / len(dataset)

    def polish_with_momentum(self, dataset: List[Dict[str, Any]], step_size: float, friction: float):
        for s in dataset:
            diff = self.predict(s["pixels"]) - s["target"]
            self.v_baseline = friction * self.v_baseline - step_size * diff
            self.baseline += self.v_baseline
            for i in range(NUM_PIXELS):
                if s["pixels"][i] > 0.0:
                    self.v_dials[i] = friction * self.v_dials[i] - step_size * diff * s["pixels"][i]
                    self.dials[i] += self.v_dials[i]

def train_hybrid(dataset: List[Dict[str, Any]], max_gens=30, initial_step=0.40, decay=0.05, friction=0.80, mutate_prob=0.20, mutate_amount=0.10):
    random.seed(42)
    population = [Individual() for _ in range(POPULATION_SIZE)]

    print("-" * 84)
    print(" Gen | Step Size | Best Error | Avg Error  | Worst Error | Champion Accuracy Bar")
    print("-" * 84)

    for gen in range(max_gens):
        current_step = initial_step / (1.0 + decay * gen)

        # 1. Momentum polishing for each individual
        for ind in population:
            ind.polish_with_momentum(dataset, current_step, friction)
            ind.evaluate(dataset)

        # 2. Sort by fitness
        population.sort(key=lambda ind: ind.error)

        best_err = population[0].error
        worst_err = population[-1].error
        avg_err = sum(x.error for x in population) / POPULATION_SIZE

        print(f" {gen:3d} |  {current_step:.4f}   |   {best_err:.4f}   |   {avg_err:.4f}   |   {worst_err:.4f}    | {render_progress_bar(best_err)}")

        if best_err < 0.002:
            print("-" * 84)
            print(f">>> Hybrid Champion converged at Generation {gen}! Near-perfect accuracy! <<<\n")
            break

        # 3. Clonal reproduction (5 champions + 19 clones each)
        next_gen = [population[i] for i in range(TOP_PICKS)]
        for p in range(TOP_PICKS):
            for _ in range(COPIES_PER_PICK):
                child = Individual()
                # Clone dials AND velocities
                child.baseline = population[p].baseline + (random.gauss(0, mutate_amount) if random.random() < mutate_prob else 0.0)
                child.v_baseline = population[p].v_baseline
                child.v_dials = list(population[p].v_dials)
                child.dials = [
                    d + (random.gauss(0, mutate_amount) if random.random() < mutate_prob else 0.0)
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

    champ = train_hybrid(dataset)
    print(f"Hybrid Python Champion on 1: {champ.predict(one_center)*100:.1f}%")
    print(f"Hybrid Python Champion on 0: {champ.predict(zero_box)*100:.1f}%")

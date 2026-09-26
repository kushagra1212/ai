# Python side-by-side equivalent: 2-Layer Brain with Leaky ReLU
import math
import random
from typing import List, Dict, Any

def squash(z: float) -> float:
    z = max(-50.0, min(50.0, z))
    return 1.0 / (1.0 + math.exp(-z))

def squash_slope(a: float) -> float:
    return a * (1.0 - a)

def leaky_ramp(z: float, alpha: float = 0.01) -> float:
    return z if z > 0.0 else alpha * z

def leaky_ramp_slope(z: float, alpha: float = 0.01) -> float:
    return 1.0 if z > 0.0 else alpha

NUM_PIXELS = 64
NUM_HIDDEN = 4

class LeakyRampDigitBrain:
    def __init__(self):
        random.seed(42)
        self.dials_layer1 = [[random.uniform(-0.2, 0.2) for _ in range(NUM_HIDDEN)] for _ in range(NUM_PIXELS)]
        self.baseline_layer1 = [0.0] * NUM_HIDDEN
        self.dials_layer2 = [random.uniform(-0.2, 0.2) for _ in range(NUM_HIDDEN)]
        self.baseline_layer2 = 0.0

    def forward(self, pixels: List[float]):
        raw_layer1 = [
            self.baseline_layer1[h] + sum(pixels[i] * self.dials_layer1[i][h] for i in range(NUM_PIXELS))
            for h in range(NUM_HIDDEN)
        ]
        hidden_acts = [leaky_ramp(z, 0.01) for z in raw_layer1]

        raw_judge = self.baseline_layer2 + sum(hidden_acts[h] * self.dials_layer2[h] for h in range(NUM_HIDDEN))
        final_guess = squash(raw_judge)
        return final_guess, raw_layer1, hidden_acts

    def train(self, dataset: List[Dict[str, Any]], passes: int = 200, step_size: float = 0.25):
        for pass_num in range(passes):
            total_err = 0.0
            for sample in dataset:
                final_guess, raw_layer1, hidden_acts = self.forward(sample["pixels"])
                err = final_guess - sample["target"]
                total_err += abs(err)

                judge_blame = err * squash_slope(final_guess)
                assistant_blame = [
                    (judge_blame * self.dials_layer2[h]) * leaky_ramp_slope(raw_layer1[h], 0.01)
                    for h in range(NUM_HIDDEN)
                ]

                # Update Judge
                self.baseline_layer2 -= step_size * judge_blame
                for h in range(NUM_HIDDEN):
                    self.dials_layer2[h] -= step_size * judge_blame * hidden_acts[h]

                # Update Assistants
                for h in range(NUM_HIDDEN):
                    self.baseline_layer1[h] -= step_size * assistant_blame[h]
                    for i in range(NUM_PIXELS):
                        self.dials_layer1[i][h] -= step_size * assistant_blame[h] * sample["pixels"][i]

if __name__ == "__main__":
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

    brain = LeakyRampDigitBrain()
    brain.train(dataset)
    pred, _, _ = brain.forward(one_center)
    print(f"Python Leaky ReLU Brain on 1: {pred*100:.1f}%")

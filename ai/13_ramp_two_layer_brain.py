# Python side-by-side equivalent: 2-Layer Brain with The Ramp (ReLU) in Layer 1
import math
import random
from typing import List, Dict, Any

def squash(z: float) -> float:
    z = max(-50.0, min(50.0, z))
    return 1.0 / (1.0 + math.exp(-z))

def squash_slope(a: float) -> float:
    return a * (1.0 - a)

# The Ramp (ReLU)
def ramp(z: float) -> float:
    return z if z > 0.0 else 0.0

# The Slope of the Ramp
def ramp_slope(z: float) -> float:
    return 1.0 if z > 0.0 else 0.0

class RampTwoLayerXORBrain:
    def __init__(self):
        random.seed(42)
        # 2 inputs -> 2 assistants
        self.dials_layer1 = [[0.6, -0.6], [-0.6, 0.6]]
        self.baseline_layer1 = [0.0, 0.0]
        # 2 assistants -> 1 judge
        self.dials_layer2 = [1.0, 1.0]
        self.baseline_layer2 = -0.5

    def forward(self, x: List[float]):
        # Layer 1: Assistants use THE RAMP
        raw_h = [
            self.baseline_layer1[h] + x[0]*self.dials_layer1[0][h] + x[1]*self.dials_layer1[1][h]
            for h in range(2)
        ]
        hidden_acts = [ramp(z) for z in raw_h]

        # Layer 2: Final Judge uses S-CURVE
        raw_judge = self.baseline_layer2 + hidden_acts[0]*self.dials_layer2[0] + hidden_acts[1]*self.dials_layer2[1]
        final_guess = squash(raw_judge)

        return final_guess, raw_h, hidden_acts

    def train(self, dataset, passes=1000, step_size=0.3):
        for _ in range(passes):
            for x, y in dataset:
                final_guess, raw_h, hidden_acts = self.forward(x)

                # Judge blame
                judge_error = final_guess - y
                judge_blame = judge_error * squash_slope(final_guess)

                # Assistants blame (using ramp_slope: 1.0 or 0.0!)
                assistant_blame = [
                    (judge_blame * self.dials_layer2[h]) * ramp_slope(raw_h[h])
                    for h in range(2)
                ]

                # Update Judge
                self.baseline_layer2 -= step_size * judge_blame
                for h in range(2):
                    self.dials_layer2[h] -= step_size * judge_blame * hidden_acts[h]

                # Update Assistants
                for h in range(2):
                    self.baseline_layer1[h] -= step_size * assistant_blame[h]
                    for i in range(2):
                        self.dials_layer1[i][h] -= step_size * assistant_blame[h] * x[i]

if __name__ == "__main__":
    xor_data = [
        ([0.0, 0.0], 0.0),
        ([1.0, 0.0], 1.0),
        ([0.0, 1.0], 1.0),
        ([1.0, 1.0], 0.0)
    ]

    brain = RampTwoLayerXORBrain()
    brain.train(xor_data, passes=1000, step_size=0.3)

    print("Python Ramp (ReLU) Brain on XOR:")
    for x, y in xor_data:
        pred, _, _ = brain.forward(x)
        print(f"  Inputs: {x} -> Target: {y} | Prediction: {pred*100:.1f}%")

# Python side-by-side equivalent: 2-Layer Brain with Backpropagation from Scratch
import math
import random
from typing import List, Dict, Any

def squash(z: float) -> float:
    z = max(-50.0, min(50.0, z))
    return 1.0 / (1.0 + math.exp(-z))

def squash_slope(a: float) -> float:
    return a * (1.0 - a)

# Experiment 1: The XOR Problem
class TwoLayerXORBrain:
    def __init__(self):
        random.seed(42)
        # 2 inputs -> 2 hidden assistants
        self.w1 = [[random.uniform(-0.5, 0.5) for _ in range(2)] for _ in range(2)]
        self.b1 = [0.0, 0.0]
        # 2 hidden -> 1 judge
        self.w2 = [random.uniform(-0.5, 0.5) for _ in range(2)]
        self.b2 = 0.0

    def forward(self, x: List[float]):
        # Layer 1
        h = [squash(self.b1[j] + x[0]*self.w1[0][j] + x[1]*self.w1[1][j]) for j in range(2)]
        # Layer 2
        out = squash(self.b2 + h[0]*self.w2[0] + h[1]*self.w2[1])
        return out, h

    def train(self, dataset, passes=5000, lr=0.5):
        for _ in range(passes):
            for x, y in dataset:
                out, h = self.forward(x)
                err = out - y
                judge_blame = err * squash_slope(out)

                # Send blame backward to assistants
                asst_blame = [(judge_blame * self.w2[j]) * squash_slope(h[j]) for j in range(2)]

                # Update Judge
                self.b2 -= lr * judge_blame
                for j in range(2):
                    self.w2[j] -= lr * judge_blame * h[j]

                # Update Assistants
                for j in range(2):
                    self.b1[j] -= lr * asst_blame[j]
                    for i in range(2):
                        self.w1[i][j] -= lr * asst_blame[j] * x[i]

if __name__ == "__main__":
    xor_data = [
        ([0.0, 0.0], 0.0),
        ([1.0, 0.0], 1.0),
        ([0.0, 1.0], 1.0),
        ([1.0, 1.0], 0.0)
    ]

    brain = TwoLayerXORBrain()
    brain.train(xor_data, passes=5000, lr=0.5)

    print("Python 2-Layer Brain on XOR:")
    for x, y in xor_data:
        pred, _ = brain.forward(x)
        print(f"  Inputs: {x} -> Target: {y} | Prediction: {pred*100:.1f}%")

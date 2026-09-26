# Python side-by-side equivalent: Momentum + Decay
import math
from typing import List, Dict, Any

GRID_SIZE = 8
NUM_PIXELS = 64

def squash(z: float) -> float:
    z = max(-50.0, min(50.0, z))
    return 1.0 / (1.0 + math.exp(-z))

def render_progress_bar(error: float) -> str:
    accuracy = max(0.0, 1.0 - error * 2.0)
    filled = min(20, int(accuracy * 20))
    return "[" + "#" * filled + "." * (20 - filled) + "]"

class MomentumDecayLearner:
    def __init__(self):
        self.dials = [0.0] * NUM_PIXELS
        self.baseline = 0.0
        self.v_dials = [0.0] * NUM_PIXELS
        self.v_baseline = 0.0

    def predict(self, pixels: List[float]) -> float:
        raw = self.baseline + sum(p * d for p, d in zip(pixels, self.dials))
        return squash(raw)

    def train(self, dataset: List[Dict[str, Any]], total_passes=60, initial_step=0.40, decay=0.05, friction=0.80):
        print("-" * 80)
        print(" Pass | Current Step | Avg Speed | Avg Error | Accuracy Progress Bar")
        print("-" * 80)

        for p in range(total_passes):
            current_step = initial_step / (1.0 + decay * p)
            total_error = 0.0

            for s in dataset:
                diff = self.predict(s["pixels"]) - s["target"]
                total_error += abs(diff)

                self.v_baseline = friction * self.v_baseline - current_step * diff
                self.baseline += self.v_baseline

                for i in range(NUM_PIXELS):
                    if s["pixels"][i] > 0.0:
                        self.v_dials[i] = friction * self.v_dials[i] - current_step * diff * s["pixels"][i]
                        self.dials[i] += self.v_dials[i]

            avg_error = total_error / len(dataset)
            avg_speed = (abs(self.v_baseline) + sum(abs(v) for v in self.v_dials)) / (NUM_PIXELS + 1)

            if p % 5 == 0 or p == total_passes - 1 or avg_error < 0.001:
                print(f"  {p:3d} | {current_step:.4f}       | {avg_speed:.4f}    | {avg_error:.4f}    | {render_progress_bar(avg_error)}")

            if avg_error < 0.0005 and p >= 15:
                print("-" * 80)
                print(f">>> Converged early at Pass {p}! Error reached near zero. <<<\n")
                break

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

    model = MomentumDecayLearner()
    model.train(dataset)

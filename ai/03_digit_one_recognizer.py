# Python side-by-side equivalent
import math
from typing import List, Dict, Any

GRID_SIZE = 8
NUM_PIXELS = GRID_SIZE * GRID_SIZE

def resize_to_8x8(patch: List[List[int]]) -> List[float]:
    result = [0.0] * NUM_PIXELS
    h = len(patch)
    w = len(patch[0])

    for r in range(GRID_SIZE):
        for c in range(GRID_SIZE):
            orig_r = min(int((r * h) / GRID_SIZE), h - 1)
            orig_c = min(int((c * w) / GRID_SIZE), w - 1)
            result[r * GRID_SIZE + c] = 1.0 if patch[orig_r][orig_c] == 1 else 0.0
    return result

def squash_to_confidence(raw_score: float) -> float:
    # 1 / (1 + e^-z)
    # Clip to avoid overflow
    raw_score = max(-50.0, min(50.0, raw_score))
    return 1.0 / (1.0 + math.exp(-raw_score))

class DigitOneRecognizer:
    def __init__(self):
        self.dials = [0.0] * NUM_PIXELS
        self.baseline = 0.0

    def predict(self, pixels: List[float]) -> float:
        raw_score = self.baseline
        for p, d in zip(pixels, self.dials):
            raw_score += p * d
        return squash_to_confidence(raw_score)

    def train(self, dataset: List[Dict[str, Any]], passes: int = 300, step_size: float = 0.15):
        for pass_num in range(passes):
            total_diff = 0.0
            for sample in dataset:
                guess = self.predict(sample["pixels"])
                diff = guess - sample["target"]
                total_diff += abs(diff)

                # Nudge baseline
                self.baseline -= step_size * diff

                # Nudge dials
                for i in range(NUM_PIXELS):
                    if sample["pixels"][i] > 0.0:
                        self.dials[i] -= step_size * diff * sample["pixels"][i]

            if pass_num in (0, 100, 200, passes - 1):
                print(f"Pass {pass_num:3d} | Average Error: {total_diff / len(dataset):.4f}")

    def print_dials(self):
        print("\n--- Learned Dials (Positive = 'looks like 1', Negative = 'looks like NOT 1') ---")
        for r in range(GRID_SIZE):
            row_str = ""
            for c in range(GRID_SIZE):
                val = self.dials[r * GRID_SIZE + c]
                if val > 0.8:
                    row_str += " +++ "
                elif val > 0.2:
                    row_str += "  +  "
                elif val < -0.8:
                    row_str += " --- "
                elif val < -0.2:
                    row_str += "  -  "
                else:
                    row_str += "  .  "
            print(row_str)
        print(f"Baseline dial value: {self.baseline:.4f}\n")

if __name__ == "__main__":
    dataset = [
        {"pixels": resize_to_8x8([[0,1,0],[0,1,0],[0,1,0],[0,1,0],[0,1,0]]), "target": 1.0},
        {"pixels": resize_to_8x8([[1,1,0],[0,1,0],[0,1,0],[0,1,0],[0,1,0]]), "target": 1.0},
        {"pixels": resize_to_8x8([[1,1,0],[0,1,0],[0,1,0],[0,1,0],[1,1,1]]), "target": 1.0},
        {"pixels": resize_to_8x8([[1,1,1,0],[0,1,1,0],[0,1,1,0],[0,1,1,0],[1,1,1,1]]), "target": 1.0},
        {"pixels": resize_to_8x8([[1,1,1],[1,0,1],[1,0,1],[1,0,1],[1,1,1]]), "target": 0.0},
        {"pixels": resize_to_8x8([[0,0,1,0,0],[0,0,1,0,0],[1,1,1,1,1],[0,0,1,0,0],[0,0,1,0,0]]), "target": 0.0},
        {"pixels": resize_to_8x8([[1,0,0,0],[1,0,0,0],[1,0,0,0],[1,1,1,1]]), "target": 0.0},
        {"pixels": resize_to_8x8([[0,0,0,0,0],[0,0,0,0,0],[1,1,1,1,1],[0,0,0,0,0],[0,0,0,0,0]]), "target": 0.0},
    ]

    model = DigitOneRecognizer()
    print("Starting Training (Tuning the dials)...")
    model.train(dataset, passes=300, step_size=0.15)
    model.print_dials()

    # Test Island #5 from Project 2
    island_5 = [
        [0, 1, 1, 0],
        [0, 0, 1, 0],
        [0, 0, 1, 0],
        [0, 0, 1, 0],
        [1, 1, 1, 1]
    ]
    conf_5 = model.predict(resize_to_8x8(island_5))
    print(f"Island #5 Confidence: {conf_5 * 100:.1f}% ===> {'Confirmed 1!' if conf_5 >= 0.7 else 'Not 1'}")

    # Test Mystery 'U' shape
    shape_u = [
        [1, 0, 0, 1],
        [1, 0, 0, 1],
        [1, 0, 0, 1],
        [1, 1, 1, 1]
    ]
    conf_u = model.predict(resize_to_8x8(shape_u))
    print(f"Mystery 'U' Shape Confidence: {conf_u * 100:.1f}% ===> {'Confirmed 1!' if conf_u >= 0.7 else 'Rejected (Not 1)'}")

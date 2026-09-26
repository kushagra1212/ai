# Python side-by-side equivalent: saving and loading weights
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
    raw_score = max(-50.0, min(50.0, raw_score))
    return 1.0 / (1.0 + math.exp(-raw_score))

class DigitOneRecognizer:
    def __init__(self):
        self.dials = [0.0] * NUM_PIXELS
        self.baseline = 0.0

    def predict(self, pixels: List[float]) -> float:
        raw_score = self.baseline + sum(p * d for p, d in zip(pixels, self.dials))
        return squash_to_confidence(raw_score)

    def train(self, dataset: List[Dict[str, Any]], passes: int = 300, step_size: float = 0.15):
        for _ in range(passes):
            for sample in dataset:
                guess = self.predict(sample["pixels"])
                diff = guess - sample["target"]
                self.baseline -= step_size * diff
                for i in range(NUM_PIXELS):
                    if sample["pixels"][i] > 0.0:
                        self.dials[i] -= step_size * diff * sample["pixels"][i]

    def save_to_file(self, filepath: str):
        with open(filepath, "w") as f:
            f.write(f"{self.baseline}\n")
            for i in range(NUM_PIXELS):
                f.write(f"{self.dials[i]}" + ("\n" if i % 8 == 7 else " "))

    def load_from_file(self, filepath: str):
        with open(filepath, "r") as f:
            lines = f.read().split()
            self.baseline = float(lines[0])
            self.dials = [float(x) for x in lines[1:NUM_PIXELS + 1]]

if __name__ == "__main__":
    dataset = [
        {"pixels": resize_to_8x8([[0,1,0],[0,1,0],[0,1,0],[0,1,0],[0,1,0]]), "target": 1.0},
        {"pixels": resize_to_8x8([[1,1,0],[0,1,0],[0,1,0],[0,1,0],[0,1,0]]), "target": 1.0},
        {"pixels": resize_to_8x8([[1,1,0],[0,1,0],[0,1,0],[0,1,0],[1,1,1]]), "target": 1.0},
        {"pixels": resize_to_8x8([[1,1,1],[1,0,1],[1,0,1],[1,0,1],[1,1,1]]), "target": 0.0},
        {"pixels": resize_to_8x8([[1,0,0,0],[1,0,0,0],[1,0,0,0],[1,1,1,1]]), "target": 0.0},
        {"pixels": resize_to_8x8([[0,0,1,0,0],[0,0,1,0,0],[1,1,1,1,1],[0,0,1,0,0],[0,0,1,0,0]]), "target": 0.0}
    ]

    # Train and save
    model = DigitOneRecognizer()
    model.train(dataset, passes=300, step_size=0.15)
    model.save_to_file("weights_python.txt")
    print("Python model trained and saved to weights_python.txt")

    # Load into a blank model
    fresh_model = DigitOneRecognizer()
    fresh_model.load_from_file("weights_python.txt")

    island_5 = resize_to_8x8([
        [0, 1, 1, 0],
        [0, 0, 1, 0],
        [0, 0, 1, 0],
        [0, 0, 1, 0],
        [1, 1, 1, 1]
    ])
    conf = fresh_model.predict(island_5)
    print(f"Loaded Python Model Confidence on Island #5: {conf * 100:.1f}%")

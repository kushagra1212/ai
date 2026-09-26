# Python side-by-side equivalent: comparing training methods
import math

NUM_PIXELS = 64

def squash(z):
    z = max(-50.0, min(50.0, z))
    return 1.0 / (1.0 + math.exp(-z))

def eval_error(dials, baseline, dataset):
    total = 0.0
    for s in dataset:
        raw = baseline + sum(p * d for p, d in zip(s["pixels"], dials))
        total += abs(squash(raw) - s["target"])
    return total / len(dataset)

# Method A: Fixed Step Size
def train_fixed(dataset, passes=50, step_size=0.15):
    dials = [0.0] * NUM_PIXELS
    baseline = 0.0
    for _ in range(passes):
        for s in dataset:
            raw = baseline + sum(p * d for p, d in zip(s["pixels"], dials))
            diff = squash(raw) - s["target"]
            baseline -= step_size * diff
            for i in range(NUM_PIXELS):
                if s["pixels"][i] > 0.0:
                    dials[i] -= step_size * diff * s["pixels"][i]
    return dials, baseline

# Method B: Decaying Step Size
def train_decay(dataset, passes=50, initial_step=0.40):
    dials = [0.0] * NUM_PIXELS
    baseline = 0.0
    for p in range(passes):
        current_step = initial_step / (1.0 + 0.05 * p)
        for s in dataset:
            raw = baseline + sum(p * d for p, d in zip(s["pixels"], dials))
            diff = squash(raw) - s["target"]
            baseline -= current_step * diff
            for i in range(NUM_PIXELS):
                if s["pixels"][i] > 0.0:
                    dials[i] -= current_step * diff * s["pixels"][i]
    return dials, baseline

# Method C: Momentum
def train_momentum(dataset, passes=50, step_size=0.15, friction=0.85):
    dials = [0.0] * NUM_PIXELS
    baseline = 0.0
    v_dials = [0.0] * NUM_PIXELS
    v_base = 0.0
    for _ in range(passes):
        for s in dataset:
            raw = baseline + sum(p * d for p, d in zip(s["pixels"], dials))
            diff = squash(raw) - s["target"]
            v_base = friction * v_base - step_size * diff
            baseline += v_base
            for i in range(NUM_PIXELS):
                if s["pixels"][i] > 0.0:
                    v_dials[i] = friction * v_dials[i] - step_size * diff * s["pixels"][i]
                    dials[i] += v_dials[i]
    return dials, baseline

if __name__ == "__main__":
    # 1: vertical line
    one = [0.0] * 64
    for r in range(8): one[r * 8 + 4] = 1.0

    # 0: border box
    zero = [0.0] * 64
    for i in range(8):
        zero[0 * 8 + i] = 1.0; zero[7 * 8 + i] = 1.0
        zero[i * 8 + 0] = 1.0; zero[i * 8 + 7] = 1.0

    dataset = [{"pixels": one, "target": 1.0}, {"pixels": zero, "target": 0.0}]

    d_a, b_a = train_fixed(dataset, passes=50)
    print(f"Python Method A (Fixed):    Error = {eval_error(d_a, b_a, dataset):.5f}")

    d_b, b_b = train_decay(dataset, passes=50)
    print(f"Python Method B (Decay):    Error = {eval_error(d_b, b_b, dataset):.5f}")

    d_c, b_c = train_momentum(dataset, passes=50)
    print(f"Python Method C (Momentum): Error = {eval_error(d_c, b_c, dataset):.5f}")

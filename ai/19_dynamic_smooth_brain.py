import math
import random


def squash(z: float) -> float:
  z = max(-50.0, min(50.0, z))
  return 1.0 / (1.0 + math.exp(-z))


def squash_slope(activation: float) -> float:
  return activation * (1.0 - activation)


def smooth_ramp(z: float) -> float:
  return z * squash(z)


def smooth_ramp_slope(z: float) -> float:
  s = squash(z)
  return s + z * s * (1.0 - s)


NUM_PIXELS = 64
NUM_HIDDEN = 4


class DynamicLeakyBrain:

  def __init__(self):
    random.seed(42)
    self.dials_layer1 = [
        [random.uniform(-0.2, 0.2) for _ in range(NUM_HIDDEN)]
        for _ in range(NUM_PIXELS)
    ]
    self.baseline_layer1 = [0.0] * NUM_HIDDEN
    self.alpha_layer1 = [0.10] * NUM_HIDDEN  # Dynamic leak dials

    self.dials_layer2 = [random.uniform(-0.2, 0.2) for _ in range(NUM_HIDDEN)]
    self.baseline_layer2 = 0.0

  def forward(self, pixels):
    raw_layer1 = [self.baseline_layer1[h] for h in range(NUM_HIDDEN)]
    for h in range(NUM_HIDDEN):
      for i in range(NUM_PIXELS):
        raw_layer1[h] += pixels[i] * self.dials_layer1[i][h]

    hidden_acts = [
        (raw_layer1[h] if raw_layer1[h] > 0.0 else self.alpha_layer1[h] * raw_layer1[h])
        for h in range(NUM_HIDDEN)
    ]

    raw_judge = self.baseline_layer2 + sum(
        hidden_acts[h] * self.dials_layer2[h] for h in range(NUM_HIDDEN)
    )
    return squash(raw_judge), raw_layer1, hidden_acts

  def train(self, dataset, passes=200, step_size=0.25):
    for p in range(passes):
      total_err = 0.0
      for sample in dataset:
        final_guess, raw_layer1, hidden_acts = self.forward(sample['pixels'])
        err = final_guess - sample['target']
        total_err += abs(err)

        judge_blame = err * squash_slope(final_guess)

        assistant_blame = [0.0] * NUM_HIDDEN
        alpha_grad = [0.0] * NUM_HIDDEN

        for h in range(NUM_HIDDEN):
          z = raw_layer1[h]
          incoming = judge_blame * self.dials_layer2[h]

          slope_z = 1.0 if z > 0.0 else self.alpha_layer1[h]
          assistant_blame[h] = incoming * slope_z

          if z <= 0.0:
            alpha_grad[h] = incoming * z

        # Update Judge
        self.baseline_layer2 -= step_size * judge_blame
        for h in range(NUM_HIDDEN):
          self.dials_layer2[h] -= step_size * judge_blame * hidden_acts[h]

        # Update Assistants and dynamic alpha
        for h in range(NUM_HIDDEN):
          self.baseline_layer1[h] -= step_size * assistant_blame[h]
          for i in range(NUM_PIXELS):
            self.dials_layer1[i][h] -= (
                step_size * assistant_blame[h] * sample['pixels'][i]
            )
          self.alpha_layer1[h] -= step_size * alpha_grad[h]

      if p % 50 == 0 or p == passes - 1:
        print(f"  Pass {p:3d} | Avg Error: {total_err / len(dataset):.4f}")


class SmoothRampBrain:

  def __init__(self):
    random.seed(42)
    self.dials_layer1 = [
        [random.uniform(-0.2, 0.2) for _ in range(NUM_HIDDEN)]
        for _ in range(NUM_PIXELS)
    ]
    self.baseline_layer1 = [0.0] * NUM_HIDDEN
    self.dials_layer2 = [random.uniform(-0.2, 0.2) for _ in range(NUM_HIDDEN)]
    self.baseline_layer2 = 0.0

  def forward(self, pixels):
    raw_layer1 = [self.baseline_layer1[h] for h in range(NUM_HIDDEN)]
    for h in range(NUM_HIDDEN):
      for i in range(NUM_PIXELS):
        raw_layer1[h] += pixels[i] * self.dials_layer1[i][h]

    hidden_acts = [smooth_ramp(raw_layer1[h]) for h in range(NUM_HIDDEN)]
    raw_judge = self.baseline_layer2 + sum(
        hidden_acts[h] * self.dials_layer2[h] for h in range(NUM_HIDDEN)
    )
    return squash(raw_judge), raw_layer1, hidden_acts

  def train(self, dataset, passes=200, step_size=0.25):
    for p in range(passes):
      total_err = 0.0
      for sample in dataset:
        final_guess, raw_layer1, hidden_acts = self.forward(sample['pixels'])
        err = final_guess - sample['target']
        total_err += abs(err)

        judge_blame = err * squash_slope(final_guess)

        assistant_blame = [0.0] * NUM_HIDDEN
        for h in range(NUM_HIDDEN):
          slope_h = smooth_ramp_slope(raw_layer1[h])
          assistant_blame[h] = (judge_blame * self.dials_layer2[h]) * slope_h

        self.baseline_layer2 -= step_size * judge_blame
        for h in range(NUM_HIDDEN):
          self.dials_layer2[h] -= step_size * judge_blame * hidden_acts[h]

        for h in range(NUM_HIDDEN):
          self.baseline_layer1[h] -= step_size * assistant_blame[h]
          for i in range(NUM_PIXELS):
            self.dials_layer1[i][h] -= (
                step_size * assistant_blame[h] * sample['pixels'][i]
            )

      if p % 50 == 0 or p == passes - 1:
        print(f"  Pass {p:3d} | Avg Error: {total_err / len(dataset):.4f}")


if __name__ == "__main__":
  dataset = []
  one_center = [0.0] * 64
  for r in range(8):
    one_center[r * 8 + 3] = 1.0
  dataset.append({'pixels': one_center, 'target': 1.0, 'label': '1 (center)'})

  one_hook = list(one_center)
  one_hook[0 * 8 + 2] = 1.0
  dataset.append({'pixels': one_hook, 'target': 1.0, 'label': '1 (hook)'})

  one_base = list(one_center)
  for c in range(2, 5):
    one_base[7 * 8 + c] = 1.0
  dataset.append({'pixels': one_base, 'target': 1.0, 'label': '1 (base)'})

  zero_box = [0.0] * 64
  for i in range(8):
    zero_box[0 * 8 + i] = 1.0
    zero_box[7 * 8 + i] = 1.0
    zero_box[i * 8 + 0] = 1.0
    zero_box[i * 8 + 7] = 1.0
  dataset.append({'pixels': zero_box, 'target': 0.0, 'label': '0 (box)'})

  letter_L = [0.0] * 64
  for r in range(8):
    letter_L[r * 8 + 1] = 1.0
  for c in range(1, 7):
    letter_L[7 * 8 + c] = 1.0
  dataset.append({'pixels': letter_L, 'target': 0.0, 'label': 'Letter L'})

  plus_sign = [0.0] * 64
  for i in range(8):
    plus_sign[3 * 8 + i] = 1.0
    plus_sign[i * 8 + 3] = 1.0
  dataset.append({'pixels': plus_sign, 'target': 0.0, 'label': 'Plus sign'})

  print("--- PART 1: DYNAMIC LEAK BRAIN ---")
  db = DynamicLeakyBrain()
  print("Initial alphas:", [round(a, 4) for a in db.alpha_layer1])
  db.train(dataset, 200, 0.25)
  print("Final learned alphas:", [round(a, 4) for a in db.alpha_layer1])

  print("\n--- PART 2: SMOOTH RAMP BRAIN ---")
  sb = SmoothRampBrain()
  sb.train(dataset, 200, 0.25)

import math
import random


def squash(z: float) -> float:
  z = max(-50.0, min(50.0, z))
  return 1.0 / (1.0 + math.exp(-z))


def squash_slope(a: float) -> float:
  return a * (1.0 - a)


def leaky_ramp(z: float, alpha: float = 0.01) -> float:
  return z if z > 0.0 else alpha * z


def leaky_ramp_slope(z: float, alpha: float = 0.01) -> float:
  return 1.0 if z > 0.0 else alpha


class StencilBrain:

  def __init__(self, seed: int = 1):
    rng = random.Random(seed)
    # Stencil 0 (9 dials + bias)
    self.stencil_0 = [
        [rng.uniform(-0.2, 0.2) for _ in range(3)] for _ in range(3)
    ]
    self.bias_0 = 0.0

    # Stencil 1 (9 dials + bias)
    self.stencil_1 = [
        [rng.uniform(-0.2, 0.2) for _ in range(3)] for _ in range(3)
    ]
    self.bias_1 = 0.0

    # Judge dials (2 dials + baseline)
    self.judge_dial_0 = rng.uniform(-0.2, 0.2)
    self.judge_dial_1 = rng.uniform(-0.2, 0.2)
    self.judge_baseline = 0.0

  def forward(self, img):
    raw_0 = [[0.0] * 6 for _ in range(6)]
    act_0 = [[0.0] * 6 for _ in range(6)]
    raw_1 = [[0.0] * 6 for _ in range(6)]
    act_1 = [[0.0] * 6 for _ in range(6)]

    peak_0 = -1e9
    peak_1 = -1e9
    max_pos_0 = (0, 0)
    max_pos_1 = (0, 0)

    for r in range(6):
      for c in range(6):
        sum_0 = self.bias_0
        sum_1 = self.bias_1
        for kr in range(3):
          for kc in range(3):
            px = img[r + kr][c + kc]
            sum_0 += px * self.stencil_0[kr][kc]
            sum_1 += px * self.stencil_1[kr][kc]

        raw_0[r][c] = sum_0
        raw_1[r][c] = sum_1
        act_0[r][c] = leaky_ramp(sum_0, 0.01)
        act_1[r][c] = leaky_ramp(sum_1, 0.01)

        if act_0[r][c] > peak_0:
          peak_0 = act_0[r][c]
          max_pos_0 = (r, c)
        if act_1[r][c] > peak_1:
          peak_1 = act_1[r][c]
          max_pos_1 = (r, c)

    raw_judge = (
        self.judge_baseline
        + peak_0 * self.judge_dial_0
        + peak_1 * self.judge_dial_1
    )
    return (
        squash(raw_judge),
        raw_0,
        act_0,
        max_pos_0,
        raw_1,
        act_1,
        max_pos_1,
        peak_0,
        peak_1,
    )

  def train(self, dataset, passes=400, step_size=0.20):
    for p in range(passes):
      total_err = 0.0
      for sample in dataset:
        (
            guess,
            raw_0,
            act_0,
            max_pos_0,
            raw_1,
            act_1,
            max_pos_1,
            peak_0,
            peak_1,
        ) = self.forward(sample['image'])
        err = guess - sample['target']
        total_err += abs(err)

        # 1. Judge blame
        judge_blame = err * squash_slope(guess)

        # 2. Blame flowing to peaks
        blame_peak_0 = judge_blame * self.judge_dial_0
        blame_peak_1 = judge_blame * self.judge_dial_1

        # 3. Peak-finder backward: ONLY winner gets blame
        r0, c0 = max_pos_0
        delta_0 = blame_peak_0 * leaky_ramp_slope(raw_0[r0][c0], 0.01)

        r1, c1 = max_pos_1
        delta_1 = blame_peak_1 * leaky_ramp_slope(raw_1[r1][c1], 0.01)

        # 4. Update Judge dials
        self.judge_baseline -= step_size * judge_blame
        self.judge_dial_0 -= step_size * judge_blame * peak_0
        self.judge_dial_1 -= step_size * judge_blame * peak_1

        # 5. Update Stencil 0 dials
        self.bias_0 -= step_size * delta_0
        for kr in range(3):
          for kc in range(3):
            self.stencil_0[kr][kc] -= (
                step_size * delta_0 * sample['image'][r0 + kr][c0 + kc]
            )

        # 6. Update Stencil 1 dials
        self.bias_1 -= step_size * delta_1
        for kr in range(3):
          for kc in range(3):
            self.stencil_1[kr][kc] -= (
                step_size * delta_1 * sample['image'][r1 + kr][c1 + kc]
            )

      if p % 50 == 0 or p == passes - 1:
        print(f"  Pass {p:3d} | Avg Error: {total_err / len(dataset):.4f}")


def build_dataset():
  dataset = []
  one_c = [[0.0] * 8 for _ in range(8)]
  for r in range(8):
    one_c[r][3] = 1.0
  dataset.append({'image': one_c, 'target': 1.0, 'label': '1 (Center)'})

  one_l = [[0.0] * 8 for _ in range(8)]
  for r in range(8):
    one_l[r][1] = 1.0
  dataset.append({'image': one_l, 'target': 1.0, 'label': '1 (Shifted Left)'})

  one_r = [[0.0] * 8 for _ in range(8)]
  for r in range(8):
    one_r[r][5] = 1.0
  dataset.append(
      {'image': one_r, 'target': 1.0, 'label': '1 (Shifted Right)'}
  )

  one_hook = [row[:] for row in one_c]
  one_hook[0][2] = 1.0
  dataset.append({'image': one_hook, 'target': 1.0, 'label': '1 (With hook)'})

  box = [[0.0] * 8 for _ in range(8)]
  for i in range(8):
    box[0][i] = 1.0
    box[7][i] = 1.0
    box[i][0] = 1.0
    box[i][7] = 1.0
  dataset.append({'image': box, 'target': 0.0, 'label': '0 (Box)'})

  l_shape = [[0.0] * 8 for _ in range(8)]
  for r in range(8):
    l_shape[r][1] = 1.0
  for c in range(1, 7):
    l_shape[7][c] = 1.0
  dataset.append({'image': l_shape, 'target': 0.0, 'label': 'Letter L'})

  plus = [[0.0] * 8 for _ in range(8)]
  for i in range(8):
    plus[3][i] = 1.0
    plus[i][3] = 1.0
  dataset.append({'image': plus, 'target': 0.0, 'label': 'Plus Sign'})

  horiz = [[0.0] * 8 for _ in range(8)]
  for c in range(8):
    horiz[4][c] = 1.0
  dataset.append({'image': horiz, 'target': 0.0, 'label': 'Horizontal Bar'})
  return dataset


if __name__ == '__main__':
  dataset = build_dataset()
  brain = StencilBrain(seed=1)
  print('--- TRAINING 23-DIAL CONVOLUTIONAL BRAIN IN PYTHON ---')
  brain.train(dataset, 400, 0.20)

  print('\n--- TESTING UNSEEN SHIFTED "1" AT COLUMN 2 ---')
  unseen_col2 = [[0.0] * 8 for _ in range(8)]
  for r in range(8):
    unseen_col2[r][2] = 1.0
  res = brain.forward(unseen_col2)[0]
  print(f'  Unseen Digit "1" at Column 2 -> Confidence: {res * 100.0:.1f}%')

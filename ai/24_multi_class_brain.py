import math
import random


def competing_probabilities(raw_scores: list[float]) -> list[float]:
  max_z = max(raw_scores)
  exps = [math.exp(z - max_z) for z in raw_scores]
  sum_exp = sum(exps)
  return [e / sum_exp for e in exps]


def leaky_ramp(z: float, alpha: float = 0.01) -> float:
  return z if z > 0.0 else alpha * z


def leaky_ramp_slope(z: float, alpha: float = 0.01) -> float:
  return 1.0 if z > 0.0 else alpha


NUM_FILTERS = 4
NUM_CLASSES = 4


class MultiClassCNN:

  def __init__(self, seed: int = 42):
    rng = random.Random(seed)
    self.stencils = [
        [[rng.uniform(-0.2, 0.2) for _ in range(3)] for _ in range(3)]
        for _ in range(NUM_FILTERS)
    ]
    self.filter_biases = [0.0] * NUM_FILTERS

    self.judge_dials = [
        [rng.uniform(-0.2, 0.2) for _ in range(NUM_FILTERS)]
        for _ in range(NUM_CLASSES)
    ]
    self.judge_baselines = [0.0] * NUM_CLASSES

  def forward(self, img):
    raw_maps = [[[0.0] * 6 for _ in range(6)] for _ in range(NUM_FILTERS)]
    act_maps = [[[0.0] * 6 for _ in range(6)] for _ in range(NUM_FILTERS)]
    peak_coords = [(0, 0)] * NUM_FILTERS
    peaks = [-1e9] * NUM_FILTERS

    for f in range(NUM_FILTERS):
      for r in range(6):
        for c in range(6):
          total = self.filter_biases[f]
          for kr in range(3):
            for kc in range(3):
              total += img[r + kr][c + kc] * self.stencils[f][kr][kc]

          raw_maps[f][r][c] = total
          act = leaky_ramp(total, 0.01)
          act_maps[f][r][c] = act

          if act > peaks[f]:
            peaks[f] = act
            peak_coords[f] = (r, c)

    raw_judges = [0.0] * NUM_CLASSES
    for k in range(NUM_CLASSES):
      raw_judges[k] = self.judge_baselines[k] + sum(
          peaks[f] * self.judge_dials[k][f] for f in range(NUM_FILTERS)
      )

    probs = competing_probabilities(raw_judges)
    return probs, raw_maps, act_maps, peak_coords, peaks

  def train(self, dataset, passes=600, step_size=0.25):
    for p in range(passes):
      total_loss = 0.0
      correct = 0

      for sample in dataset:
        probs, raw_maps, act_maps, peak_coords, peaks = self.forward(
            sample['image']
        )
        target = sample['target']

        p_correct = max(1e-12, probs[target])
        total_loss += -math.log(p_correct)

        best_class = probs.index(max(probs))
        if best_class == target:
          correct += 1

        # Exact gradient: Prediction - Target!
        judge_blames = [
            probs[k] - (1.0 if k == target else 0.0) for k in range(NUM_CLASSES)
        ]

        peak_blames = [
            sum(judge_blames[k] * self.judge_dials[k][f] for k in range(NUM_CLASSES))
            for f in range(NUM_FILTERS)
        ]

        for k in range(NUM_CLASSES):
          self.judge_baselines[k] -= step_size * judge_blames[k]
          for f in range(NUM_FILTERS):
            self.judge_dials[k][f] -= step_size * judge_blames[k] * peaks[f]

        for f in range(NUM_FILTERS):
          wr, wc = peak_coords[f]
          z = raw_maps[f][wr][wc]
          delta = peak_blames[f] * leaky_ramp_slope(z, 0.01)

          self.filter_biases[f] -= step_size * delta
          for kr in range(3):
            for kc in range(3):
              self.stencils[f][kr][kc] -= (
                  step_size * delta * sample['image'][wr + kr][wc + kc]
              )

      if p % 50 == 0 or p == passes - 1:
        acc = (correct / len(dataset)) * 100.0
        print(
            f'  Pass {p:3d} | Avg Loss: {total_loss / len(dataset):.4f} | Acc:'
            f' {acc:.1f}%'
        )


def build_multiclass_dataset():
  dataset = []

  def make_grid():
    return [[0.0] * 8 for _ in range(8)]

  # Class 0: Box
  d0_a = make_grid()
  for i in range(1, 7):
    d0_a[1][i] = 1.0
    d0_a[6][i] = 1.0
    d0_a[i][1] = 1.0
    d0_a[i][6] = 1.0
  dataset.append({'image': d0_a, 'target': 0, 'label': 'Digit 0 (Box)'})

  d0_b = make_grid()
  for i in range(8):
    d0_b[0][i] = 1.0
    d0_b[7][i] = 1.0
    d0_b[i][0] = 1.0
    d0_b[i][7] = 1.0
  dataset.append({'image': d0_b, 'target': 0, 'label': 'Digit 0 (Big Box)'})

  # Class 1: Vertical line
  d1_c = make_grid()
  for r in range(8):
    d1_c[r][3] = 1.0
  dataset.append({'image': d1_c, 'target': 1, 'label': 'Digit 1 (Center)'})

  d1_l = make_grid()
  for r in range(8):
    d1_l[r][1] = 1.0
  dataset.append(
      {'image': d1_l, 'target': 1, 'label': 'Digit 1 (Shifted Left)'}
  )

  d1_r = make_grid()
  for r in range(8):
    d1_r[r][6] = 1.0
  dataset.append(
      {'image': d1_r, 'target': 1, 'label': 'Digit 1 (Shifted Right)'}
  )

  # Class 2: Digit 2
  d2 = make_grid()
  for c in range(1, 6):
    d2[1][c] = 1.0
  d2[2][5] = 1.0
  d2[3][4] = 1.0
  d2[4][3] = 1.0
  d2[5][2] = 1.0
  for c in range(1, 7):
    d2[6][c] = 1.0
  dataset.append({'image': d2, 'target': 2, 'label': 'Digit 2 (Standard)'})

  # Class 3: Digit 7
  d7_a = make_grid()
  for c in range(1, 7):
    d7_a[1][c] = 1.0
  for r in range(1, 8):
    d7_a[r][6] = 1.0
  dataset.append({'image': d7_a, 'target': 3, 'label': 'Digit 7 (Top + Right)'})

  d7_b = make_grid()
  for c in range(6):
    d7_b[0][c] = 1.0
  for r in range(8):
    d7_b[r][5] = 1.0
  dataset.append({'image': d7_b, 'target': 3, 'label': 'Digit 7 (Shifted)'})

  return dataset


if __name__ == '__main__':
  dataset = build_multiclass_dataset()
  brain = MultiClassCNN(seed=42)
  print('--- TRAINING MULTI-CLASS CONVOLUTIONAL BRAIN (PYTHON) ---')
  brain.train(dataset, passes=600, step_size=0.25)

  unseen_1 = [[0.0] * 8 for _ in range(8)]
  for r in range(8):
    unseen_1[r][4] = 1.0
  p1 = brain.forward(unseen_1)[0]
  print(f'Unseen Shifted 1 -> P(1) = {p1[1] * 100:.1f}%')

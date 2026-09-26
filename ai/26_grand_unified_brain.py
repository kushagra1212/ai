import copy
import math
import random


def squash(z: float) -> float:
  z = max(-50.0, min(50.0, z))
  return 1.0 / (1.0 + math.exp(-z))


def competing_probabilities(raw_scores: list[float]) -> list[float]:
  max_z = max(raw_scores)
  exps = [math.exp(z - max_z) for z in raw_scores]
  sum_exp = sum(exps)
  return [e / sum_exp for e in exps]


def smooth_ramp(z: float) -> float:
  return z * squash(z)


def smooth_ramp_slope(z: float) -> float:
  s = squash(z)
  return s + z * s * (1.0 - s)


def dynamic_leaky_act(z: float, alpha: float) -> float:
  return z if z > 0.0 else alpha * z


def dynamic_leaky_slope_z(z: float, alpha: float) -> float:
  return 1.0 if z > 0.0 else alpha


NUM_FILTERS = 4
NUM_CLASSES = 4


class UnifiedOrganism:

  def __init__(self, rng=None):
    if rng is None:
      rng = random.Random(42)

    self.stencils = [
        [[rng.uniform(-0.2, 0.2) for _ in range(3)] for _ in range(3)]
        for _ in range(NUM_FILTERS)
    ]
    self.v_stencils = [[[0.0] * 3 for _ in range(3)] for _ in range(NUM_FILTERS)]

    self.biases = [0.0] * NUM_FILTERS
    self.v_biases = [0.0] * NUM_FILTERS

    self.judge_dials = [
        [rng.uniform(-0.2, 0.2) for _ in range(NUM_FILTERS)]
        for _ in range(NUM_CLASSES)
    ]
    self.v_judge_dials = [
        [0.0] * NUM_FILTERS for _ in range(NUM_CLASSES)
    ]

    self.judge_baselines = [0.0] * NUM_CLASSES
    self.v_judge_baselines = [0.0] * NUM_CLASSES

    self.loss = 999.0
    self.accuracy = 0.0

  def forward(self, img):
    raw_maps = [[[0.0] * 6 for _ in range(6)] for _ in range(NUM_FILTERS)]
    peaks = [-1e9] * NUM_FILTERS
    peak_coords = [(0, 0)] * NUM_FILTERS

    for f in range(NUM_FILTERS):
      for r in range(6):
        for c in range(6):
          total = self.biases[f]
          for kr in range(3):
            for kc in range(3):
              total += img[r + kr][c + kc] * self.stencils[f][kr][kc]

          raw_maps[f][r][c] = total
          act = smooth_ramp(total)  # Smooth curved activation

          if act > peaks[f]:
            peaks[f] = act
            peak_coords[f] = (r, c)

    raw_judges = [0.0] * NUM_CLASSES
    for k in range(NUM_CLASSES):
      raw_judges[k] = self.judge_baselines[k] + sum(
          peaks[f] * self.judge_dials[k][f] for f in range(NUM_FILTERS)
      )

    probs = competing_probabilities(raw_judges)
    return probs, raw_maps, peak_coords, peaks

  def evaluate(self, dataset):
    total_loss = 0.0
    correct = 0

    for s in dataset:
      probs = self.forward(s['image'])[0]
      total_loss += -math.log(max(1e-12, probs[s['target']]))
      if probs.index(max(probs)) == s['target']:
        correct += 1

    self.loss = total_loss / len(dataset)
    self.accuracy = (correct * 100.0) / len(dataset)

  def polish_with_momentum(self, dataset, step_size=0.20, friction=0.30):
    for sample in dataset:
      probs, raw_maps, peak_coords, peaks = self.forward(sample['image'])
      target = sample['target']

      judge_blames = [
          probs[k] - (1.0 if k == target else 0.0) for k in range(NUM_CLASSES)
      ]
      peak_blames = [
          sum(judge_blames[k] * self.judge_dials[k][f] for k in range(NUM_CLASSES))
          for f in range(NUM_FILTERS)
      ]

      for k in range(NUM_CLASSES):
        self.v_judge_baselines[k] = (
            friction * self.v_judge_baselines[k] + step_size * judge_blames[k]
        )
        self.judge_baselines[k] -= self.v_judge_baselines[k]

        for f in range(NUM_FILTERS):
          self.v_judge_dials[k][f] = (
              friction * self.v_judge_dials[k][f]
              + step_size * judge_blames[k] * peaks[f]
          )
          self.judge_dials[k][f] -= self.v_judge_dials[k][f]

      for f in range(NUM_FILTERS):
        wr, wc = peak_coords[f]
        z = raw_maps[f][wr][wc]
        delta = peak_blames[f] * smooth_ramp_slope(z)

        self.v_biases[f] = friction * self.v_biases[f] + step_size * delta
        self.biases[f] -= self.v_biases[f]

        for kr in range(3):
          for kc in range(3):
            g = delta * sample['image'][wr + kr][wc + kc]
            self.v_stencils[f][kr][kc] = (
                friction * self.v_stencils[f][kr][kc] + step_size * g
            )
            self.stencils[f][kr][kc] -= self.v_stencils[f][kr][kc]


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
  rng = random.Random(42)

  population = [UnifiedOrganism(rng) for _ in range(16)]

  print(
      '--- TRAINING GRAND UNIFIED BRAIN (POPULATION = 16, GENERATIONS = 30) ---'
  )

  for gen in range(30):
    for org in population:
      for _ in range(3):
        org.polish_with_momentum(dataset, step_size=0.20, friction=0.30)
      org.evaluate(dataset)

    population.sort(key=lambda o: o.loss)

    if gen % 5 == 0 or gen == 29:
      champ = population[0]
      print(
          f'  Gen {gen:2d} | Champion Loss: {champ.loss:.4f} | Accuracy:'
          f' {champ.accuracy:.1f}%'
      )

    if population[0].loss < 0.005:
      print(f'  >>> Converged early at Generation {gen}!')
      break

    # Elitism + cloning
    next_gen = []
    for i in range(4):
      next_gen.append(population[i])
      for _ in range(3):
        clone = copy.deepcopy(population[i])
        for f in range(NUM_FILTERS):
          for kr in range(3):
            for kc in range(3):
              if rng.random() < 0.10:
                clone.stencils[f][kr][kc] += rng.gauss(0.0, 0.02)
        next_gen.append(clone)
    population = next_gen

  champ = population[0]
  names = ['Digit 0', 'Digit 1', 'Digit 2', 'Digit 7']
  print('\n--- TESTING CHAMPION ON UNSEEN SHIFTED DIGITS ---')

  # Unseen shifted 1
  test_1 = [[0.0] * 8 for _ in range(8)]
  for r in range(8):
    test_1[r][4] = 1.0
  p1 = champ.forward(test_1)[0]
  print(
      '  Unseen Shifted "1" ->'
      f' {", ".join([f"{names[k]}: {p1[k] * 100:.1f}%" for k in range(4)])}'
  )

  # Unseen shifted 7
  test_7 = [[0.0] * 8 for _ in range(8)]
  for c in range(2, 6):
    test_7[1][c] = 1.0
  for r in range(1, 8):
    test_7[r][5] = 1.0
  p7 = champ.forward(test_7)[0]
  print(
      '  Unseen Shifted "7" ->'
      f' {", ".join([f"{names[k]}: {p7[k] * 100:.1f}%" for k in range(4)])}'
  )

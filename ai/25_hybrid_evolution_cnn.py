import copy
import math
import random


def squash(z: float) -> float:
  z = max(-50.0, min(50.0, z))
  return 1.0 / (1.0 + math.exp(-z))


def squash_slope(a: float) -> float:
  return a * (1.0 - a)


def dynamic_leaky_act(z: float, alpha: float) -> float:
  return z if z > 0.0 else alpha * z


def dynamic_leaky_slope_z(z: float, alpha: float) -> float:
  return 1.0 if z > 0.0 else alpha


class ConvOrganism:

  def __init__(self, rng=None):
    if rng is None:
      rng = random.Random(42)

    self.stencil_0 = [
        [rng.uniform(-0.2, 0.2) for _ in range(3)] for _ in range(3)
    ]
    self.v_stencil_0 = [[0.0] * 3 for _ in range(3)]
    self.bias_0 = 0.0
    self.v_bias_0 = 0.0
    self.alpha_0 = rng.uniform(0.01, 0.25)
    self.v_alpha_0 = 0.0

    self.stencil_1 = [
        [rng.uniform(-0.2, 0.2) for _ in range(3)] for _ in range(3)
    ]
    self.v_stencil_1 = [[0.0] * 3 for _ in range(3)]
    self.bias_1 = 0.0
    self.v_bias_1 = 0.0
    self.alpha_1 = rng.uniform(0.01, 0.25)
    self.v_alpha_1 = 0.0

    self.judge_dial_0 = rng.uniform(-0.2, 0.2)
    self.v_judge_dial_0 = 0.0
    self.judge_dial_1 = rng.uniform(-0.2, 0.2)
    self.v_judge_dial_1 = 0.0
    self.judge_baseline = 0.0
    self.v_judge_baseline = 0.0

    self.error = 999.0

  def forward(self, img):
    raw_0 = [[0.0] * 6 for _ in range(6)]
    raw_1 = [[0.0] * 6 for _ in range(6)]
    peak_0 = -1e9
    peak_1 = -1e9
    max_pos_0 = (0, 0)
    max_pos_1 = (0, 0)

    for r in range(6):
      for c in range(6):
        s0 = self.bias_0
        s1 = self.bias_1
        for kr in range(3):
          for kc in range(3):
            px = img[r + kr][c + kc]
            s0 += px * self.stencil_0[kr][kc]
            s1 += px * self.stencil_1[kr][kc]

        raw_0[r][c] = s0
        raw_1[r][c] = s1
        a0 = dynamic_leaky_act(s0, self.alpha_0)
        a1 = dynamic_leaky_act(s1, self.alpha_1)

        if a0 > peak_0:
          peak_0, max_pos_0 = a0, (r, c)
        if a1 > peak_1:
          peak_1, max_pos_1 = a1, (r, c)

    raw_j = (
        self.judge_baseline
        + peak_0 * self.judge_dial_0
        + peak_1 * self.judge_dial_1
    )
    return squash(raw_j), raw_0, max_pos_0, raw_1, max_pos_1, peak_0, peak_1

  def evaluate(self, dataset):
    total = 0.0
    for s in dataset:
      guess = self.forward(s['image'])[0]
      total += abs(guess - s['target'])
    self.error = total / len(dataset)

  def polish_with_momentum(self, dataset, step_size=0.20, friction=0.85):
    for sample in dataset:
      guess, raw_0, mp0, raw_1, mp1, p0, p1 = self.forward(sample['image'])
      err = guess - sample['target']

      j_blame = err * squash_slope(guess)
      blame_p0 = j_blame * self.judge_dial_0
      blame_p1 = j_blame * self.judge_dial_1

      r0, c0 = mp0
      z0 = raw_0[r0][c0]
      delta_0 = blame_p0 * dynamic_leaky_slope_z(z0, self.alpha_0)

      r1, c1 = mp1
      z1 = raw_1[r1][c1]
      delta_1 = blame_p1 * dynamic_leaky_slope_z(z1, self.alpha_1)

      alpha_0_grad = (blame_p0 * z0) if z0 <= 0.0 else 0.0
      alpha_1_grad = (blame_p1 * z1) if z1 <= 0.0 else 0.0

      # Update Judge
      self.v_judge_baseline = (
          friction * self.v_judge_baseline + step_size * j_blame
      )
      self.judge_baseline -= self.v_judge_baseline

      self.v_judge_dial_0 = (
          friction * self.v_judge_dial_0 + step_size * j_blame * p0
      )
      self.judge_dial_0 -= self.v_judge_dial_0

      self.v_judge_dial_1 = (
          friction * self.v_judge_dial_1 + step_size * j_blame * p1
      )
      self.judge_dial_1 -= self.v_judge_dial_1

      # Update Stencil 0
      self.v_bias_0 = friction * self.v_bias_0 + step_size * delta_0
      self.bias_0 -= self.v_bias_0

      self.v_alpha_0 = friction * self.v_alpha_0 + step_size * alpha_0_grad
      self.alpha_0 -= self.v_alpha_0

      for kr in range(3):
        for kc in range(3):
          g = delta_0 * sample['image'][r0 + kr][c0 + kc]
          self.v_stencil_0[kr][kc] = (
              friction * self.v_stencil_0[kr][kc] + step_size * g
          )
          self.stencil_0[kr][kc] -= self.v_stencil_0[kr][kc]

      # Update Stencil 1
      self.v_bias_1 = friction * self.v_bias_1 + step_size * delta_1
      self.bias_1 -= self.v_bias_1

      self.v_alpha_1 = friction * self.v_alpha_1 + step_size * alpha_1_grad
      self.alpha_1 -= self.v_alpha_1

      for kr in range(3):
        for kc in range(3):
          g = delta_1 * sample['image'][r1 + kr][c1 + kc]
          self.v_stencil_1[kr][kc] = (
              friction * self.v_stencil_1[kr][kc] + step_size * g
          )
          self.stencil_1[kr][kc] -= self.v_stencil_1[kr][kc]


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
  rng = random.Random(42)

  population = [ConvOrganism(rng) for _ in range(20)]

  print(
      '--- TRAINING HYBRID EVOLUTIONARY CNN (POPULATION = 20, GENERATIONS = 50)'
      ' ---'
  )

  for gen in range(50):
    for org in population:
      org.polish_with_momentum(dataset, step_size=0.20, friction=0.85)
      org.evaluate(dataset)

    population.sort(key=lambda o: o.error)

    if gen % 10 == 0 or gen == 49:
      champ = population[0]
      print(
          f'  Gen {gen:2d} | Champion Error: {champ.error:.4f} | Alpha_0:'
          f' {champ.alpha_0:.3f} | Alpha_1: {champ.alpha_1:.3f}'
      )

    if population[0].error < 0.005:
      print(f'  >>> Converged early at Generation {gen}!')
      break

    # Top 4 champions reproduce
    next_gen = []
    for i in range(4):
      next_gen.append(population[i])  # Keep champion
      for _ in range(4):
        clone = copy.deepcopy(population[i])  # Inherits velocity!
        for kr in range(3):
          for kc in range(3):
            if rng.random() < 0.20:
              clone.stencil_0[kr][kc] += rng.gauss(0.0, 0.03)
            if rng.random() < 0.20:
              clone.stencil_1[kr][kc] += rng.gauss(0.0, 0.03)
        if rng.random() < 0.20:
          clone.alpha_0 = max(0.001, clone.alpha_0 + rng.gauss(0.0, 0.015))
        if rng.random() < 0.20:
          clone.alpha_1 = max(0.001, clone.alpha_1 + rng.gauss(0.0, 0.015))
        next_gen.append(clone)
    population = next_gen

  champ = population[0]
  print('\n--- TESTING CHAMPION ON UNSEEN SHIFTED SAMPLES ---')
  unseen_col2 = [[0.0] * 8 for _ in range(8)]
  for r in range(8):
    unseen_col2[r][2] = 1.0
  conf2 = champ.forward(unseen_col2)[0]
  print(f'  Unseen Digit "1" at Column 2 -> Confidence: {conf2 * 100:.1f}%')

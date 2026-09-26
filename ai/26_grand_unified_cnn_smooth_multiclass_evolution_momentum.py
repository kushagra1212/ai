import math
import random
import struct
import time

# =============================================================================
# PROGRAM 26 (PYTHON): GRAND UNIFIED CNN + SMOOTH RAMP + MULTICLASS + EVOLUTION
# REAL MNIST 28x28 (784 PIXELS) & ALL 10 DIGITS (0 to 9)
# =============================================================================


def load_mnist_raw(img_path: str, lbl_path: str, max_count: int):
  with open(img_path, 'rb') as f_img, open(lbl_path, 'rb') as f_lbl:
    _ = struct.unpack('>I', f_img.read(4))[0]
    num_images = struct.unpack('>I', f_img.read(4))[0]
    _ = struct.unpack('>I', f_img.read(4))[0]
    _ = struct.unpack('>I', f_img.read(4))[0]

    _ = struct.unpack('>I', f_lbl.read(4))[0]
    _ = struct.unpack('>I', f_lbl.read(4))[0]

    count = min(max_count, num_images)
    dataset = []

    for _ in range(count):
      lbl = struct.unpack('B', f_lbl.read(1))[0]
      raw_pixels = f_img.read(784)
      pixels = [px / 255.0 for px in raw_pixels]
      dataset.append({'label': lbl, 'pixels': pixels})

  return dataset


def squash(z: float) -> float:
  z = max(-40.0, min(40.0, z))
  return 1.0 / (1.0 + math.exp(-z))


def smooth_ramp(z: float) -> float:
  return z * squash(z)


def smooth_ramp_slope(z: float) -> float:
  s = squash(z)
  return s + z * s * (1.0 - s)


def competing_probabilities(raw_scores: list[float]) -> list[float]:
  max_z = max(raw_scores)
  exps = [math.exp(z - max_z) for z in raw_scores]
  sum_exp = sum(exps)
  return [e / sum_exp for e in exps]


KERNEL_SIZE = 5
NUM_FILTERS = 16
NUM_CLASSES = 10
OUT_DIM = 28 - KERNEL_SIZE + 1  # 24
MID_POINT = OUT_DIM // 2  # 12
POOL_REGIONS = 4
NUM_FEATURES = NUM_FILTERS * POOL_REGIONS  # 64


class UnifiedOrganism:

  def __init__(self, rng=None):
    if rng is None:
      rng = random.Random(42)

    self.stencils = [
        [rng.gauss(0.0, 0.05) for _ in range(KERNEL_SIZE * KERNEL_SIZE)]
        for _ in range(NUM_FILTERS)
    ]
    self.biases = [0.0] * NUM_FILTERS
    self.v_stencils = [
        [0.0] * (KERNEL_SIZE * KERNEL_SIZE) for _ in range(NUM_FILTERS)
    ]
    self.v_biases = [0.0] * NUM_FILTERS

    self.judge_dials = [
        [rng.gauss(0.0, 0.05) for _ in range(NUM_FEATURES)]
        for _ in range(NUM_CLASSES)
    ]
    self.judge_baselines = [0.0] * NUM_CLASSES
    self.v_judge_dials = [[0.0] * NUM_FEATURES for _ in range(NUM_CLASSES)]
    self.v_judge_baselines = [0.0] * NUM_CLASSES

    self.loss = 999.0
    self.accuracy = 0.0

  def forward(self, pixels):
    features = [-1e9] * NUM_FEATURES
    feat_pos = [0] * NUM_FEATURES
    feat_raw_z = [0.0] * NUM_FEATURES

    for f in range(NUM_FILTERS):
      st = self.stencils[f]
      b = self.biases[f]

      for r in range(OUT_DIM):
        r_quad = 0 if r < MID_POINT else 1
        for c in range(OUT_DIM):
          c_quad = 0 if c < MID_POINT else 1
          q_idx = r_quad * 2 + c_quad
          feat_id = f * POOL_REGIONS + q_idx

          total = b
          for kr in range(KERNEL_SIZE):
            row_off = (r + kr) * 28
            k_off = kr * KERNEL_SIZE
            for kc in range(KERNEL_SIZE):
              total += pixels[row_off + (c + kc)] * st[k_off + kc]

          act = smooth_ramp(total)
          if act > features[feat_id]:
            features[feat_id] = act
            feat_pos[feat_id] = r * 28 + c
            feat_raw_z[feat_id] = total

    raw_scores = [0.0] * NUM_CLASSES
    for c in range(NUM_CLASSES):
      raw_scores[c] = self.judge_baselines[c] + sum(
          features[i] * self.judge_dials[c][i] for i in range(NUM_FEATURES)
      )

    return (
        competing_probabilities(raw_scores),
        features,
        feat_pos,
        feat_raw_z,
    )

  def polish_with_momentum(self, batch, lr: float, friction: float):
    for sample in batch:
      probs, features, feat_pos, feat_raw_z = self.forward(sample['pixels'])

      judge_blame = [
          probs[c] - (1.0 if c == sample['label'] else 0.0)
          for c in range(NUM_CLASSES)
      ]

      feat_blame = [0.0] * NUM_FEATURES
      for i in range(NUM_FEATURES):
        feat_blame[i] = sum(
            judge_blame[c] * self.judge_dials[c][i] for c in range(NUM_CLASSES)
        )

      for c in range(NUM_CLASSES):
        self.v_judge_baselines[c] = (
            friction * self.v_judge_baselines[c] + lr * judge_blame[c]
        )
        self.judge_baselines[c] -= self.v_judge_baselines[c]

        for i in range(NUM_FEATURES):
          self.v_judge_dials[c][i] = (
              friction * self.v_judge_dials[c][i]
              + lr * judge_blame[c] * features[i]
          )
          self.judge_dials[c][i] -= self.v_judge_dials[c][i]

      for i in range(NUM_FEATURES):
        f = i // POOL_REGIONS
        delta = feat_blame[i] * smooth_ramp_slope(feat_raw_z[i])

        self.v_biases[f] = (
            friction * self.v_biases[f] + (lr / POOL_REGIONS) * delta
        )
        self.biases[f] -= self.v_biases[f]

        pr = feat_pos[i] // 28
        pc = feat_pos[i] % 28

        for kr in range(KERNEL_SIZE):
          row_off = (pr + kr) * 28
          k_off = kr * KERNEL_SIZE
          for kc in range(KERNEL_SIZE):
            g = delta * sample['pixels'][row_off + (pc + kc)]
            self.v_stencils[f][k_off + kc] = (
                friction * self.v_stencils[f][k_off + kc]
                + (lr / POOL_REGIONS) * g
            )
            self.stencils[f][k_off + kc] -= self.v_stencils[f][k_off + kc]

  def evaluate(self, dataset):
    total_loss = 0.0
    correct = 0
    for s in dataset:
      probs = self.forward(s['pixels'])[0]
      total_loss += -math.log(max(1e-12, probs[s['label']]))
      pred = max(range(NUM_CLASSES), key=lambda c: probs[c])
      if pred == s['label']:
        correct += 1

    self.loss = total_loss / len(dataset)
    self.accuracy = correct * 100.0 / len(dataset)

  def save_to_file(self, filepath: str):
    with open(filepath, 'w') as f:
      f.write('# GRAND UNIFIED REAL MNIST WEIGHTS (PYTHON)\n\n')
      for f_idx in range(NUM_FILTERS):
        f.write(f'# FILTER {f_idx} BIAS\n{self.biases[f_idx]:.8f}\n')
        f.write(f'# FILTER {f_idx} STENCIL 5x5\n')
        for r in range(KERNEL_SIZE):
          f.write(
              ' '.join(
                  f'{self.stencils[f_idx][r * KERNEL_SIZE + c]:.8f}'
                  for c in range(KERNEL_SIZE)
              )
              + '\n'
          )
        f.write('\n')

      for c in range(NUM_CLASSES):
        f.write(f'# JUDGE {c} BASELINE\n{self.judge_baselines[c]:.8f}\n')
        f.write(f'# JUDGE {c} DIALS\n')
        f.write(
            ' '.join(f'{self.judge_dials[c][i]:.8f}' for i in range(NUM_FEATURES))
            + '\n'
        )
        f.write('\n')


def main():
  import sys
  generations = int(sys.argv[1]) if len(sys.argv) > 1 else 6

  print('====================================================================')
  print(' PROGRAM 26 (PYTHON): GRAND UNIFIED REAL MNIST EVOLUTION')
  print(f' Running for {generations} Generations...')
  print('====================================================================\n')

  train_img_p = 'data/mnist/train-images-idx3-ubyte'
  train_lbl_p = 'data/mnist/train-labels-idx1-ubyte'
  test_img_p = 'data/mnist/t10k-images-idx3-ubyte'
  test_lbl_p = 'data/mnist/t10k-labels-idx1-ubyte'

  train_data = load_mnist_raw(train_img_p, train_lbl_p, 1000)
  test_data = load_mnist_raw(test_img_p, test_lbl_p, 200)

  rng = random.Random(42)
  population = [UnifiedOrganism(rng) for _ in range(4)]

  print(f'Trained on {len(train_data)} samples with 4 organisms...')
  for gen in range(1, generations + 1):
    for org in population:
      org.polish_with_momentum(train_data, lr=0.015, friction=0.50)
      org.evaluate(test_data)

    population.sort(key=lambda o: o.loss)
    champ = population[0]
    print(
        f' Gen {gen:2d}/{generations} | Loss: {champ.loss:.4f} | Real Test Acc:'
        f' {champ.accuracy:.1f}%'
    )

  champ.save_to_file('grand_unified_mnist_weights_python.txt')
  print('Saved weights to grand_unified_mnist_weights_python.txt')


if __name__ == '__main__':
  main()

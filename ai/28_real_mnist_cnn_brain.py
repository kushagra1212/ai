import math
import struct
import time


def load_mnist_dataset(img_path: str, lbl_path: str, max_count: int):
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


KERNEL_SIZE = 5
NUM_FILTERS = 8
NUM_CLASSES = 10
OUT_DIM = 28 - KERNEL_SIZE + 1  # 24
MID_POINT = OUT_DIM // 2  # 12
POOL_REGIONS = 4
NUM_FEATURES = NUM_FILTERS * POOL_REGIONS  # 32


class RealMNISTBrain:

  def __init__(self, rng=None):
    import random

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
    self.v_judge_dials = [
        [0.0] * NUM_FEATURES for _ in range(NUM_CLASSES)
    ]
    self.v_judge_baselines = [0.0] * NUM_CLASSES

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

    max_s = max(raw_scores)
    exps = [math.exp(s - max_s) for s in raw_scores]
    sum_exp = sum(exps)
    probs = [e / sum_exp for e in exps]

    return probs, features, feat_pos, feat_raw_z

  def train_sample(self, sample, lr: float, friction: float):
    probs, features, feat_pos, feat_raw_z = self.forward(sample['pixels'])
    loss = -math.log(max(1e-12, probs[sample['label']]))

    judge_blame = [
        probs[c] - (1.0 if c == sample['label'] else 0.0)
        for c in range(NUM_CLASSES)
    ]

    feat_blame = [0.0] * NUM_FEATURES
    for i in range(NUM_FEATURES):
      feat_blame[i] = sum(
          judge_blame[c] * self.judge_dials[c][i] for c in range(NUM_CLASSES)
      )

    # Update Judges
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

    # Update Stencils
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

    return loss

  def save_to_file(self, filepath: str):
    with open(filepath, 'w') as f:
      f.write('# REAL MNIST CNN WEIGHTS (PYTHON)\n\n')
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
  print('====================================================================')
  print(' PROGRAM 28: REAL MNIST CNN TRAINING ON REAL HUMAN HANDWRITING (PYTHON)')
  print('====================================================================\n')

  train_img_p = 'data/mnist/train-images-idx3-ubyte'
  train_lbl_p = 'data/mnist/train-labels-idx1-ubyte'
  test_img_p = 'data/mnist/t10k-images-idx3-ubyte'
  test_lbl_p = 'data/mnist/t10k-labels-idx1-ubyte'

  print('Loading MNIST datasets...')
  train_data = load_mnist_dataset(train_img_p, train_lbl_p, 2500)
  test_data = load_mnist_dataset(test_img_p, test_lbl_p, 1000)
  print(f'Loaded {len(train_data)} train samples and {len(test_data)} test samples.\n')

  brain = RealMNISTBrain()

  epochs = 5
  lr = 0.02
  friction = 0.50

  t0 = time.time()
  for epoch in range(1, epochs + 1):
    total_loss = 0.0
    correct = 0
    for sample in train_data:
      loss = brain.train_sample(sample, lr, friction)
      total_loss += loss

    # Quick test check
    test_correct = 0
    for sample in test_data[:200]:
      probs = brain.forward(sample['pixels'])[0]
      pred = max(range(NUM_CLASSES), key=lambda c: probs[c])
      if pred == sample['label']:
        test_correct += 1

    print(
        f' Epoch {epoch:2d}/{epochs} | Loss: {total_loss / len(train_data):.4f}'
        f' | Test Acc (subset): {test_correct / 200 * 100:.1f}%'
    )

  t1 = time.time()
  print(f'\nPython training finished in {t1 - t0:.1f}s.')
  brain.save_to_file('mnist_cnn_weights_python.txt')
  print('Weights saved to mnist_cnn_weights_python.txt')


if __name__ == '__main__':
  main()

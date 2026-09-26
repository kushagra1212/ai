import math
import sys


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


NUM_FILTERS = 4
NUM_CLASSES = 4


class UnifiedBrain:

  def __init__(self):
    self.stencils = [[[0.0] * 3 for _ in range(3)] for _ in range(NUM_FILTERS)]
    self.biases = [0.0] * NUM_FILTERS
    self.judge_dials = [[0.0] * NUM_FILTERS for _ in range(NUM_CLASSES)]
    self.judge_baselines = [0.0] * NUM_CLASSES

  def load_from_file(self, filepath: str) -> bool:
    tokens = []
    try:
      with open(filepath, 'r') as f:
        for line in f:
          line = line.strip()
          if not line or line.startswith('#'):
            continue
          tokens.extend(line.split())
    except IOError:
      return False

    idx = 0
    for k in range(NUM_FILTERS):
      self.biases[k] = float(tokens[idx])
      idx += 1
      for r in range(3):
        for c in range(3):
          self.stencils[k][r][c] = float(tokens[idx])
          idx += 1

    for k in range(NUM_CLASSES):
      self.judge_baselines[k] = float(tokens[idx])
      idx += 1
      for f in range(NUM_FILTERS):
        self.judge_dials[k][f] = float(tokens[idx])
        idx += 1

    return True

  def forward(self, img):
    peaks = [-1e9] * NUM_FILTERS

    # 1. 3x3 Stencils with smooth ramp
    for f in range(NUM_FILTERS):
      for r in range(8 - 3 + 1):
        for c in range(8 - 3 + 1):
          total = self.biases[f]
          for kr in range(3):
            for kc in range(3):
              total += img[r + kr][c + kc] * self.stencils[f][kr][kc]

          act = smooth_ramp(total)
          if act > peaks[f]:
            peaks[f] = act

    # 2. Judges
    raw_judges = [0.0] * NUM_CLASSES
    for k in range(NUM_CLASSES):
      raw_judges[k] = self.judge_baselines[k] + sum(
          peaks[f] * self.judge_dials[k][f] for f in range(NUM_FILTERS)
      )

    return competing_probabilities(raw_judges), peaks


def make_grid():
  return [[0.0] * 8 for _ in range(8)]


def print_grid(grid):
  for r in range(8):
    row_str = ' '.join(
        ['#' if val > 0.5 else ('.' if val > 0.1 else ' ') for val in grid[r]]
    )
    print(f'      [ {row_str} ]')


def print_probability_bar(label: str, prob: float, is_winner: bool):
  bar_width = 25
  filled = int(prob * bar_width)
  char = '#' if is_winner else '='
  bar = char * filled + ' ' * (bar_width - filled)
  winner_tag = '  <-- WINNER' if is_winner else ''
  print(f'      {label:<8} [{bar}] {prob * 100.0:5.1f}%{winner_tag}')


def main():
  weights_file = (
      sys.argv[1] if len(sys.argv) > 1 else 'grand_unified_weights_python.txt'
  )

  print(
      '===================================================================================='
  )
  print(' PROGRAM 26 TEST SUITE: TESTING LOADED GRAND UNIFIED BRAIN (PYTHON)')
  print(f' Loading Weights from: {weights_file}')
  print(
      '====================================================================================\n'
  )

  brain = UnifiedBrain()
  if not brain.load_from_file(weights_file):
    # Try fallback to grand_unified_weights.txt (from C++)
    if not brain.load_from_file('grand_unified_weights.txt'):
      print(f'ERROR: Failed to load weights from {weights_file}!')
      return

  print(f'>>> Successfully loaded 60 dials from {weights_file}!\n')

  class_names = ['Digit 0', 'Digit 1', 'Digit 2', 'Digit 7']

  # Build Test Suite
  test_suite = []

  # 1. Training Set
  d0 = make_grid()
  for i in range(1, 7):
    d0[1][i] = 1.0
    d0[6][i] = 1.0
    d0[i][1] = 1.0
    d0[i][6] = 1.0
  test_suite.append(
      {'name': 'Standard Box 0', 'target': 0, 'cat': 'Original', 'img': d0}
  )

  d1 = make_grid()
  for r in range(8):
    d1[r][3] = 1.0
  test_suite.append(
      {'name': 'Center Vertical 1', 'target': 1, 'cat': 'Original', 'img': d1}
  )

  d2 = make_grid()
  for c in range(1, 6):
    d2[1][c] = 1.0
  d2[2][5] = 1.0
  d2[3][4] = 1.0
  d2[4][3] = 1.0
  d2[5][2] = 1.0
  for c in range(1, 7):
    d2[6][c] = 1.0
  test_suite.append(
      {'name': 'Standard Digit 2', 'target': 2, 'cat': 'Original', 'img': d2}
  )

  d7 = make_grid()
  for c in range(1, 7):
    d7[1][c] = 1.0
  for r in range(1, 8):
    d7[r][6] = 1.0
  test_suite.append(
      {'name': 'Standard Digit 7', 'target': 3, 'cat': 'Original', 'img': d7}
  )

  # 2. Shift Invariance
  d1_col0 = make_grid()
  for r in range(8):
    d1_col0[r][0] = 1.0
  test_suite.append({
      'name': 'Shifted 1 (Extreme Left Col 0)',
      'target': 1,
      'cat': 'Shift Invariance',
      'img': d1_col0,
  })

  d1_col4 = make_grid()
  for r in range(8):
    d1_col4[r][4] = 1.0
  test_suite.append({
      'name': 'Shifted 1 (Col 4)',
      'target': 1,
      'cat': 'Shift Invariance',
      'img': d1_col4,
  })

  d0_tl = make_grid()
  for i in range(5):
    d0_tl[0][i] = 1.0
    d0_tl[4][i] = 1.0
    d0_tl[i][0] = 1.0
    d0_tl[i][4] = 1.0
  test_suite.append({
      'name': 'Shifted 0 (Small Box Top-Left)',
      'target': 0,
      'cat': 'Shift Invariance',
      'img': d0_tl,
  })

  d7_c = make_grid()
  for c in range(1, 6):
    d7_c[2][c] = 1.0
  for r in range(2, 8):
    d7_c[r][5] = 1.0
  test_suite.append({
      'name': 'Shifted 7 (Center Lowered)',
      'target': 3,
      'cat': 'Shift Invariance',
      'img': d7_c,
  })

  # 3. Broken Strokes
  d1_gap = make_grid()
  d1_gap[0][3] = 1.0
  d1_gap[1][3] = 1.0
  d1_gap[4][3] = 1.0
  d1_gap[5][3] = 1.0
  d1_gap[6][3] = 1.0
  d1_gap[7][3] = 1.0
  test_suite.append({
      'name': 'Broken 1 (Missing Middle Gap)',
      'target': 1,
      'cat': 'Broken Strokes',
      'img': d1_gap,
  })

  d7_gap = make_grid()
  d7_gap[1][1] = 1.0
  d7_gap[1][4] = 1.0
  d7_gap[1][5] = 1.0
  for r in range(1, 8):
    d7_gap[r][5] = 1.0
  test_suite.append({
      'name': 'Broken 7 (Gap in Top Bar)',
      'target': 3,
      'cat': 'Broken Strokes',
      'img': d7_gap,
  })

  # 4. Edge Cases
  blank = make_grid()
  test_suite.append(
      {'name': 'Blank Canvas (0 Pixels)', 'target': -1, 'cat': 'Edge Case', 'img': blank}
  )

  # Execute
  passed = 0
  testable_count = 0

  for i, test in enumerate(test_suite):
    probs, peaks = brain.forward(test['img'])
    best = max(range(NUM_CLASSES), key=lambda k: probs[k])

    ok = (test['target'] == -1) or (best == test['target'])
    if test['target'] != -1:
      testable_count += 1
      if ok:
        passed += 1

    print(
        '===================================================================================='
    )
    print(f" TEST {i + 1} / {len(test_suite)}: {test['name']}")
    expected_str = (
        f" | Expected: {class_names[test['target']]}"
        if test['target'] != -1
        else ''
    )
    print(f" Category: {test['cat']}{expected_str}")
    print(
        '------------------------------------------------------------------------------------'
    )
    print_grid(test['img'])
    print()

    print(
        '      Filter Peaks: ['
        f' {" ".join([f"F{f}:{peaks[f]:.1f}" for f in range(NUM_FILTERS)])} ]\n'
    )

    for k in range(NUM_CLASSES):
      print_probability_bar(class_names[k], probs[k], k == best)

    if test['target'] == -1:
      print(
          f'\n      VERDICT: [INFO] Defaulted to {class_names[best]} with'
          f' {probs[best] * 100.0:.1f}% confidence.'
      )
    elif ok:
      print(
          f'\n      VERDICT: [PASS] Correctly classified as {class_names[best]}'
          f' ({probs[best] * 100.0:.1f}% confidence)!'
      )
    else:
      print(
          f"\n      VERDICT: [FAIL] Misclassified! Expected {class_names[test['target']]} but predicted {class_names[best]}"
      )

  print(
      '\n===================================================================================='
  )
  print(
      f' SUMMARY: Passed {passed} / {testable_count} ({passed * 100.0 / testable_count:.1f}%)'
  )
  print(
      '====================================================================================\n'
  )


if __name__ == '__main__':
  main()

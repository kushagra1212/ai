import math


def squash(z: float) -> float:
  z = max(-50.0, min(50.0, z))
  return 1.0 / (1.0 + math.exp(-z))


def smooth_ramp(z: float) -> float:
  return z * squash(z)


vertical_stencil = [[-0.2, 0.6, -0.2], [-0.2, 0.6, -0.2], [-0.2, 0.6, -0.2]]


def score_patch(patch):
  total = 0.0
  for r in range(3):
    for c in range(3):
      total += patch[r][c] * vertical_stencil[r][c]
  return total


if __name__ == '__main__':
  solid = [[0.0, 1.0, 0.0], [0.0, 1.0, 0.0], [0.0, 1.0, 0.0]]

  broken = [
      [0.0, 1.0, 0.0],
      [0.0, 0.0, 0.0],  # Gap!
      [0.0, 1.0, 0.0],
  ]

  empty = [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]

  print(
      f'1. Solid Line:  Raw = {score_patch(solid):.2f}, Activation ='
      f' {smooth_ramp(score_patch(solid)):.2f}'
  )
  print(
      f'2. Broken Line: Raw = {score_patch(broken):.2f}, Activation ='
      f' {smooth_ramp(score_patch(broken)):.2f}'
  )
  print(
      f'3. Empty Space: Raw = {score_patch(empty):.2f}, Activation ='
      f' {smooth_ramp(score_patch(empty)):.2f}'
  )

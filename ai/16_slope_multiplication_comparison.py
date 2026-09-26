import math


def sigmoid_slope(z: float) -> float:
  a = 1.0 / (1.0 + math.exp(-z))
  return a * (1.0 - a)


def relu_slope(z: float) -> float:
  return 1.0 if z > 0.0 else 0.0


def leaky_relu_slope(z: float, alpha: float = 0.01) -> float:
  return 1.0 if z > 0.0 else alpha


def propagate_blame_sigmoid(
    initial_blame: float, raw_scores: list[float]
) -> float:
  blame = initial_blame
  for z in raw_scores:
    blame *= sigmoid_slope(z)
  return blame


def propagate_blame_relu(initial_blame: float, raw_scores: list[float]) -> float:
  blame = initial_blame
  for z in raw_scores:
    blame *= relu_slope(z)
  return blame


def propagate_blame_leaky(
    initial_blame: float, raw_scores: list[float], alpha: float = 0.01
) -> float:
  blame = initial_blame
  for z in raw_scores:
    blame *= leaky_relu_slope(z, alpha)
  return blame


if __name__ == "__main__":
  initial_blame = 1.0

  # Scenario 1: All active
  active_scores = [1.5, 0.8, 2.1, 0.5]
  sig1 = propagate_blame_sigmoid(initial_blame, active_scores)
  relu1 = propagate_blame_relu(initial_blame, active_scores)
  leaky1 = propagate_blame_leaky(initial_blame, active_scores)

  print("SCENARIO 1: All active [1.5, 0.8, 2.1, 0.5]")
  print(f"  S-Curve (Sigmoid)   : {sig1:.6f}")
  print(f"  Standard Ramp (ReLU): {relu1:.6f}")
  print(f"  Leaky Ramp          : {leaky1:.6f}")
  print()

  # Scenario 2: One negative
  mixed_scores = [1.5, 0.8, -1.2, 0.5]
  sig2 = propagate_blame_sigmoid(initial_blame, mixed_scores)
  relu2 = propagate_blame_relu(initial_blame, mixed_scores)
  leaky2 = propagate_blame_leaky(initial_blame, mixed_scores)

  print("SCENARIO 2: One negative [1.5, 0.8, -1.2, 0.5]")
  print(f"  S-Curve (Sigmoid)   : {sig2:.6f}")
  print(f"  Standard Ramp (ReLU): {relu2:.6f}  <-- Completely severed!")
  print(f"  Leaky Ramp          : {leaky2:.6f}  <-- Still alive!")

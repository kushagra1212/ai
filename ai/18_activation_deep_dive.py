import math


def sigmoid(z: float) -> float:
  return 1.0 / (1.0 + math.exp(-z))


def sigmoid_slope(z: float) -> float:
  a = sigmoid(z)
  return a * (1.0 - a)


def tanh_act(z: float) -> float:
  return math.tanh(z)


def tanh_slope(z: float) -> float:
  a = math.tanh(z)
  return 1.0 - a * a


def relu(z: float) -> float:
  return z if z > 0.0 else 0.0


def relu_slope(z: float) -> float:
  return 1.0 if z > 0.0 else 0.0


def leaky_relu(z: float, alpha: float = 0.01) -> float:
  return z if z > 0.0 else alpha * z


def leaky_relu_slope(z: float, alpha: float = 0.01) -> float:
  return 1.0 if z > 0.0 else alpha


def silu(z: float) -> float:
  return z * sigmoid(z)


def silu_slope(z: float) -> float:
  s = sigmoid(z)
  return s + z * s * (1.0 - s)


if __name__ == "__main__":
  test_z = [-3.0, -1.0, 0.0, 1.0, 3.0]
  header = f"{'Raw (z)':>8} {'Sigmoid Slope':>15} {'Tanh Slope':>15} {'ReLU Slope':>15} {'Leaky Slope':>15} {'SiLU Slope':>15}"
  print(header)
  print("-" * len(header))
  for z in test_z:
    print(
        f"{z:8.4f} {sigmoid_slope(z):15.4f} {tanh_slope(z):15.4f}"
        f" {relu_slope(z):15.4f} {leaky_relu_slope(z):15.4f}"
        f" {silu_slope(z):15.4f}"
    )

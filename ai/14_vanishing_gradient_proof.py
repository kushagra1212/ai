# Python side-by-side equivalent: Vanishing Gradient Proof
import math

def squash_slope(a: float) -> float:
    return a * (1.0 - a)

def ramp_slope(z: float) -> float:
    return 1.0 if z > 0.0 else 0.0

if __name__ == "__main__":
    initial_error = 1.0

    print("--- METHOD 1: Using S-CURVE (Sigmoid) ---")
    blame_sigmoid = initial_error
    for layer in range(5, 0, -1):
        blame_sigmoid *= squash_slope(0.5) # 0.25
        print(f"  Layer {layer}: Blame = {blame_sigmoid:.6f}")

    print("\n--- METHOD 2: Using THE RAMP (ReLU) ---")
    blame_ramp = initial_error * squash_slope(0.5)
    print(f"  Layer 5: Blame = {blame_ramp:.6f}")
    for layer in range(4, 0, -1):
        blame_ramp *= ramp_slope(1.0) # 1.0
        print(f"  Layer {layer}: Blame = {blame_ramp:.6f}")

    print(f"\nAt Layer 1, The Ramp is {blame_ramp / blame_sigmoid:.1f}x stronger!")

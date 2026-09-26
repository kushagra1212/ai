# -----------------------------------------------------------------------------
# FIRST PRINCIPLES: SLIDING STENCIL (CONVOLUTION)
# -----------------------------------------------------------------------------


def convolve_2d(
    image: list[list[float]], stencil: list[list[float]]
) -> list[list[float]]:
  H = len(image)
  W = len(image[0])
  K_rows = len(stencil)
  K_cols = len(stencil[0])

  out_H = H - K_rows + 1
  out_W = W - K_cols + 1

  output = [[0.0 for _ in range(out_W)] for _ in range(out_H)]

  for r in range(out_H):
    for c in range(out_W):
      total = 0.0
      for kr in range(K_rows):
        for kc in range(K_cols):
          total += image[r + kr][c + kc] * stencil[kr][kc]
      output[r][c] = total

  return output


def print_matrix(title: str, mat: list[list[float]]) -> None:
  print(title)
  for row in mat:
    line = "  "
    for val in row:
      line += f"{val:6.1f} "
    print(line)
  print()


if __name__ == "__main__":
  img_center = [
      [0, 0, 1, 0, 0],
      [0, 0, 1, 0, 0],
      [0, 0, 1, 0, 0],
      [0, 0, 1, 0, 0],
      [0, 0, 1, 0, 0],
  ]

  img_shifted = [
      [0, 1, 0, 0, 0],
      [0, 1, 0, 0, 0],
      [0, 1, 0, 0, 0],
      [0, 1, 0, 0, 0],
      [0, 1, 0, 0, 0],
  ]

  vertical_stencil = [
      [-1.0, 2.0, -1.0],
      [-1.0, 2.0, -1.0],
      [-1.0, 2.0, -1.0],
  ]

  print_matrix("IMAGE A: 5x5 (Vertical Line at Column 2)", img_center)
  print_matrix(
      "IMAGE B: 5x5 (Vertical Line Shifted to Column 1)", img_shifted
  )
  print_matrix("3x3 VERTICAL STENCIL (9 Dials)", vertical_stencil)

  result_A = convolve_2d(img_center, vertical_stencil)
  result_B = convolve_2d(img_shifted, vertical_stencil)

  print_matrix("OUTPUT A: Response Grid for Image A (3x3)", result_A)
  print_matrix("OUTPUT B: Response Grid for Image B (3x3)", result_B)

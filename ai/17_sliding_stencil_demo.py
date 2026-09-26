# -----------------------------------------------------------------------------
# FIRST-PRINCIPLES SLIDING STENCIL (CONVOLUTION)
# -----------------------------------------------------------------------------

VERTICAL_STENCIL = [[-1.0, 2.0, -1.0], [-1.0, 2.0, -1.0], [-1.0, 2.0, -1.0]]


def apply_sliding_stencil(
    image: list[list[float]], stencil: list[list[float]]
) -> list[list[float]]:
  img_rows = len(image)
  img_cols = len(image[0])
  sten_rows = len(stencil)
  sten_cols = len(stencil[0])

  out_rows = img_rows - sten_rows + 1  # 8 - 3 + 1 = 6
  out_cols = img_cols - sten_cols + 1  # 8 - 3 + 1 = 6

  output = [[0.0 for _ in range(out_cols)] for _ in range(out_rows)]

  for r in range(out_rows):
    for c in range(out_cols):
      total = 0.0
      for sr in range(sten_rows):
        for sc in range(sten_cols):
          total += image[r + sr][c + sc] * stencil[sr][sc]
      output[r][c] = total

  return output


def print_grid(title: str, grid: list[list[float]]) -> None:
  print(title)
  for row in grid:
    line = "  "
    for val in row:
      if val > 0.5 or val < -0.5:
        line += f"{val:5.1f} "
      else:
        line += "    . "
    print(line)
  print()


if __name__ == "__main__":
  center_one = [
      [0, 0, 0, 1, 1, 0, 0, 0],
      [0, 0, 0, 1, 1, 0, 0, 0],
      [0, 0, 0, 1, 1, 0, 0, 0],
      [0, 0, 0, 1, 1, 0, 0, 0],
      [0, 0, 0, 1, 1, 0, 0, 0],
      [0, 0, 0, 1, 1, 0, 0, 0],
      [0, 0, 0, 1, 1, 0, 0, 0],
      [0, 0, 0, 1, 1, 0, 0, 0],
  ]

  shifted_one = [
      [0, 1, 1, 0, 0, 0, 0, 0],
      [0, 1, 1, 0, 0, 0, 0, 0],
      [0, 1, 1, 0, 0, 0, 0, 0],
      [0, 1, 1, 0, 0, 0, 0, 0],
      [0, 1, 1, 0, 0, 0, 0, 0],
      [0, 1, 1, 0, 0, 0, 0, 0],
      [0, 1, 1, 0, 0, 0, 0, 0],
      [0, 1, 1, 0, 0, 0, 0, 0],
  ]

  print_grid("IMAGE 1: Center '1' (8x8)", center_one)
  print_grid("IMAGE 2: Shifted-Left '1' (8x8)", shifted_one)

  result_center = apply_sliding_stencil(center_one, VERTICAL_STENCIL)
  result_shifted = apply_sliding_stencil(shifted_one, VERTICAL_STENCIL)

  print_grid(
      "OUTPUT 1: Stencil Responses for Center '1' (6x6)", result_center
  )
  print_grid(
      "OUTPUT 2: Stencil Responses for Shifted-Left '1' (6x6)", result_shifted
  )

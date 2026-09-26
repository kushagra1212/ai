# Python side-by-side equivalent
from collections import deque
from typing import List, Dict, Any

def find_islands(image: List[List[int]]) -> List[Dict[str, Any]]:
    if not image or not image[0]:
        return []

    rows, cols = len(image), len(image[0])
    visited = [[False]*cols for _ in range(rows)]
    islands = []
    current_id = 1

    for r in range(rows):
        for c in range(cols):
            if image[r][c] == 1 and not visited[r][c]:
                q = deque([(r, c)])
                visited[r][c] = True
                pixels = []
                min_r, max_r = r, r
                min_c, max_c = c, c

                while q:
                    curr_r, curr_c = q.popleft()
                    pixels.append((curr_r, curr_c))
                    min_r, max_r = min(min_r, curr_r), max(max_r, curr_r)
                    min_c, max_c = min(min_c, curr_c), max(max_c, curr_c)

                    for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                        nr, nc = curr_r + dr, curr_c + dc
                        if 0 <= nr < rows and 0 <= nc < cols:
                            if image[nr][nc] == 1 and not visited[nr][nc]:
                                visited[nr][nc] = True
                                q.append((nr, nc))

                islands.append({
                    "id": current_id,
                    "pixels": pixels,
                    "min_r": min_r, "max_r": max_r,
                    "min_c": min_c, "max_c": max_c,
                    "width": max_c - min_c + 1,
                    "height": max_r - min_r + 1
                })
                current_id += 1
    return islands

def classify_shape(island: Dict[str, Any], image: List[List[int]]) -> str:
    width = island["width"]
    height = island["height"]
    area = len(island["pixels"])
    box_area = width * height

    if width < 3 or height < 3:
        return "Not Identifiable (too small / line)"

    fill_ratio = area / box_area
    aspect_ratio = width / height

    # 1. Solid Box Check
    if fill_ratio >= 0.88:
        if 0.80 <= aspect_ratio <= 1.25:
            return "Square"
        else:
            return "Rectangle"

    # 2. Triangle Check
    if 0.40 <= fill_ratio <= 0.65:
        row_counts = [0] * height
        for r, c in island["pixels"]:
            row_counts[r - island["min_r"]] += 1

        pointing_up = all(row_counts[i] <= row_counts[i+1] for i in range(height - 1))
        pointing_down = all(row_counts[i] >= row_counts[i+1] for i in range(height - 1))

        if pointing_up or pointing_down:
            return "Triangle"

    # 3. Circle Check
    if 0.80 <= aspect_ratio <= 1.25 and 0.65 <= fill_ratio <= 0.85:
        corners = [
            image[island["min_r"]][island["min_c"]],
            image[island["min_r"]][island["max_c"]],
            image[island["max_r"]][island["min_c"]],
            image[island["max_r"]][island["max_c"]]
        ]
        if all(c == 0 for c in corners):
            return "Circle"

    return "Not Identifiable"

if __name__ == "__main__":
    canvas = [
        [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
        [0, 1,1,1,1, 0,0, 1,1,1,1,1,1, 0,0,0,0,0,0,0],
        [0, 1,1,1,1, 0,0, 1,1,1,1,1,1, 0,0,0,0,0,0,0],
        [0, 1,1,1,1, 0,0, 1,1,1,1,1,1, 0,0,0,0,0,0,0],
        [0, 1,1,1,1, 0,0, 0,0,0,0,0,0, 0,0,0,0,0,0,0],
        [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
        [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
        [0, 0,0,1,0,0, 0,0, 0,1,1,1,0, 0,0, 0,1,1,0, 0],
        [0, 0,1,1,1,0, 0,0, 1,1,1,1,1, 0,0, 0,0,1,0, 0],
        [0, 1,1,1,1,1, 0,0, 1,1,1,1,1, 0,0, 0,0,1,0, 0],
        [0, 0,0,0,0,0, 0,0, 1,1,1,1,1, 0,0, 0,0,1,0, 0],
        [0, 0,0,0,0,0, 0,0, 0,1,1,1,0, 0,0, 1,1,1,1, 0],
        [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
    ]

    islands = find_islands(canvas)
    print(f"Total Islands Found: {len(islands)}")
    for isl in islands:
        shape = classify_shape(isl, canvas)
        print(f"Island #{isl['id']} ({isl['width']}x{isl['height']}) ===> [{shape}]")

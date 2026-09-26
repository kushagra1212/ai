# Python equivalent for comparison
from collections import deque
from typing import List, Tuple, Dict, Any

def find_islands(image: List[List[int]]) -> List[Dict[str, Any]]:
    if not image or not image[0]:
        return []

    rows, cols = len(image), len(image[0])
    visited = [[False for _ in range(cols)] for _ in range(rows)]
    islands = []
    island_id = 1

    for r in range(rows):
        for c in range(cols):
            # Found an unvisited black pixel
            if image[r][c] == 1 and not visited[r][c]:
                # Explore island with BFS (queue)
                q = deque([(r, c)])
                visited[r][c] = True
                pixels = []
                min_r, max_r = r, r
                min_c, max_c = c, c

                while q:
                    curr_r, curr_c = q.popleft()
                    pixels.append((curr_r, curr_c))

                    min_r = min(min_r, curr_r)
                    max_r = max(max_r, curr_r)
                    min_c = min(min_c, curr_c)
                    max_c = max(max_c, curr_c)

                    # Up, Down, Left, Right
                    for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                        nr, nc = curr_r + dr, curr_c + dc
                        if 0 <= nr < rows and 0 <= nc < cols:
                            if image[nr][nc] == 1 and not visited[nr][nc]:
                                visited[nr][nc] = True
                                q.append((nr, nc))

                islands.append({
                    "id": island_id,
                    "pixels": pixels,
                    "min_r": min_r, "max_r": max_r,
                    "min_c": min_c, "max_c": max_c,
                    "width": max_c - min_c + 1,
                    "height": max_r - min_r + 1
                })
                island_id += 1

    return islands

if __name__ == "__main__":
    screen = [
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 1, 1, 0, 0, 0, 1, 0, 0, 0],
        [0, 1, 1, 0, 0, 0, 1, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 1, 1, 1, 0, 0, 0, 0, 0],
        [0, 0, 1, 0, 1, 0, 0, 0, 0, 0],
        [0, 0, 1, 1, 1, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    ]

    islands = find_islands(screen)
    print(f"Found {len(islands)} black islands!")
    for isl in islands:
        print(f"Island #{isl['id']}: Area={len(isl['pixels'])}, Width={isl['width']}, Height={isl['height']}")

import heapq
from collections import deque

def get_neighbors(x, y, grid):
    neighbors = []
    rows, cols = len(grid), len(grid[0])
    for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
        nx, ny = x + dx, y + dy
        if 0 <= nx < rows and 0 <= ny < cols:
            neighbors.append((nx, ny))
    return neighbors

def shortest_path(grid, start, end):
    """Finds the shortest path using BFS (ignoring weights)."""
    queue = deque([(start, [start])])
    visited = set([start])
    
    while queue:
        (x, y), path = queue.popleft()
        if (x, y) == end:
            return path
        
        for nx, ny in get_neighbors(x, y, grid):
            if (nx, ny) not in visited:
                visited.add((nx, ny))
                queue.append(((nx, ny), path + [(nx, ny)]))
    
    return None  # No path found

def safest_path(grid, start, end):
    """Finds the path with the minimum risk using Dijkstra's algorithm."""
    rows, cols = len(grid), len(grid[0])
    pq = [(grid[start[0]][start[1]], start, [start])]
    visited = set()
    
    while pq:
        risk, (x, y), path = heapq.heappop(pq)
        
        if (x, y) == end:
            return path
        
        if (x, y) in visited:
            continue
        visited.add((x, y))
        
        for nx, ny in get_neighbors(x, y, grid):
            if (nx, ny) not in visited:
                heapq.heappush(pq, (risk + grid[nx][ny], (nx, ny), path + [(nx, ny)]))
    
    return None  # No path found

# Example usage
grid = [
    [0, 0, 1, 0],
    [1, 2, 2, 2],
    [0, 0, 1, 0],
    [0, 1, 0, 0]
]
start = (0, 0)
end = (3, 3)

print("Shortest Path:", shortest_path(grid, start, end))
print("Safest Path:", safest_path(grid, start, end))
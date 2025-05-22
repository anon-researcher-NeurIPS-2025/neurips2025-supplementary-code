import numpy as np

def phi(state_vector):
    """
    Extracts handcrafted features from the state vector.
    The state is expected to have the format:
    [x1, x2, y1, y2, m0, m1, ..., m63] where (x1, x2) are Agent 1's coordinates,
    (y1, y2) are Agent 2's coordinates, and m0–m63 are the flattened 8x8 apple grid (0 or 1).
    
    Returns:
        np.ndarray: Vector of handcrafted features.
    """
    x1, x2, y1, y2 = state_vector[:4]
    apple_grid = np.array(state_vector[4:]).reshape(8, 8)
    apples = np.argwhere(apple_grid == 1)

    # Feature 1: Total remaining apples
    f1 = len(apples)

    # Agent positions
    pos_a1 = np.array([x1, x2])
    pos_a2 = np.array([y1, y2])

    # Feature 2: Distance from agent 1 to the nearest apple
    # Feature 3: Distance from agent 2 to the nearest apple
    # Feature 4: Absolute difference in distances to nearest apple
    dists_a1 = [np.linalg.norm(pos_a1 - np.array(p)) for p in apples] if apples.size > 0 else [0]
    dists_a2 = [np.linalg.norm(pos_a2 - np.array(p)) for p in apples] if apples.size > 0 else [0]
    f2 = min(dists_a1)
    f3 = min(dists_a2)
    f4 = abs(f2 - f3)

    # Local apple count in the 3x3 area around each agent
    def count_apples_near(pos):
        x, y = pos
        return sum(
            apple_grid[nx, ny]
            for dx in [-1, 0, 1]
            for dy in [-1, 0, 1]
            if 0 <= (nx := x + dx) < 8 and 0 <= (ny := y + dy) < 8
        )

    # Feature 5: Apples near agent 1
    # Feature 6: Apples near agent 2
    f5 = count_apples_near(pos_a1)
    f6 = count_apples_near(pos_a2)

    return np.array([f1, f2, f3, f4, f5, f6])

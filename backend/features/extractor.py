import numpy as np

def extract_features(angles):
    if len(angles) < 10:
        return [0] * 18

    angles = np.array(angles)

    # Raw angle stats
    mean_a = np.mean(angles)
    std_a = np.std(angles)
    min_a = np.min(angles)
    max_a = np.max(angles)
    range_a = max_a - min_a
    vel_a = np.mean(np.abs(np.diff(angles)))

    # Smoothed angles
    smooth = np.convolve(angles, np.ones(5)/5, mode="valid")

    mean_s = np.mean(smooth)
    std_s = np.std(smooth)
    min_s = np.min(smooth)
    max_s = np.max(smooth)
    range_s = max_s - min_s
    vel_s = np.mean(np.abs(np.diff(smooth)))

    # Movement dynamics
    total_movement = np.sum(np.abs(np.diff(angles)))
    peak_velocity = np.max(np.abs(np.diff(angles)))
    zero_crossings = np.sum(np.diff(np.sign(np.diff(angles))) != 0)
    active_ratio = np.count_nonzero(np.abs(np.diff(angles)) > 1) / len(angles)
    consistency = 1 / (1 + std_a)
    normalized_movement = total_movement / len(angles)

    return [
        mean_a, std_a, min_a, max_a, range_a, vel_a,
        mean_s, std_s, min_s, max_s, range_s, vel_s,
        total_movement, peak_velocity, zero_crossings,
        active_ratio, consistency, normalized_movement
    ]

import numpy as np

def smoothstep(t):
    t = np.clip(t, 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)

def interpolate_heights(first_height, second_height, distance, blend_width):
    t = distance / blend_width
    t = smoothstep(t)

    return (1.0 - t) * first_height + t * second_height
point interpolation


Place peaks at (x, y) with desired z.

Build a base surface with natural neighbour, RBF, or IDW, ensuring it passes through the peak heights.

Compute influence fields from each peak using a falloff such as linear, inverse-square, or Gaussian.

Find balance regions where two or more fields have similar strength.

Subtract a crevice field there, modulated by noise and a distance falloff.

Smooth lightly, but protect the original peak heights.

Add high-frequency detail and optionally hydraulic erosion for final realism.



1. Morphological dilation per level (the literal version)
Keep a boolean mask of “cells at or above z”. For each level, dilate the mask with a disk of radius Δz / slope, and give the newly added cells the height z. It’s very simple, and it’s easy to randomize: dilate with a random radius per step, or let each frontier cell claim its neighbours with some probability. Downsides are staircase artifacts and a fixed relationship between Δz and the grid. If Δz / slope is less than about one cell, nothing grows, so you must choose Δz to match your resolution.

2. Eikonal equation / fast marching
Level-by-level growth with a constant slope is exactly the eikonal equation |∇h| = s. The height map is then z0 − s · (geodesic distance from the peak), and with several peaks the highest value wins, as in your np.maximum. Fast marching (e.g. scikit-fmm) solves this directly, and it lets s vary across space, say from a smooth noise field (“hard rock” versus “loose scree”). It doesn’t handle direction-dependent slopes out of the box, since that needs an anisotropic version of the equation. It’s fast and clean, but it’s a black box compared to approach 1 or 3.

3. Priority-queue (Dijkstra-style) region growing
Put the peak cells in a priority queue keyed by height. Repeatedly pop the highest cell and set each unvisited neighbour’s height to h − slope(direction, state) · step_length · (1 + noise). Processing cells from the highest down is exactly your “z from the top to 0” iteration, without discretizing z. Because the noise enters at every step and accumulates along each path, you get organic ridges and valleys instead of white noise on top of smooth cones. It supports direction-dependent slopes naturally, since each edge has a known direction. It’s a Python loop over all cells, so use numba if you go past roughly 500×500.

4. Slope that depends on the existing profile (feedback)
This most directly matches “extend from the existing profile”. At each level, measure something about the current frontier and use it to set the next drop, for example:

the local curvature of the contour (convex tips steeper, concave notches gentler),
the gradient of the previous layer,
a talus-angle rule, as in thermal erosion: no cell may be more than a maximum slope above its neighbour.

This gives the most natural-looking terrain, but also the most tuning.

5. Vector contours
Keep each level as a polygon. The next contour is previous.buffer(Δz / slope) (with shapely), plus a smooth periodic random offset along the outline so it isn’t a perfect offset curve. Overlapping peaks merge automatically through the polygon union. Then rasterize or triangulate the stack of contours. It’s resolution-independent and good for looks, but meshing between contours is the awkward part.

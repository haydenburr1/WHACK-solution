point interpolation


Place peaks at (x, y) with desired z.

Build a base surface with natural neighbour, RBF, or IDW, ensuring it passes through the peak heights.

Compute influence fields from each peak using a falloff such as linear, inverse-square, or Gaussian.

Find balance regions where two or more fields have similar strength.

Subtract a crevice field there, modulated by noise and a distance falloff.

Smooth lightly, but protect the original peak heights.

Add high-frequency detail and optionally hydraulic erosion for final realism.


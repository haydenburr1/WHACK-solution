
smooth_noise is a Gaussian blur of white noise, done in frequency space. I’ll go through the math in order and give the numbers from a quick numerical check at the end.

1. White noise

rng.normal(size=(n, n)) gives independent values w(x) with mean 0 and variance 1:

E[w(x) w(y)] = 1 if x = y, else 0

Because every cell is independent, the value at one cell tells you nothing about its neighbours. That’s the grain.

2. Smoothing = convolution with a Gaussian

To make neighbouring values similar, replace each value by a weighted average of its surroundings:

field(x) = Σ_y g(x − y) w(y),     g(r) = 1/(2πσ²) · exp(−|r|² / (2σ²))

g is a Gaussian kernel with standard deviation σ (in cells, so scale=15 means σ = 15). Doing this directly costs about n² × (kernel size) operations, which is slow for large σ.

3. The convolution theorem

The Fourier transform turns a convolution into a plain product:

F{ w * g } = F{w} · F{g}

So instead of averaging, you transform the noise, multiply it by F{g} at every frequency, and transform back. With the FFT this costs about n² log n, independent of σ.

4. The Fourier transform of a Gaussian

With the convention ĝ(f) = ∫ g(x) e^(−2πi f·x) dx, the transform of a Gaussian of standard deviation σ is again a Gaussian, with width inverted. In 1D:

∫ (1/(σ√2π)) e^(−x²/2σ²) e^(−2πi f x) dx = e^(−2π²σ²f²)

In 2D the kernel factorises, so with f² = fx² + fy²:

ĝ(f) = exp(−2π² σ² (fx² + fy²))

This is exactly np.exp(-2 * np.pi**2 * scale**2 * f2). The shape matters: it equals 1 at frequency 0 and decays rapidly as f grows, so it keeps slow variations and removes fast ones. A wider blur (bigger σ) gives a narrower filter in frequency space.

5. The discrete version

np.fft.fftfreq(n) returns k/n for k = 0, 1, ..., n/2−1, −n/2, ..., −1. These are frequencies in cycles per cell, ranging from −0.5 to 0.5 (the Nyquist limit).
fftfreq(n)[:, None]**2 + fftfreq(n)[None, :]**2 uses broadcasting to build the n×n array f2[k, l] = fx[k]² + fy[l]².
fft2 computes X[k,l] = Σ x[m,n] e^(−2πi(km + ln)/n), and ifft2 inverts it (with a 1/n² factor).

Together the code computes field = ifft2( fft2(w) · H ) with H[k,l] = exp(−2π²σ² f2[k,l]).

Two caveats:

Periodicity: the DFT treats the array as repeating, so this is a circular convolution. The field wraps around, and the right edge continues into the left edge.
Sampling: the continuous Gaussian formula is accurate when σ ≳ 1 cell. For smaller σ the filter is cut off by the Nyquist limit (0.5) and no longer matches a true Gaussian blur.

6. Why .real

H is real and symmetric (H[k] = H[−k]), and w is real, so the exact result is real. The imaginary part left over is rounding error, about 1e−17 in my test, and .real discards it.

7. Why divide by the standard deviation

Blurring averages many independent values, so the variance shrinks. By Parseval’s theorem:

Var(field) = (1/n²) Σ_{k,l} H[k,l]²  ≈  ∫ exp(−4π²σ²f²) d²f  =  π / (4π²σ²)  =  1/(4πσ²)

So the standard deviation is about 1/(2√π σ) ≈ 0.282/σ: roughly 0.056 for σ = 5 and 0.019 for σ = 15. A raw blurred field is tiny, and the more it’s smoothed the smaller it gets. Dividing by field.std() rescales it to standard deviation 1, which makes field_strength and the layer weights mean the same thing at every scale.

8. Statistics of the result

Each field value is a weighted sum of independent Gaussians, so the field is a Gaussian random field: every value is N(0, 1) after the normalisation, and any set of values is jointly Gaussian. What distinguishes it from white noise is the covariance between two points a distance r apart:

C(r) = (g ⋆ g)(r)       (the autocorrelation of the kernel, since w is white)

Its transform is |ĝ|² = exp(−4π²σ²f²), which is the transform of a Gaussian with variance 2σ². Therefore:

C(r) = exp(−r² / (4σ²))        (after normalising so C(0) = 1)

So the correlation drops to 1/e at r = 2σ, not σ. The scale parameter is a correlation length only up to that factor of 2, and the features you see are about 2σ wide.

Numerical check

I generated fields on a 512×512 grid and compared them with these formulas:

σ	raw std (measured)	theory 1/(2√π σ)	corr at r = σ (measured / theory)	at r = 2σ
5	0.056	0.0564	0.77 / 0.78	0.34 / 0.37
15	0.018	0.0188	0.77 / 0.78	0.31 / 0.37

At r = 4σ the correlation was about 0, as predicted. The small differences are sampling error from using a single realization.

9. Summing layers

In field = Σ w_i · smooth_noise(s_i), the layers are independent. Their variances add, giving Σ w_i² (1.25 for weights 1.0 and 0.5), and their covariances add. The result has structure at each scale in proportion to its weight.


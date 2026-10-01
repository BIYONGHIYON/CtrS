"""Conservative per-scene RGB translation; HSI geometry and spectra stay unchanged."""
import numpy as np
from scipy.ndimage import gaussian_filter, sobel, shift
from scipy.signal import fftconvolve


def edges(image):
    gray = np.asarray(image, dtype=np.float32).mean(axis=2)
    gray = gaussian_filter(gray, 1)
    return np.hypot(sobel(gray, axis=0), sobel(gray, axis=1))


def score(a, b):
    a, b = a.ravel().astype(float), b.ravel().astype(float)
    a, b = a-a.mean(), b-b.mean()
    return float(np.dot(a, b) / max(np.linalg.norm(a)*np.linalg.norm(b), 1e-12))


def estimate(reference, moving, radius=12):
    """Return integer shift applied to RGB, accepted only with spatial consensus."""
    a, b = edges(reference), edges(moving)
    def candidate(a, b):
        a, b = a-a.mean(), b-b.mean()
        corr = fftconvolve(a, b[::-1, ::-1], mode='full')
        cy, cx = np.array(b.shape)-1
        region = corr[cy-radius:cy+radius+1, cx-radius:cx+radius+1]
        y, x = np.unravel_index(np.argmax(region), region.shape)
        return int(y-radius), int(x-radius)
    dy, dx = candidate(a, b)
    h, w = a.shape
    locals_ = [candidate(a[y:y+h//2, x:x+w//2], b[y:y+h//2, x:x+w//2])
               for y in (0, h//2) for x in (0, w//2)]
    support = sum(abs(y-dy)<=2 and abs(x-dx)<=2 for y,x in locals_)
    m = radius+2
    before = score(a[m:-m,m:-m], b[m:-m,m:-m])
    corrected = shift(b, (dy,dx), order=0, mode='constant', cval=0, prefilter=False)
    after = score(a[m:-m,m:-m], corrected[m:-m,m:-m])
    accepted = support>=3 and after>=0.25 and after-before>=0.01 and max(abs(dy),abs(dx))<radius
    return dict(dy=dy if accepted else 0, dx=dx if accepted else 0,
                candidate_dy=dy, candidate_dx=dx, accepted=bool(accepted),
                before=before, after_candidate=after, support=support)


def warp_rgb(rgb, dy, dx):
    aligned = shift(rgb, (dy,dx,0), order=0, mode='constant', cval=0, prefilter=False)
    valid = shift(np.ones(rgb.shape[:2],dtype=np.uint8), (dy,dx), order=0,
                  mode='constant', cval=0, prefilter=False).astype(bool)
    return aligned, valid

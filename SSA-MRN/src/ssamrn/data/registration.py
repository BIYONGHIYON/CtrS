"""Per-scene RGB translation/projective alignment; HSI is the fixed reference."""
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


def warp_projective(rgb,matrix):
    """Matrix maps moving RGB coordinates to fixed HSI coordinates."""
    import cv2
    matrix=np.asarray(matrix,dtype=np.float64)
    if matrix.shape!=(3,3) or not np.isfinite(matrix).all() or abs(np.linalg.det(matrix))<1e-8:
        raise ValueError('Invalid homography')
    if np.allclose(matrix,np.eye(3),atol=1e-9):
        return rgb.copy(),np.ones(rgb.shape[:2],dtype=bool)
    h,w=rgb.shape[:2]
    aligned=cv2.warpPerspective(rgb,matrix,(w,h),flags=cv2.INTER_LANCZOS4,borderMode=cv2.BORDER_CONSTANT)
    mask=np.zeros((h,w),dtype=np.uint8);mask[4:-4,4:-4]=1
    valid=cv2.warpPerspective(mask,matrix,(w,h),flags=cv2.INTER_NEAREST).astype(bool)
    return aligned,valid


def estimate_projective(reference,moving):
    """ECC planar homography with bounded geometry and regional score checks."""
    import cv2
    cv2.setNumThreads(1)
    initial=estimate(reference,moving)
    fallback=np.array([[1,0,initial['dx']],[0,1,initial['dy']],[0,0,1]],dtype=float)
    a,b=edges(reference),edges(moving)
    m=24
    before=score(a[m:-m,m:-m],b[m:-m,m:-m])
    fallback_rgb,fallback_valid=warp_projective(moving,fallback)
    fallback_edge=edges(fallback_rgb)
    fallback_score=score(a[m:-m,m:-m],fallback_edge[m:-m,m:-m])
    record=dict(matrix=fallback.tolist(),accepted=initial['accepted'],transform='translation' if initial['accepted'] else 'identity',
                before=before,after=fallback_score,projective_accepted=False)
    def normalize(edge):
        return np.clip(edge/max(float(np.percentile(edge[m:-m,m:-m],99)),1e-8),0,1).astype('float32')
    try:
        inverse=np.linalg.inv(fallback).astype('float32')
        # Exclude acquisition/warping borders from ECC as well as score checks.
        # Otherwise strong empty-border edges can bias the fitted perspective.
        input_mask=np.zeros(a.shape,dtype='uint8');input_mask[m:-m,m:-m]=255
        cc,inverse=cv2.findTransformECC(normalize(a),normalize(b),inverse,cv2.MOTION_HOMOGRAPHY,
                                        (cv2.TERM_CRITERIA_COUNT|cv2.TERM_CRITERIA_EPS,60,1e-5),input_mask,5)
        matrix=np.linalg.inv(inverse.astype(float));matrix/=matrix[2,2]
        h,w=a.shape
        corners=np.array([[0,0],[w-1,0],[w-1,h-1],[0,h-1]],dtype='float64')
        homogeneous=np.column_stack([corners,np.ones(4)])@matrix.T
        if not np.isfinite(matrix).all() or np.min(homogeneous[:,2])<=0:
            raise ValueError('Invalid projection')
        moved=homogeneous[:,:2]/homogeneous[:,2:]
        displacement=float(np.linalg.norm(moved-corners,axis=1).max())
        area=cv2.contourArea(moved.astype('float32'),oriented=True)/((w-1)*(h-1))
        if displacement>32 or not 0.85<=area<=1.15:
            raise ValueError('Projection exceeds geometric bounds')
        aligned,valid=warp_projective(moving,matrix);c=edges(aligned)
        after=score(a[m:-m,m:-m],c[m:-m,m:-m])
        support=0
        for y in (0,h//2):
            for x in (0,w//2):
                sl=np.s_[y+m:y+h//2-m,x+m:x+w//2-m]
                support+=score(a[sl],c[sl])>=score(a[sl],fallback_edge[sl])-0.005
        accepted=after>=0.25 and after-fallback_score>=0.005 and support>=3 and valid.mean()>=0.85
        record.update(candidate_matrix=matrix.tolist(),candidate_score=after,ecc=float(cc),support=int(support),max_corner_displacement=displacement)
        if accepted:
            record.update(matrix=matrix.tolist(),accepted=True,transform='homography',projective_accepted=True,after=after)
    except (cv2.error,ValueError,np.linalg.LinAlgError) as error:
        record['rejection']=type(error).__name__+': '+str(error)[:120]
    return record

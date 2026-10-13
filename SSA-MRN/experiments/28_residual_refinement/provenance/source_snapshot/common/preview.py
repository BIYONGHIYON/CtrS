"""Save QuickBird input and output images for display-only comparison."""

from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


def _quickbird_rgb(chw):
    """Display QuickBird B,G,R,NIR data as R,G,B with per-image contrast."""
    if chw.ndim != 3 or chw.shape[0] != 4:
        raise ValueError("Expected a four-band QuickBird CHW image")
    rgb = np.moveaxis(chw[[2, 1, 0]], 0, -1)
    low, high = np.percentile(rgb, (1, 99), axis=(0, 1))
    scale = np.where(high > low, high - low, 1.0)
    return np.rint(np.clip((rgb - low) / scale, 0, 1) * 255).astype(np.uint8)


def save_prediction(output, output_dir, size):
    """Save the four-band prediction and an R,G,B display preview."""
    prediction = output.squeeze(0).detach().cpu().numpy().astype(np.float32, copy=False)
    if prediction.ndim != 3 or prediction.shape[0] != 4:
        raise ValueError("Expected a four-band QuickBird prediction")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    array_path = output_dir / f"random_output_{size}.npy"
    image_path = output_dir / f"random_output_preview_{size}.png"
    np.save(array_path, prediction)

    # QuickBird is treated as B,G,R,NIR; the display order is R,G,B.
    preview = _quickbird_rgb(prediction)
    Image.fromarray(preview).save(image_path)
    return array_path, image_path


def save_pan_preview(pan, output_dir, size):
    """Save the PAN input as a contrast-stretched grayscale PNG."""
    if pan.ndim != 4 or pan.shape[:2] != (1, 1):
        raise ValueError("Expected a single NCHW panchromatic image")
    array = pan[0, 0].detach().cpu().numpy().astype(np.float32, copy=False)
    low, high = np.percentile(array, (1, 99))
    scale = high - low if high > low else 1.0
    preview = np.rint(np.clip((array - low) / scale, 0, 1) * 255).astype(np.uint8)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    image_path = output_dir / f"pan_input_{size}.png"
    Image.fromarray(preview).save(image_path)
    return image_path


def save_ms_rgb_preview(ms, output_dir, size):
    """Save native low-resolution MS as an enlarged RGB display PNG."""
    if ms.ndim != 4 or ms.shape[:2] != (1, 4):
        raise ValueError("Expected a single four-band QuickBird MS image")
    if size < 4 or size % 4 or ms.shape[-2:] != (size // 4, size // 4):
        raise ValueError("MS must be one-quarter of the display size")
    array = ms[0].detach().cpu().numpy().astype(np.float32, copy=False)
    rgb = _quickbird_rgb(array)
    preview = Image.fromarray(rgb).resize((size, size), Image.Resampling.NEAREST)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    image_path = output_dir / f"ms_input_rgb_{size}.png"
    preview.save(image_path)
    return image_path


def panels(sample,prediction,path,label,index):
    bands=[2,1,0]
    reference=sample['gt'][bands].transpose(1,2,0)
    lo=np.percentile(reference,1,axis=(0,1));hi=np.percentile(reference,99,axis=(0,1))
    def rgb(a):
        a=a[bands].transpose(1,2,0)
        return Image.fromarray((np.clip((a-lo)/np.maximum(hi-lo,1e-8),0,1)*255).astype('uint8'))
    pan=sample['pan'][0];p0,p1=np.percentile(pan,[1,99])
    gray=Image.fromarray((np.clip((pan-p0)/max(p1-p0,1e-8),0,1)*255).astype('uint8')).convert('RGB')
    images=[('LR MS',rgb(sample['ms'])),('PAN',gray),(label,rgb(prediction)),('GT',rgb(sample['gt']))]
    h,w=pan.shape;canvas=Image.new('RGB',(4*w,h+38),'white');draw=ImageDraw.Draw(canvas)
    for j,(name,image) in enumerate(images):
        canvas.paste(image.resize((w,h),Image.Resampling.NEAREST),(j*w,38));draw.text((j*w+8,6),name,fill='black')
    draw.text((8,23),f'QB RR index {index} | full scene | shared GT 1-99% display stretch',fill='black')
    canvas.save(path)
    return {'rgb_assumed_bands_zero_based':bands,'display_lo':lo.tolist(),'display_hi':hi.tolist(),'pan_display_lo_hi':[float(p0),float(p1)],'no_inference_crop':True}

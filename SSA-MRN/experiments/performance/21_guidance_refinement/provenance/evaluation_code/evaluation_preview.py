import numpy as np
from PIL import Image, ImageDraw
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

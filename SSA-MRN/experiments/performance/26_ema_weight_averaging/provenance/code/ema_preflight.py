files={
    'train':'QuickBird__Training_Dataset__train_qb.h5',
    'val':'QuickBird__Training_Dataset__valid_qb.h5',
    'RR':'QuickBird__Testing_Dataset_ReducedData_H5_Format__test_qb_multiExm1.h5',
    'FR':'QuickBird__Testing_Dataset_FullData_H5_Format__test_qb_OrigScale_multiExm1.h5'}
data={}; scientific_data={}
dashboard.refresh('4개 H5의 유한값·범위·SHA-256 검사 중 · 수분이 걸릴 수 있습니다')
for split,name in files.items():
    dashboard.refresh(split+' 파일 검사 중 · 학습 시작 전 데이터 확인')
    path=Path(DATA_ROOT)/name
    if not path.is_file() or not h5py.is_hdf5(path):
        raise FileNotFoundError(f'{path}: attach the private Dataset via Add Input; ignore ._ metadata files')
    info={'file_name':name,'sha256':sha(path),'bytes':path.stat().st_size,'arrays':{}}
    required=('pan','lms','ms','gt') if split!='FR' else ('pan','lms','ms')
    with h5py.File(path,'r') as h:
        if set(required)-set(h.keys()):
            raise ValueError(f'{split}: missing keys {set(required)-set(h.keys())}; actual {list(h.keys())}')
        shapes={k:tuple(h[k].shape) for k in required}
        if any(len(s)!=4 for s in shapes.values()): raise ValueError(f'Expected NCHW: {shapes}')
        n,c,height,width=shapes['lms']
        if n<=0 or c!=4 or min(height,width)<4 or height%4 or width%4:
            raise ValueError(f'QB 4 bands / x4 required: {shapes}')
        if shapes['pan']!=(n,1,height,width) or shapes['ms']!=(n,4,height//4,width//4):
            raise ValueError(f'Incompatible PAN/MS shapes: {shapes}')
        if split!='FR' and shapes['gt']!=shapes['lms']: raise ValueError('GT shape mismatch')
        if split in ('RR','FR') and (height%32 or width%32): raise ValueError('Q2n requires 32-aligned full scenes')
        for key in required:
            ds=h[key]; low=float('inf'); high=-float('inf')
            if not np.issubdtype(ds.dtype,np.number): raise ValueError('Non-numeric H5 dtype')
            per_sample=math.prod(ds.shape[1:])*max(ds.dtype.itemsize,4)
            chunk=max(1,min(64,(64<<20)//per_sample))
            for start in range(0,n,chunk):
                array=np.asarray(ds[start:min(n,start+chunk)],dtype=np.float32)
                if not np.isfinite(array).all(): raise ValueError(f'{split}/{key}: nonfinite values')
                low=min(low,float(array.min())); high=max(high,float(array.max()))
            info['arrays'][key]={'shape':list(ds.shape),'dtype':str(ds.dtype),'raw_min':low,'raw_max':high,
                                  'normalized_min':low/2047.,'normalized_max':high/2047.,'all_finite':True}
            # Allow tiny FP32-scale numerical excursions; preserve every original value.
            # This is a validation tolerance in RAW DN units, not clipping/normalization.
            range_tolerance_dn = 1e-4
            info['arrays'][key]['range_tolerance_dn'] = range_tolerance_dn
            info['arrays'][key]['tiny_range_excursion'] = bool(low<0 or high>2047)
            # Supplied interpolated LMS can have real overshoot; its range is recorded.
            if key in ('pan','ms','gt') and (low < -range_tolerance_dn or high > 2047+range_tolerance_dn or high<=1):
                raise ValueError(f'{split}/{key}: unexpected QB raw DN range [{low},{high}]; verify normalization explicitly')
        used=min(n,64 if split=='train' else 32) if MODE=='smoke' and split in ('train','val') else n
        info['used_samples']=used
        info['subset_rule']='first 64 train / first 32 val H5 indices' if MODE=='smoke' and split in ('train','val') else 'all samples'
        info['resident_fp32_bytes']=sum(used*math.prod(shapes[k][1:])*4 for k in required) if split in ('train','val') else 0
        if split=='FR': info['gt_policy']='No GT required, synthesized or used, even if an extra key exists'
    scientific_data[split]=info
    data[split]={'path':str(path),**info}
    dashboard.data_checked(split,info)
if len({v['sha256'] for v in scientific_data.values()})!=4:
    raise ValueError('The four split files are not distinct by content')
rr_count=scientific_data['RR']['arrays']['pan']['shape'][0]
if rr_count<5: raise ValueError('At least 5 distinct RR scenes required')
selection={'seed':SEED,'indices':np.random.default_rng(SEED).choice(rr_count,5,replace=False).tolist(),
           'rule':'uniform without replacement before training/scoring','scene_id':'zero-based H5 sample index',
           'full_scene':True,'display_order':['LR MS nearest','PAN','Prediction','GT'],
           'contrast':'per-scene GT 1/99 percentiles; identical per-band contrast for LR MS, prediction, GT'}

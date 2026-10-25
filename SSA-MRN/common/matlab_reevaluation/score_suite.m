function score_suite(output_dir, dlpan_dir, cut_bounds, dim_cut, clip_output)
% Score the SAME retained-model predictions using actual DLPan MATLAB functions.
% No Python metric port is used for these scores. FR has no high-resolution GT.
arguments
    output_dir (1,:) char
    dlpan_dir (1,:) char
    cut_bounds (1,1) logical = false
    dim_cut (1,1) double = 21
    clip_output (1,1) logical = false
end
old_dir=pwd; cleanup=onCleanup(@() cd(old_dir));
output_dir=char(java.io.File(output_dir).getCanonicalPath());
dlpan_dir=char(java.io.File(dlpan_dir).getCanonicalPath());
addpath(genpath(dlpan_dir));
meta=jsondecode(fileread(fullfile(output_dir,'prediction_manifest.json')));
assert(meta.scenes==20,'Full original 20 scenes are required');
assert(dim_cut>=1 && dim_cut==floor(dim_cut),'dim_cut must be a positive MATLAB index');
required={'indexes_evaluation','indexes_evaluation_FS','q2n','SAM','ERGAS','SCC','QNR'};
resolved=struct();
for i=1:numel(required)
    found=which(required{i});
    assert(startsWith(found,dlpan_dir),'Unexpected MATLAB function shadowing: %s',found);
    resolved.(required{i})=found;
end
rows=struct('protocol',{},'scene',{},'metrics',{},'seconds',{});
start=tic;
for i=1:numel(meta.files)
    item=meta.files(i);
    x=load(fullfile(output_dir,item.file));
    fused=double(x.fused);
    assert(all(isfinite(fused(:))),'Nonfinite prediction');
    cd(dlpan_dir);
    if strcmp(item.protocol,'FR')
        [dl,ds,qnr]=indexes_evaluation_FS(fused,double(x.ms),double(x.pan),meta.bits,clip_output,double(x.lms),meta.case.sensor,4,1);
        scores=struct('D_lambda',dl,'D_s',ds,'QNR',qnr);
    else
        gt=double(x.reference);
        [q,sam,ergas,scc,q2]=indexes_evaluation(fused,gt,4,meta.bits,32,cut_bounds,dim_cut,clip_output);
        % DLPan wrapper does not define PSNR. Report BOTH common conventions,
        % explicitly at the target sensor's peak, rather than selecting the closest paper value.
        if cut_bounds
            gt=gt(dim_cut:end-dim_cut,dim_cut:end-dim_cut,:);
            fused=fused(dim_cut:end-dim_cut,dim_cut:end-dim_cut,:);
        end
        if clip_output
            fused(fused>2^meta.bits)=2^meta.bits;
            fused(fused<0)=0;
        end
        err=(fused-gt).^2;
        global_mse=mean(err(:));
        band_mse=reshape(mean(mean(err,1),2),[],1);
        psnr_global=10*log10(meta.peak^2/global_mse);
        psnr_band=mean(10*log10(meta.peak^2./band_mse));
        scores=struct('Q',q,'SAM',sam,'ERGAS',ergas,'SCC',scc,'Q2n',q2,...
            'PSNR_global',psnr_global,'PSNR_bandmean',psnr_band);
    end
    fields=fieldnames(scores);
    for j=1:numel(fields)
        assert(isreal(scores.(fields{j})) && isfinite(scores.(fields{j})),'Undefined MATLAB score %s scene %d',fields{j},item.scene);
    end
    rows(i)=struct('protocol',item.protocol,'scene',item.scene,'metrics',scores,'seconds',toc(start)); %#ok<AGROW>
    fprintf('%s MATLAB %s scene %d (%d/%d) elapsed %.1fs\n',meta.case.id,item.protocol,item.scene,i,numel(meta.files),toc(start));
end
report=struct('case',meta.case,'rows',rows,'matlab_version',version,'matlab_release',version('-release'),...
    'products',ver,'resolved_functions',resolved,'dlpan_commit',meta.dlpan_commit,...
    'profile',struct('cut_bounds',cut_bounds,'dim_cut',dim_cut,'clip_output',clip_output,...
        'clipping_upper_if_enabled',2^meta.bits,'peak',meta.peak,'bits',meta.bits,'block',32,'ratio',4,...
        'QNR_variant','DLPan Toolbox 1.0 flag_orig_paper=0 p=q=alpha=beta=1'),...
    'psnr_note','PSNR is computed in this MATLAB wrapper; DLPan indexes_evaluation returns Q, SAM, ERGAS, SCC, Q2n, not PSNR',...
    'seconds',toc(start));
fid=fopen(fullfile(output_dir,'matlab_scores.json'),'w','n','UTF-8');
assert(fid~=-1,'Cannot open result JSON');
file_cleanup=onCleanup(@() fclose(fid));
fwrite(fid,jsonencode(report,PrettyPrint=true),'char');
end

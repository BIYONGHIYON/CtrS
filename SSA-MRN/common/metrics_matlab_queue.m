function metrics_matlab_queue(work)
% Private local-file queue. Never upload images or original test data to CI.
addpath(fullfile(work,'reference'));
metrics_matlab_reference(work);
if ~isfolder(fullfile(work,'done')), mkdir(fullfile(work,'done')); end
fprintf('MATLAB %s; image toolbox %s\n',version,ver('images').Version);
started=tic; cached_name=''; cached=struct;
while toc(started)<5.5*3600
    jobs=dir(fullfile(work,'queue','*.mat'));
    for j=1:numel(jobs)
        source=fullfile(jobs(j).folder,jobs(j).name);
        target=fullfile(work,'done',[jobs(j).name '.json']);
        if isfile(target), continue; end
        f=load(source); result=struct; result.id=f.id;
        result.matlab_version=version; result.matlab_release=version('-release');
        if isfield(f,'data_file')
            if ~strcmp(cached_name,f.data_file), cached=load(fullfile(work,'inputs',f.data_file)); cached_name=f.data_file; end
        end
        result.rows=cell(1,numel(f.scenes));
        for k=1:numel(f.scenes)
            s=f.scenes{k};
            if isfield(f,'data_file')
                pred=double(s.normalized_pred)*double(f.scale); idx=double(s.index)+1; s=cached.scenes{idx}; s.index=idx-1;
            else
                pred=double(s.pred);
            end
            row=struct; row.index=s.index;
            if strcmp(f.protocol,'RR')
                gt=double(s.gt); delta=gt-pred;
                row.PSNR=mean(10*log10(double(f.peak)^2./squeeze(mean(delta.^2,[1 2]))));
                row.SAM=SAM(gt,pred); row.ERGAS=ERGAS(gt,pred,double(f.ratio));
                row.SCC=SCC(pred,gt); row.Q2n=q2n(gt,pred,32,32);
            else
                row.D_lambda=D_lambda(pred,double(s.lms),double(s.ms),32,double(f.ratio),1);
                row.D_s=D_s(pred,double(s.lms),double(s.ms),double(s.pan),double(f.ratio),32,1);
                row.QNR=(1-row.D_lambda)*(1-row.D_s);
            end
            result.rows{k}=row;
        end
        result.rows=[result.rows{:}]; result.protocol=f.protocol;
        fields=fieldnames(result.rows); result.mean=struct;
        for h=1:numel(fields)
            key=fields{h}; if strcmp(key,'index'), continue; end
            result.mean.(key)=mean([result.rows.(key)]);
        end
        temp=[target '.tmp']; fid=fopen(temp,'w'); fprintf(fid,'%s',jsonencode(result)); fclose(fid); movefile(temp,target);
        fprintf('Completed %s: %d scenes\n',f.id,numel(f.scenes));
    end
    if isfile(fullfile(work,'STOP')), fprintf('Queue complete\n'); return; end
    pause(2);
end
error('Queue deadline reached before STOP marker');
end

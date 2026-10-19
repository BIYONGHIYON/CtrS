function metrics_matlab_queue(work, maxSeconds)
% Bounded, restartable reference evaluation. Every completed JSON is atomic.
if nargin<2, maxSeconds=1500; end
addpath(fullfile(work,'reference'));
if ~isfolder(fullfile(work,'done')), mkdir(fullfile(work,'done')); end
if ~isfolder(fullfile(work,'errors')), mkdir(fullfile(work,'errors')); end
metrics_matlab_reference(work);
started=tic; cached_name=''; cached=struct;
while toc(started)<maxSeconds
    jobs=dir(fullfile(work,'queue','*.mat'));
    for j=1:numel(jobs)
        if toc(started)>=maxSeconds, break; end
        source=fullfile(jobs(j).folder,jobs(j).name);
        target=fullfile(work,'done',[jobs(j).name '.json']);
        try
            f=load(source);
            if isfile(target)
                old=jsondecode(fileread(target));
                if isfield(old,'contract_id') && strcmp(old.contract_id,f.contract_id), continue; end
            end
            result=struct; result.id=f.id;
            result.matlab_version=version; result.matlab_release=version('-release');
            result.contract_id=f.contract_id;
            if ~strcmp(cached_name,f.data_file)
                cached=load(fullfile(work,'inputs',f.data_file)); cached_name=f.data_file;
            end
            result.rows=cell(1,numel(f.scenes));
            for k=1:numel(f.scenes)
                prediction=f.scenes{k}; idx=double(prediction.index)+1;
                s=cached.scenes{idx}; pred=double(prediction.normalized_pred)*double(f.scale);
                row=struct; row.index=idx-1;
                if strcmp(f.protocol,'RR')
                    gt=double(s.gt); delta=gt-pred;
                    row.PSNR=mean(10*log10(double(f.peak)^2./squeeze(mean(delta.^2,[1 2]))));
                    row.MSE=mean(delta.^2,'all')/double(f.peak)^2;
                    row.SSIM=metrics_uniform_ssim(gt,pred,f.peak);
                    row.SAM=SAM(gt,pred); row.ERGAS=ERGAS(gt,pred,double(f.ratio));
                    row.SCC=SCC(pred,gt); row.Q2n=q2n(gt,pred,32,32);
                else
                    row.D_lambda=D_lambda(pred,double(s.lms),double(s.ms),32,double(f.ratio),1);
                    row.D_s=D_s(pred,double(s.lms),double(s.ms),double(s.pan),double(f.ratio),32,1);
                    row.QNR=(1-row.D_lambda)*(1-row.D_s);
                end
                values=cell2mat(struct2cell(row));
                if ~all(isfinite(values)), error('Nonfinite reference metric'); end
                result.rows{k}=row;
            end
            result.rows=[result.rows{:}]; result.protocol=f.protocol;
            fields=fieldnames(result.rows); result.mean=struct;
            for h=1:numel(fields)
                key=fields{h}; if strcmp(key,'index'), continue; end
                result.mean.(key)=mean([result.rows.(key)]);
            end
            atomicJSON(target,result);
            fprintf('Completed %s: %d scenes\n',f.id,numel(f.scenes));
        catch err
            atomicJSON(fullfile(work,'errors',[jobs(j).name '.json']),struct('source',jobs(j).name,'message',err.message,'identifier',err.identifier));
            fprintf('Deferred %s: %s\n',jobs(j).name,err.identifier);
        end
    end
    atomicJSON(fullfile(work,'matlab_status.json'),struct('state','running','elapsed_seconds',toc(started),'version',version));
    if isfile(fullfile(work,'STOP')), break; end
    pause(3);
end
atomicJSON(fullfile(work,'matlab_status.json'),struct('state','batch_finished','elapsed_seconds',toc(started),'version',version));
fprintf('Bounded batch finished; remaining files are resumable.\n');
end
function atomicJSON(path, value)
temp=[path '.tmp']; fid=fopen(temp,'w'); assert(fid>=0); fprintf(fid,'%s',jsonencode(value)); fclose(fid); movefile(temp,path);
end

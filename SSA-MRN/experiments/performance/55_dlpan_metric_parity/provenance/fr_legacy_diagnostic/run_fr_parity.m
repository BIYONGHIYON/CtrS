
root = fileparts(mfilename('fullpath'));
oldpath = path;
cleanup = onCleanup(@() path(oldpath));
addpath(fullfile(root,'Quality_Indices'),'-begin');
addpath(fullfile(root,'Tools'),'-begin');
d = load(fullfile(root,'inputs.mat'));
n = numel(d.pans);
official_scores = zeros(n,3);
shared_scores = zeros(n,3);
down_diff = zeros(n,1);
filter_shared_diff = zeros(n,1);
for k=1:n
    lms = d.lms_rows{k}; ms = d.ms_rows{k}; pan = d.pans{k};
    [qn,dl,ds] = QNR(lms,lms,ms,pan,4,32,1,1,1,1);
    official_scores(k,:) = [dl,ds,qn];
    md = imresize(pan,1/4);
    down_diff(k) = max(abs(md(:)-d.python_down{k}(:)));
    shared_filter = interp23tap(d.python_down{k},4);
    filter_shared_diff(k) = max(abs(shared_filter(:)-d.python_filtered{k}(:)));
    shared_ds = 0;
    for b=1:size(lms,3)
        high = local_uqi(lms(:,:,b),pan,32);
        low = local_uqi(lms(:,:,b),shared_filter,32);
        shared_ds = shared_ds + abs(high-low)/size(lms,3);
    end
    shared_scores(k,:) = [dl,shared_ds,(1-dl)*(1-shared_ds)];
    fprintf('%d/%d scenes evaluated\n',k,n);
end
labels = ["D_lambda","D_s","QNR"];
Scene = []; Metric = strings(0,1);
Python = []; MATLAB = []; AbsDiff = []; SharedPANDiff = [];
for k=1:n
    for j=1:3
        Scene(end+1,1)=k-1;
        Metric(end+1,1)=labels(j);
        Python(end+1,1)=d.python_scores(k,j);
        MATLAB(end+1,1)=official_scores(k,j);
        AbsDiff(end+1,1)=abs(d.python_scores(k,j)-official_scores(k,j));
        SharedPANDiff(end+1,1)=abs(d.python_scores(k,j)-shared_scores(k,j));
    end
end
T=table(Scene,Metric,Python,MATLAB,AbsDiff,SharedPANDiff);
writetable(T,fullfile(root,'fr_comparison.csv'));
P=table((0:n-1)',down_diff,filter_shared_diff, ...
    'VariableNames',{'Scene','PANDownMaxAbsDN','SharedDownFilterMaxAbsDN'});
writetable(P,fullfile(root,'pan_comparison.csv'));
save(fullfile(root,'fr_results.mat'),'official_scores','shared_scores','T','P');
fid=fopen(fullfile(root,'matlab_version.txt'),'w');
fprintf(fid,'%s\n',version); fclose(fid);
format long g
for j=1:3
    m=Metric==labels(j);
    fprintf('%s: full max diff %.17g; shared-PAN max diff %.17g\n', ...
        labels(j),max(AbsDiff(m)),max(SharedPANDiff(m)));
end
fprintf('PAN down max difference (DN): %.17g\n',max(down_diff));
fprintf('Same down PAN -> interp23tap max difference (DN): %.17g\n',max(filter_shared_diff));
zip(fullfile(root,'fr_parity_results.zip'), ...
    {'fr_comparison.csv','pan_comparison.csv','fr_results.mat', ...
     'metadata.json','matlab_version.txt','metrics_snapshot.py','run_fr_parity.m'},root);

function value=local_uqi(a,b,S)
    values=zeros(size(a,1)/S,size(a,2)/S);
    for r=1:size(values,1)
        for c=1:size(values,2)
            rows=(r-1)*S+(1:S); cols=(c-1)*S+(1:S);
            x=double(a(rows,cols)); x=x(:);
            y=double(b(rows,cols)); y=y(:);
            C=cov(x,y);
            values(r,c)=4*C(1,2)*mean(x)*mean(y) ...
                /(C(1,1)+C(2,2))/(mean(x)^2+mean(y)^2);
        end
    end
    value=mean(values(:));
end

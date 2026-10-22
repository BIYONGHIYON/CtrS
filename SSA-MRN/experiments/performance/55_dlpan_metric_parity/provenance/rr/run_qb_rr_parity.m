
root = fileparts(mfilename('fullpath'));
oldpath = path;
restorepath = onCleanup(@() path(oldpath));
addpath(fullfile(root,'Quality_Indices'),'-begin');
d = load(fullfile(root,'inputs.mat'));
n = numel(d.refs);
values = zeros(n,5);
for k = 1:n
    gt = double(d.refs{k});
    pred = double(d.preds{k});
    mse = mean(mean((pred-gt).^2,1),2);
    values(k,:) = [SAM(gt,pred), ERGAS(gt,pred,4), ...
        q2n(gt,pred,32,32), SCC(pred,gt), ...
        mean(10*log10(2047^2 ./ max(mse(:),1e-30)))];
    fprintf('%d/%d scenes evaluated\n',k,n);
end
labels = ["SAM","ERGAS","Q2n","SCC","PSNR_project"];
Case = strings(n*5,1); Metric = strings(n*5,1);
Python = zeros(n*5,1); MATLAB = zeros(n*5,1);
AbsDiff = zeros(n*5,1); Status = strings(n*5,1);
row = 0;
for k = 1:n
    for j = 1:5
        row = row+1;
        Case(row) = string(d.case_names{k});
        Metric(row) = labels(j);
        a = d.python_scores(k,j); b = values(k,j);
        Python(row) = a; MATLAB(row) = b;
        AbsDiff(row) = abs(a-b);
        if ~isfinite(a) || ~isfinite(b)
            Status(row) = "REVIEW_NONFINITE";
        elseif abs(a-b) <= 1e-10 + 1e-10*abs(b)
            Status(row) = "PASS";
        else
            Status(row) = "DIFF";
        end
    end
end
T = table(Case,Metric,Python,MATLAB,AbsDiff,Status);
writetable(T,fullfile(root,'rr_comparison.csv'));
save(fullfile(root,'rr_results.mat'),'values','T');
fid = fopen(fullfile(root,'matlab_version.txt'),'w');
fprintf(fid,'%s\n',version); fclose(fid);
format long g
for j = 1:5
    mask = Metric == labels(j);
    fprintf('%s: max difference %.17g\n',labels(j),max(AbsDiff(mask)));
end
fprintf('PASS=%d / DIFF=%d / REVIEW_NONFINITE=%d\n', ...
    sum(Status=="PASS"),sum(Status=="DIFF"), ...
    sum(Status=="REVIEW_NONFINITE"));
disp(T(Status~="PASS",:));
zip(fullfile(root,'rr_parity_results.zip'), ...
    {'rr_comparison.csv','rr_results.mat','metadata.json', ...
     'matlab_version.txt','metrics_snapshot.py','run_qb_rr_parity.m'},root);

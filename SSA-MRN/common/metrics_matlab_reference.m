function metrics_matlab_reference(work)
% Run unmodified pinned DLPan MATLAB functions on deterministic synthetic data.
addpath(fullfile(work,'reference'));
f = load(fullfile(work,'fixtures.mat'));
rows = cell(1,numel(f.names));
for k = 1:numel(f.names)
    id = f.names{k};
    gt = f.([id '_gt']); pred = f.([id '_pred']);
    lms = f.([id '_lms']); ms = f.([id '_ms']); pan = f.([id '_pan']);
    row.name = id;
    row.SAM = SAM(gt,pred);
    row.ERGAS = ERGAS(gt,pred,4);
    row.SCC = SCC(pred,gt);
    row.Q2n = q2n(gt,pred,32,32);
    row.D_lambda = D_lambda(pred,lms,ms,32,4,1);
    row.D_s = D_s(pred,lms,ms,pan,4,32,1);
    row.QNR = (1-row.D_lambda)*(1-row.D_s);
    rows{k} = row;
    fprintf('%s Q2n=%.12g QNR=%.12g\n',id,row.Q2n,row.QNR);
end
result.rows = [rows{:}];
for n = [64,128,256,512]
    weights = imresize(eye(n),[n/4,n],'bicubic','Antialiasing',true);
    result.(sprintf('resize_%d',n)) = weights;
    x = reshape(sin((1:n*n)/17),n,n);
    actual = imresize(x,0.25,'bicubic','Antialiasing',true);
    result.(sprintf('resize_operator_error_%d',n)) = max(abs(actual-weights*x*weights'),[],'all');
end
for c = [4,8]
    a = f.(sprintf('onion_a%d',c)); b = f.(sprintf('onion_b%d',c));
    v = zeros(size(a));
    for j = 1:size(a,1)
        v(j,:) = onion_mult(a(j,:),b(j,:));
    end
    result.(sprintf('onion%d',c)) = v;
end
result.uint16_ties = uint16(f.uint16_ties);
result.matlab_version = version;
result.matlab_release = version('-release');
result.image_toolbox_version = ver('images');
save(fullfile(work,'reference.mat'),'-struct','result','-v7');
fid = fopen(fullfile(work,'matlab_environment.json'),'w');
fprintf(fid,'%s',jsonencode(struct('version',version,'release',version('-release'),'image_toolbox',ver('images'))));
fclose(fid);
end

function value=metrics_uniform_ssim(reference,prediction,peak)
% Match the historical variable-scale protocol: uniform7, sample covariance,
% K1=.01, K2=.03, valid interior, mean over bands. No clipping.
a=double(reference)/double(peak); b=double(prediction)/double(peak);
w=min([7,size(a,1),size(a,2)]); if mod(w,2)==0, w=w-1; end
kernel=ones(w,w)/(w*w); scores=zeros(1,size(a,3));
for band=1:size(a,3)
    x=a(:,:,band); y=b(:,:,band);
    ux=conv2(x,kernel,'valid'); uy=conv2(y,kernel,'valid');
    factor=(w*w)/(w*w-1);
    vx=factor*(conv2(x.*x,kernel,'valid')-ux.*ux);
    vy=factor*(conv2(y.*y,kernel,'valid')-uy.*uy);
    covariance=factor*(conv2(x.*y,kernel,'valid')-ux.*uy);
    map=((2*ux.*uy+.01^2).*(2*covariance+.03^2))./((ux.*ux+uy.*uy+.01^2).*(vx+vy+.03^2));
    scores(band)=mean(map,'all');
end
value=mean(scores);
end

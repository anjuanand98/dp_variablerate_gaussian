function lambda
clc;
close all;
clear all;
%initializing x 
num=100;%number of elements in x range
%range of x: [a,b]
% a=0;
% b=1;
a=-3;
b=3;
sigma_xsq=1;
mu=0;
%samples of x
delta=(b-a)/num;
xval=[a+delta/2:delta:b-delta/2];
%pmf of samples of x
fun=@(xv) ((1/sqrt(2*pi*sigma_xsq))*exp(-(xv-mu).^2/(2*sigma_xsq)));
% fun=@(xv) 1/(b-a);
scal=1/integral(fun,a,b,'ArrayValued',true);
px=zeros(length(xval),1);
for i=1:length(px)
    px(i)=integral(fun,xval(i)-delta/2,xval(i)+delta/2,'ArrayValued',true)*scal;
end
xval=[xval b];
D1val=zeros((length(xval)-1)*length(xval)/2,2);%stores all possible intervals [alpha,beta]
H1val=zeros((length(xval)-1)*length(xval)/2,1);
k=1;%iteration over all possible (alpha,beta) pairs
for i=1:length(xval)-1%for each alpha value possible
    for j=i+1:length(xval)%for each beta value possible, given the alpha value
        D1val(k,1:2)=[xval(i) xval(j)];%(alpha,beta)
        H1val(k)=entropy([xval(i);xval(j)],xval,px);
        k=k+1;
    end
end
Dnxoxval=D1val(1:length(xval)-1,2);%(xo,x) - possible x values
lambdaval=[0.001 0.01 0.02 0.05 0.3 0.4 0.5 0.6 0.7 1 3];
lambdaval=5;
lambdaval=sort(lambdaval);
tl=length(lambdaval);
Hvar=zeros(1,tl);%optimal H corresponding to optimal lambda
Nvar=zeros(1,tl);%the N value for which the optimal H is attained
dvar=zeros(1,tl);%the encoder distortion value for which the optimal H is attained
ddvar=zeros(1,tl);%the decoder distortion value for which the optimal H is attained
thresh=0.005;
n1=10;
for lambda=lambdaval%for each lambda
    %initializing D1 for each [alpha, beta] pair in D1val
    D1temp=zeros(size(D1val,1),1);%D1([alpha,beta])=minimum over y integral(alpha to beta (f(x,x-y)p(x)dx))
    qtemp=zeros(size(D1val,1),1);
    %calculating D1
    for i=1:size(D1val,1)%iterate over possible (alpha,beta) pairs
        qtemp(i)=decoder([D1val(i,1);D1val(i,2)],xval,px,lambda);
        D1temp(i)=encoderdistortion([D1val(i,1);D1val(i,2)],qtemp(i),xval,px);
    end
    D1=D1temp+lambda*H1val;
    dval=zeros(1,n1);%distortion for each N - to calculate stopping criteria - difference between successive N values is negligible
    diffdv=zeros(1,n1);%fractional difference in distortion values
    ite=1;
    diffdv(ite)=1;
    N=1;
    while(diffdv(ite)>thresh)
        [H Xnopt Yn dtemp dtempdec]=main(lambda,N,xval,px,Dnxoxval,D1val,D1,H1val);
        Hvar(find(lambda==lambdaval))=H;
        Nvar(find(lambda==lambdaval))=N;
        dvar(find(lambda==lambdaval))=dtemp;
        ddvar(find(lambda==lambdaval))=dtempdec;
        dval(N)=dtemp;%distortion for each N
        %condition to stop looping over N - decrease in distortion becomes negligible
        if N>1
            diffd=abs(dval(N-1)-dval(N))/dval(N);
            ite=ite+1;
            diffdv(ite)=diffd;
        end
        N=N+1;
    end
end
de=dvar-lambdaval.*Hvar;
dd=ddvar-lambdaval.*Hvar;
% % f=figure;
% % plot(Hvar,de,'*-');
% % hold on;
% % plot(Hvar,dd,'o-');
% % hold off;
% % legend({'encoder distortion','decoder distortion'},'FontSize',14);
% % xlabel('H','FontSize',14);
% % ylabel('distortion','FontSize',14);
% % title('variable rate xcubed uniform','FontSize',14);
% % saveas(f,'variablerate_xcubed_uniform.fig');
% % saveas(f,'variablerate_xcubed_uniform.png');
save('varrate_gaussian.mat','Hvar','de','dd','lambdaval','Nvar');

function [H Xnopt Yn dtemp dtempdec]=main(lambda,N,xval,px,Dnxoxval,D1val,D1,H1val)
if N==1
    Xnopt=[xval(1);xval(end)];
    Yn=decoder(Xnopt,xval,px,lambda);
    %computing entropy for a given lambda
    H=entropy(Xnopt,xval,px);%entropy for given lambda
    %encoder distortion
    dtemp=encoderdistortion(Xnopt,Yn,xval,px)+lambda*H;
    dtempdec=decoderdistortion(Xnopt,Yn,xval,px)+lambda*H;
    return;
end
[Xnopt]=nonstrategic_quantization_hloop(xval,N,px,Dnxoxval,D1val,D1);
Yn=decoder(Xnopt,xval,px,lambda);
%computing entropy for a given lambda
H=entropy(Xnopt,xval,px);%entropy for given lambda
%encoder distortion
dtemp=encoderdistortion(Xnopt,Yn,xval,px)+lambda*H;
dtempdec=decoderdistortion(Xnopt,Yn,xval,px)+lambda*H;

function [H]=entropy(x,xval,px)
N=length(x)-1;
p=zeros(1,N);
for i=1:N%iterate over possible (alpha,beta) pairs
    in1=find(x(i)==xval);
    in2=find(x(i+1)==xval)-1;
    p(i)=ones(1,length(px(in1:in2)))*px(in1:in2);
end
H=-(p)*log(p')/log(2);

function [y]=decoder(x,xval,px,lambda)
N=length(x)-1;
y=zeros(1,N);
for j=1:N
in1=find(x(j)==xval);
in2=find(x(j+1)==xval);
v=zeros(1,length(xval));
Hl=lambda*entropy([xval(in1),xval(in2)],xval,px);
for i=1:length(xval)%iterate over each region
    v(i)=decoderdistortion([xval(in1) xval(in2)],xval(i),xval,px)+Hl;
end
[v1,v2]=min(v);
y(j)=xval(v2);
end

function endistortion=encoderdistortion(x,y,xval,px)
N=length(y);
endistortion=0;
for n=1:N%iterate over each region
    in1=find(x(n)==xval);
    in2=find(x(n+1)==xval)-1;
    endistortion=endistortion+((xval(in1:in2)-y(n)).^2)*px(in1:in2);  
end
    
function dedistortion=decoderdistortion(x,y,xval,px)
N=length(y);
dedistortion=0;
for n=1:N%iterate over each region
    in1=find(x(n)==xval);
    in2=find(x(n+1)==xval)-1;
    dedistortion=dedistortion+((xval(in1:in2)-y(n)).^2)*px(in1:in2);  
end 

function [Xnopt]=nonstrategic_quantization_hloop(xval,N,px,Dnxoxval,D1val,D1)
%initializing x optimal end points
Xnopt=10^7*ones(N+1,1);
Xnopt(1)=xval(1);
Xnopt(end)=xval(end);
%initializing [alpha, beta] pairs for x
Dn=10^7*ones(length(Dnxoxval),N-1);%stores Dn values for each n (2:N), for each x in (xo,x)
Xn=10^7*ones(length(Dnxoxval),N-1);%stores X_(n-1) values for each n (1:N-1), for each x in (xo,x)
for n=2:N
    for i=n+1:length(xval)%Dn(xo,x(i)) - there should be sufficient levels between xo and x(i), so i starts from n, e.g., n=2 level quantization implies you need atleast three points in x 
        [minval xboundary]=nonstr_recursion_1(xval,i,n,Xn,Dn,Dnxoxval,D1val,D1);%finding Dn and X_(n-1) values for each (xo,x) for a given n level 
        %indexed such that x(i) location in Dnxoxval gives corresponding Dn and Xn values
        Dn(find((Dnxoxval==xval(i))),n-1)=minval;
        Xn(find((Dnxoxval==xval(i))),n-1)=xboundary;
    end
end
for n=N:-1:2%backward iteration to find Xnopt
    Xnopt(n)=Xn(find((Dnxoxval==Xnopt(n+1))),n-1);%X_(n-1) optimal=X_(n-1)(xo,X_n opt)
end

function [minval xboundary]=nonstr_recursion_1(xval,xind,n,Xn,Dn,Dnxoxval,D1val,D1)%returns distortion and corresponding value of x and theta
    if n==1%n=1 values are already computed in D1
        minval=D1(find(D1val(:,1)==xval(1) & D1val(:,2)==xval(xind)));
        xboundary=xval(xind);
        return;
    end
    alpharange=[n:xind-1];%indices of alpha values possible in x range (xo<alpha<x)
    arr=zeros(length(alpharange),1);%for each alpha value possible, finding minimum (D_(n)(xo,x)=min over alpha, ao<alpha<x D_(n-1)(xo,alpha)+D1(alpha,x))
    %D_n(xo<alpha<x, thetao<beta<theta)
    for arrx=1:length(alpharange)
        if n-2>=1 && Xn(find((Dnxoxval==xval(alpharange(arrx)))),n-2)~=10^7
            xboundary1=Xn(find((Dnxoxval==xval(alpharange(arrx)))),n-2);
            minval1=Dn(find((Dnxoxval==xval(alpharange(arrx)))),n-2);
        else
        [minval1 xboundary1]=nonstr_recursion_1(xval,alpharange(arrx),n-1,Xn,Dn,Dnxoxval,D1val,D1);%D_(n-1)(xo,alpha) 
        end
        %D_n(xo,x)=min over alpha, xo<alpha<x (D_(n-1)(xo,alpha)+D1(alpha,x))
        arr(arrx)=minval1+D1(find(D1val(:,1)==xval(alpharange(arrx)) & D1val(:,2)==xval(xind)));
    end
    minval=min(arr);%returning minimum distortion value
    in=find(arr==minval);
    xboundary=xval(alpharange(in(1)));
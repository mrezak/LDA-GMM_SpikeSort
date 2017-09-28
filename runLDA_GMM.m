% Implementation of LDA-GMM algorithm with initialization and outlier
% Detection

% [H,U,Y,V] = runLDA_GMM(X, k, d, minSampleSize, SampIter, maxiter)
% Inputs:
% X, Input spike matrix, dim:(n_spikes, n_sample_per_spike)
% k, Number of clusters
% d, dimension of the feature subspace
% minSampleSize, sample size for initialization
% SampleIter, number of iterations for initialization
% maxiter, maximun number of LDA-GMM iterations for subspace learning
% Output:
% H, sorted spike labels
% U, projection matrix to learned feature subspace
% Y, spike features i.e. projected samples into the subspace
% V, trace ratio value for the final iteration 

% Reference:
% [1] Mohammad Reza Keshtkaran and Zhi Yang, "Noise-robust unsupervised spike 
% sorting based on discriminative subspace learning with outlier handling",
% Journal of Neural Engineering 14 (3), 2017
% [2] Mohammad Reza Keshtkaran and Zhi Yang "Unsupervised spike sorting based on 
% discriminative subspace learning", EMBC 2014

% Author: Mohammad Reza Keshtkaran (keshtkaran@u.nus.edu)

%   Licence:
%   Downloaded from: https://github.com/mrezak/spikeSort
%   Copyright (C) 2017, Mohammad Reza Keshtkaran <keshtkaran@u.nus.edu>  or <keshtkaran.github@gmail.com>
% 
%   This program is free software: you can redistribute it and/or modify
%   it under the terms of the GNU General Public License as published by
%   the Free Software Foundation, either version 3 of the License, or
%   (at your option) any later version.
% 
%   This program is distributed in the hope that it will be useful,
%   but WITHOUT ANY WARRANTY; without even the implied warranty of
%   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
%   GNU General Public License for more details.
% 
%   You should have received a copy of the GNU General Public License
%   along with this program.  If not, see <http://www.gnu.org/licenses/>.


function [H,U,Y,V] = runLDA_GMM(X, k, d, minSampleSize, SampIter, maxiter)
    X = X'; %(n_spikes * n_sample_per_spike)
    if ~exist('minSampleSize','var')
        minSampleSize = 1000;
    end
    if ~exist('maxiter','var')
        maxiter = 50;
    end
    if ~exist('SampIter','var')
        SampIter = 6;
    end
    
    dd = size(X,1);
    if ~exist('d','var')
        d = min(k-1, dd);
    elseif isempty(d)
        d = min(k-1, dd);
    elseif d >= k
        d = k-1;
    end

    X = bsxfun(@minus,X,mean(X,2));
    X = diff(X);
    [Wpca,Xpca,eigV] = pca(X'); 
    idx = eigV>max(eigV)*1e-10;
    %idx = 1:floor(length(eigV)*0.9);
    Xpca = Xpca(:,idx)'; 
    Wpca = Wpca(:,idx);
    [dd,N] = size(Xpca);
    
    Vsamp = zeros(1,SampIter);
    Usamp = cell(1,SampIter);

    %Xsamp = Xpca;
    %[~, U0] = pca(Xpca'); U0 = U0';    
    for i = 1:SampIter
        Sampidx = randperm(N,minSampleSize);
        Xsamp = Xpca(:,Sampidx);
        if mod(i,2)
            U0 = pca(Xsamp'); U0 = U0';
        else
            %U0 = pca(Xsamp'); U0 = U0';
            U0 = randn(dd,d);
        end
        [Hsamp{i}, Uout,Vout] = doDisClust(Xsamp,k,d,15,U0,0);  % Outlier Detection disabled
        Usamp{i} = Uout;
        Vsamp(i) = abs(Vout);
        %Vsamp(i) = ClusQualMeasure((Uout'*Xsamp)',Hsamp{i});
    end
    %disp(Vsamp);
    [~,VmaxIdx] = max(Vsamp);
    U0 = Usamp{VmaxIdx};
    
    %Y = U0' * Xsamp;
    %gscatter3(Y,Hsamp{VmaxIdx});    
    
    [H,U,V] = doDisClust(Xpca,k,d,maxiter,U0,1);  % Outlier Detection enabled
        
    %fprintf(1,'Exited at iteration: %d\n',iter);
    H = H(:)';
    %idx = H > 0;
    %[U,V] = LDAd(Xs(:,idx),H(idx)',k); U = U(:,1:d);
    U = Wpca * U;
    Y = U' * X;
    % gscatter3(Y,H);  
end

function [H,U,V,C] = doDisClust(X,k,d,maxiter,U0,OLdetect)
    warning('off','stats:gmdistribution:FailedToConverge')
    X = bsxfun(@minus,X,mean(X,2));
    N = size(X,2);
    U = U0(:,1:d);
    H = ones(N,1);    
    oldV = inf;
    V = 0;
    iter = 0;
    if OLdetect
        numComp = k+1;
        meanMat = zeros(d,1);
        gmmS.Sigma = 10*eye(d,d);
        for j=1:k
            gmmS.Sigma(:,:,j+1) = 1*eye(d,d);
        end
        gmmS.PComponents = [0.01 0.99*ones(1,k)/k];
    else
        numComp = k;
        meanMat = [];
        for j=1:k
            gmmS.Sigma(:,:,j) = 1*eye(d,d);
        end
        gmmS.PComponents = ones(1,k)/k;
    end
    
    while ( abs(V-oldV) > 1e-8 && iter < maxiter)   %check for convergance
        iter = iter + 1;
        Y = U' * X;         %Projectin to discriminative subspace
        Y = bsxfun(@minus,Y,mean(Y,2));
        [~,Y] = pca(Y'); Y = Y';
        S = std(Y,[],2)+1e-8;
        Y = bsxfun(@rdivide, Y, S);
        %Y = bsxfun(@rdivide, Y,max(abs(Y),[],2));
        
        [H,M] = kmeans(Y',k,'start','plus','Distance','sqeuclidean','EmptyAction','singleton','Replicates',3,'maxiter',200);
        Hp = H;
        Xp = X;
        if OLdetect
            gmmS.mu = [meanMat M']';
            gm = gmdistribution.fit(Y',numComp, 'start', gmmS, 'CovType', 'full', 'Regularize',1e-4,'Replicates',1);
            %gm = gmdistribution.fit(Y',k+1, 'start', 'plus', 'CovType', 'full', 'Regularize',1e-4,'Replicates',1);
            H = cluster(gm,Y')';
            for j=1:k+1
                SigNorm(j) = norm(gm.Sigma(:,:,j));
            end
            Hp = H;
            [~,idx] = max(SigNorm);
            idx = H==idx; %outliers
            H(idx) = -1;
            if length(unique(H(H>0))) < k
                iter = iter - 1;
            else
                Hp(idx) = [];
                Xp(:,idx) = [];
            end
        end

        oldV = V;
        [U,V] = LDAd(Xp,Hp,k); U = U(:,1:d);
    end
    
    unqH = unique(H(H>0));
    for i = 1:length(unqH)
        H(H==unqH(i)) = i;
    end
    warning('on','stats:gmdistribution:FailedToConverge')
end


    
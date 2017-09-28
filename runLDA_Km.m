% Implementation of LDA-KMM algorithm with initialization

% [H,U,Y,V] = runLDA_Km(X,k,d,minSampleSize,SampIter,maxiter)
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

function [H,U,Y,V] = runLDA_Km(X, k, d, minSampleSize, SampIter, maxiter)
    X = X'; %(n_spikes * n_sample_per_spike)
    if ~exist('minSampleSize','var')
        minSampleSize = 1000;
    end
    if ~exist('maxiter','var')
        maxiter = 200;
    end
    if ~exist('SampIter','var')
        SampIter = 6;
    end
    
    dd = size(X,1);
    if ~exist('d','var')
        d = min(k-1, dd);
    elseif isempty(d)
        d = min(k-1, dd);
    elseif d>=k
        d = k - 1;
    end

    X = bsxfun(@minus,X,mean(X,2));
    [Wpca,Xpca,eigV] = pca(X'); 
    idx = eigV > max(eigV)*1e-10;
    Xpca = Xpca(:,idx)'; 
    Wpca = Wpca(:,idx);
    [dd,N] = size(Xpca);
    
    Vsamp = zeros(1,SampIter);
    Usamp = cell(1,SampIter);

    %Xsamp = Xpca;
    %[~, U0] = pca(Xpca'); U0 = U0';  
    for i = 1:SampIter
        Sampidx = randperm(N, minSampleSize);
        Xsamp = Xpca(:, Sampidx);
        %U0 = pca(Xsamp'); U0 = U0'; % PCA initialization on block
        U0 = randn(dd,d);   % Random initialization on block
        [Hsamp{i}, Uout, Vout] = doDisClust(Xsamp,k,d,15,U0);  % Outlier Detection disabled
        Usamp{i} = Uout;
        Vsamp(i) = abs(Vout);
    end
    %disp(Vsamp);
    [~,VmaxIdx] = max(Vsamp);
    U0 = Usamp{VmaxIdx};
    
    %Y = U0' * Xsamp;
    %gscatter3(Y, Hsamp{VmaxIdx});    
    
    [H,U,V] = doDisClust(Xpca, k, d, maxiter, U0);  % Outlier Detection disabled
        
    %fprintf(1,'Exited at iteration: %d\n',iter);
    H = H(:)';
    U = Wpca * U;
    Y = U' * X;
    % gscatter3(Y,H);    

end

function [H,U,V,C] = doDisClust(X,k,d,maxiter,U0)
    X = bsxfun(@minus,X,mean(X,2));
    N = size(X,2);
    U = U0(:,1:d);
    H = ones(N,1);
    oldH = zeros(N,1);
    iter = 0;
    while (any(oldH ~= H) && iter < maxiter)   %check for convergance
        iter = iter + 1;
        Y = U' * X;         %Projectin to discriminative subspace
        oldH = H;
        
        % Whitening:
        Y = bsxfun(@minus,Y,mean(Y,2));
        [~,Y] = pca(Y'); Y = Y';
        S = std(Y,[],2)+1e-10;
        Y = bsxfun(@rdivide, Y, S);
        
        [H,C] = kmeans(Y',k,'start','plus','EmptyAction','singleton','Replicates',3,'Distance','sqEuclidean','maxiter',200);    %k-means

        [U,V] = LDAd(X,H,k); U = U(:,1:d);
    end
    
    unqH = unique(H(H>0));
    for i = 1:length(unqH)
        H(H==unqH(i)) = i;
    end
end


    
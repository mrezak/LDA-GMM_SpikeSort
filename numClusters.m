%  Function for estimating the number of clusters

% Reference:
% [1] Mohammad Reza Keshtkaran and Zhi Yang, "Noise-robust unsupervised spike 
% sorting based on discriminative subspace learning with outlier handling",
% Journal of Neural Engineering 14 (3), 2017
% [2] Mohammad Reza Keshtkaran and Zhi Yang, "Unsupervised spike sorting based on 
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

function K = numClusters(Y, lbl, minClusSize)
    if ~exist('minClusSize', 'var')
        minClusSize = 10;
    end
    warning('off', 'stats:adtest:OutOfRangePLow');
    warning('off', 'stats:kmeans:FailedToConverge');
    H = lbl(:)';
    Y = Y(:,H>0);
    H = H(H>0);
    Y_org = Y;
    H_org = H;
    %M = zeros(size(Y,1),k);
    for j=unique(H)
        M(:,j) = mean(Y(:,H==j),2);
    end
    M_org = M;
    Y = bsxfun(@minus,Y,mean(Y,2));
    [~,Y] = pca(Y'); Y = Y';
    S = std(Y,[],2)+1e-10;
    Y = bsxfun(@rdivide, Y, S);
    H = kmeans(Y', [], 'Start', M', 'EmptyAction','singleton', 'MaxIter', 10)';
    for j=unique(H)
        M(:,j) = mean(Y(:,H==j),2);
    end
    
    Hunq = unique(H);
    hClus = histc(H, Hunq);
    smallClus = hClus < minClusSize; 
    if any(smallClus)
        %disp('Cluster size is less than minimum allowed size. Removing cluster\n')
        idx = logical(sum(bsxfun(@eq, H, Hunq(smallClus)')));
        M(:,smallClus) = [];
        H(:,idx) = [];
        Y(:,idx) = [];
    end
    %gscatter3(Y, H, []);
    % Projecting onto Mean vectors
    % for merging clusters
    
    K = size(M,2);  % initial number of clusters
    for i=1:size(M,2)
        for j = i+1:size(M,2)
            Pj = (M(:,i)-M(:,j))'*Y(:,H==i | H==j);
            P1 = (M(:,i)-M(:,j))'*Y(:,H==i);
            P2 = (M(:,i)-M(:,j))'*Y(:,H==j);
            minClusN = min([length(Pj), length(P1), length(P2)]);
            Pj = Pj(randperm(length(Pj) ,minClusN));
            P1 = P1(randperm(length(P1) ,minClusN));
            P2 = P2(randperm(length(P2) ,minClusN));
            [~,~,adstat1,~] = adtest(Pj);adstat1 = adstat1/length(Pj);
            [~,~,adstat11,~] = adtest(P1);adstat11 = adstat11/length(P1);
            [~,~,adstat22,~] = adtest(P2);adstat22 = adstat22/length(P2);
            if ~(adstat11 < adstat1 && adstat22 < adstat1)
                K = K - 1;
            end
        end
    end
    
    warning('on', 'stats:adtest:OutOfRangePLow');
    warning('on', 'stats:kmeans:FailedToConverge');
end
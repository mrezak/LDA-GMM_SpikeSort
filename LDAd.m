% LDA implementation
% Input:
% X, columns are features, rows are samples
% inplbl, training labels
% Output:
% W, projection matrix
% TrS, trace ratio value

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

function [W,TrS] = LDAd(X, inplbl, d)
    [grpinx,~] = grp2idx(inplbl);
    clsN = max(grpinx); %number of classes
    if nargin < 3, d = clsN-1; end
    if isempty(d) || d > clsN-1, d = clsN-1; end
    totsamp = size(X,2);
    m = mean(X,2);
    Sw = zeros(size(X,1));
    Sb=Sw;
    for k=1:clsN
        Xi = X(:, grpinx==k);    %class samples
        pri = size(Xi, 2)/totsamp;    %class prior
        mi = mean(Xi, 2);    %class mean
        Si = cov(Xi');  % class covariance
        Sw = Sw + pri*Si;     %within class scatter
        Sb = Sb + pri*(mi-m)*(mi-m)'; %Between class scatter
    end

    % Optimization

    [eigVec,eigVal] = eig(Sb,Sw);
    [~,inx] = sort(diag(eigVal),'descend');
    eigVec = eigVec(:,inx);
    W = eigVec(:,1:d); 

    TrS = trace(W'*Sb*W)/trace(W'*Sw*W);
end
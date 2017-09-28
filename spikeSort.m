% Main wrapper function to run LDA-Km or LDA-GMM algorithms 
% with estimating the number of cluster

% [labels, projU] = spikeSort(spikes, 'PARAM1',val1, 'PARAM2',val2, ...)
% Required:
% spikes, Input spike matrix, dim:(n_spikes, n_sample_per_spike)
%
% Optional (parameter name/value pairs):
%
%     'sortMethod'   -  Sorting method 'LDA-Km' or 'LDA-GMM' (default).
%   'numClusRange'   -  Search range to determine the number of cluster. 
%                       e.g. [3] to force 3 clusters. default = [2, 6] 
%            'Dim'   -  LDA subspace dimension. default = 2
%        'maxIter'   -  Maximum number of iteration between subspace
%                       selection and clustering. default = 30
%  'minSampleSize'   -  Sample size for initialization. 
%                       default = min(1000, total number of spikes.
%       'SampIter'   -  Number of iterations for initialization. default = 5
%    'minClusSize'   -  Minimum number of spikes to form a cluster. 
%                       defaut = 50
%      'earlyStop'   -  Stop the search for the number of clusters on the
%                       first occurance of over-clustering. default = false
%         'doPlot'   -  Plot the sorting results. default = true



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


function [labels, projU] = spikeSort(spikes, varargin)

    N = size(spikes, 1);
    p = inputParser;
    defaultsortMethod = 'LDAGMM';
    defaultnumClusRange = [2 6];
    defaultDim = 2;
    defaultmaxIter = 30;
    defaultminSampleSize = min(1000, N);
    defaultSampIter = 5;
    defaultminClusSize = 50;
    defaultearlyStop = false;
    
    addRequired(p,'spikes', @isnumeric);
    addParameter(p,'sortMethod',defaultsortMethod, @ischar);
    addParameter(p,'numClusRange',defaultnumClusRange, @isnumeric);
    addParameter(p,'Dim',defaultDim, @isnumeric);
    addParameter(p,'maxIter', defaultmaxIter, @isnumeric);
    addParameter(p,'minSampleSize',defaultminSampleSize, @isnumeric);
    addParameter(p,'SampIter',defaultSampIter, @isnumeric);
    addParameter(p,'minClusSize', defaultminClusSize, @isnumerical);
    addParameter(p,'earlyStop', defaultearlyStop, @islogical);
    addParameter(p,'doPlot', true, @islogical);
    
    parse(p,spikes,varargin{:});
    spikes = p.Results.spikes;
    sortMethod = p.Results.sortMethod;
    numClusRange = p.Results.numClusRange;
    d = p.Results.Dim;
    minSampleSize = p.Results.minSampleSize;
    SampIter = p.Results.SampIter;
    earlyStop = p.Results.earlyStop;
    doPlot = p.Results.doPlot;
    maxiter = p.Results.maxIter;
    minClusSize = p.Results.minClusSize;
    
    K_vec = numClusRange(1):numClusRange(end);
    k_det = nan(size(K_vec));
    i = 0;
    for k = K_vec
        i = i + 1;
        switch sortMethod
            case 'LDAGMM'
                [H{i},U{i},Y{i},V] = runLDA_GMM(spikes, k, d, minSampleSize, SampIter, maxiter);
            case 'LDAKM'
                [H{i},U{i},Y{i},V] = runLDA_Km(spikes, k, d, minSampleSize, SampIter, maxiter);
            otherwise
                error('Invalid sort method! Must be ''LDAGMM'' or ''LDAKM''. ')
        end
        % for detecting the number of clusters
        Y_km = Y{i};
        H_km = H{i};
        hClus = histc(H_km(H_km>0), unique(H_km(H_km>0)));
        if any(hClus < minClusSize)
            warning('Cluster size is less than minimum allowed size. Removing cluster\n');
            continue;
        end
        %gscatter3(Y_km(:,:), H_km, []);
        k_det(i) = numClusters(Y_km, H_km, minClusSize);

        if earlyStop && (k_det(i) < k) 
            break;  % break the for loop
        end

    end
    
    k_final_idx = find(k_det == K_vec, 1, 'last');
    k_final = K_vec(k_final_idx);
    
    % return the final labels and projection matrix
    if isempty(k_final_idx)
        warning(['Number of clusters not in range! Choosing smallest number in range:' num2str(K_vec)]);
        k_final_idx = 1;
    end
    labels = H{k_final_idx};
    projU = U{k_final_idx};
    
    % plotting
    if doPlot
        % Plot estimated number of cluster 
        figure; hold on;
        plot(K_vec,K_vec)
        plot(K_vec,k_det)
        plot(k_final,k_final, '*');
        D = Y{k_final_idx};
        hold off
        
        % Plot projections 
        gscatter3(D, labels, []);
        axis off;
        title(sortMethod)

    end
    

    
end
            
        
        
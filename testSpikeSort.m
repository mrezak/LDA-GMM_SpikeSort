% Test function to run LDA-Km and LDA-GMM algorithms

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

% Detected spikes are saved into 'spikes'
function testSpikeSort()
datapath = 'data/';
close all

% ------- Run Simulated on the wave_clus data and on real hc1 data from https://crcns.org/download
flist = dir([datapath 'spikes*.mat']);
for fname = flist'
    
    % Load extracted spikes
    fprintf(1, 'Processing %s\n', fname.name);
    load([datapath fname.name], 'spikes');
    % select first 3000 spikes
    if size(spikes,1) > 4000, spikes = spikes(1:4000,:); end
    [labels, projU] = spikeSort(spikes, 'sortMethod', 'LDAGMM', 'numClusRange', [2 5], 'Dim', 2);
    title(sprintf('Sorted spike clusters on %s data', fname.name), 'Interpreter', 'none');
    %disp('Press any key to process the next sample data..')
    %pause
end

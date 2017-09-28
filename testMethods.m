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
function testMethods()

% ------- Run on synthetic wave_clus data
datapath = 'data/';
close all
f0 = '2';
for f1= {'005','01','015','02'}
    f1 = f1{1};
    fname = ['spikes_C_Difficult' f0 '_noise' f1 '.mat'];
    load([datapath fname]);
    %load([datapath 'times_' fname]);
    % Obtaining ground truth labels
    %spikes_labels = get_org_class(cluster_class(:,2),(spike_times{1}+26)*samplingInterval,spike_class{1})';
    %spikes_overlap = logical(get_org_class(cluster_class(:,2),(spike_times{1}+26)*samplingInterval,spike_class{2}));
    %spikes_labels(spikes_overlap) = -1;     %discarding overlaping spikes in Quality Measurement
    %save([datapath 'spikes_' fname], 'spikes', 'spikes_labels');
    % spikes_labels: ground truth spike labels

    % Run spike sorting
    runSpikeSorting(spikes, spikes_labels)
end

% ------- Run on real hc1 data from https://crcns.org/download
flist = dir([datapath 'spikes_d*Data.mat']);
for fname = flist(1:end)'

    %load([datapath fname.name],'data');
    %fData = neural_filt(data(9e5:end),10e3,0,300,4000);
    %[spikes,~] = detect_spike(fData, 'AMP', 4, 64, 10, -1, -1); 
    %save([datapath 'spikes_' fname.name], 'spikes');
    
    % Load extracted spikes
    load([datapath fname.name], 'spikes');
    spikes = spikes';
    % select 3000 spikes
    if size(spikes,1) > 4000, spikes = spikes(1:4000,:); end
    spikes=spikes./max(spikes(:))*10;
    runSpikeSorting(spikes', [])
end

end


function runSpikeSorting(spikes, spikes_labels)

    figure
    %% PCA-kmeans
    [Y] = pca(spikes','NumComponents', 2);
    h = subplot(2,2,1);
    gscatter3(Y(:,1:2)', spikes_labels, [], h);
    xlim(mean(Y(:,1))+[-3*std(Y(:,1)),3*std(Y(:,1))])
    ylim(mean(Y(:,2))+[-3*std(Y(:,2)),3*std(Y(:,2))])
    axis off;
    title('PCA-Kmeans (colors from ground truth)')
    
    %% PCA-kmeans- derivative
    [Y] = pca(diff(spikes'),'NumComponents', 2);
    h = subplot(2,2,2);
    gscatter3(Y(:,1:2)', spikes_labels, [], h);
    xlim(mean(Y(:,1))+[-3*std(Y(:,1)),3*std(Y(:,1))])
    ylim(mean(Y(:,2))+[-3*std(Y(:,2)),3*std(Y(:,2))])
    axis off;
    title('DD-PCA-Kmeans (colors from ground truth)')

    %% LDA-kmeans PROPOSED METHOD (EMBC paper)
    [labels, U] = runLDA_Km(spikes, 3, 2, 500, 10, 30);
    D = U'*spikes';
    h = subplot(2,2,3);
    gscatter3(D(1:2,:), labels, [], h);
    Y = D';
    xlim(mean(Y(:,1))+[-3*std(Y(:,1)),3*std(Y(:,1))])
    ylim(mean(Y(:,2))+[-3*std(Y(:,2)),3*std(Y(:,2))])
    axis off;
    title('LDA-Kmeans')

    %% LDA-GMM PROPOSED METHOD (JNE paper)
    [labels, ~, D, ~] = runLDA_GMM(spikes, 3, 2, 500, 10, 30);
    h = subplot(2,2,4);
    gscatter3(D(1:2,:), labels, [], h);
    Y = D';
    xlim(mean(Y(:,1))+[-3*std(Y(:,1)),3*std(Y(:,1))])
    ylim(mean(Y(:,2))+[-3*std(Y(:,2)),3*std(Y(:,2))])
    axis off;
    title('LDA-GMM')
    
end


% function labels = get_org_class(detected_times,spike_times,spike_class)
% 
% i=0;
% labels = zeros(1,length(detected_times));
% for t = detected_times(:)'
%     [~,idx] = min(abs(t-spike_times));
%     i=i+1;
%     if idx<=length(spike_class)
%         labels(i) = spike_class(idx);
%     else
%         labels(i) = 0;
%     end
% end

% end
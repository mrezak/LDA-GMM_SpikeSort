# LDA-GMM_SpikeSort
Unsupervised Spike Sorting package based on discriminative subspace learning
==================

The MATLAB implementation of a noise-robust unsupervised spike sorting algorithm based on discriminative subspace learning with outlier handling, as proposed in: **Mohammad Reza Keshtkaran and Zhi Yang, "Noise-robust unsupervised spike sorting based on discriminative subspace learning with outlier handling,” Journal of Neural Engineering, vol. 14, no. 3, p. 36003, Jun. 2017, (available at http://iopscience.iop.org/article/10.1088/1741-2552/aa6089)**

If you find this program useful in your work, please give credit by citing the above paper. If you have any question regarding the algorithm or implementation, do not hesitate to write to the authors at one of the following addresses: mrezak.github AT gmail.com, keshtkaran AT u.nus.edu

You need MATLAB software to use this program.

## Usage
You can run the automated spike sorting algorithm on the `spikes` matrix (rows are spikes, columns are samples) which contains the detected and aligned spikes you want to sort:
```
>>> [labels, projU] = spikeSort(spikes)
```
You can pass in the optional arguments:
```
Main wrapper function to run LDA-Km or LDA-GMM algorithms
with estimating the number of clusters

[labels, projU] = spikeSort(spikes, 'PARAM1',val1, 'PARAM2',val2, ...)
Required:
spikes, Input spike matrix, dim:(n_spikes, n_sample_per_spike)

Optional (parameter name/value pairs):

   'sortMethod'   -  Sorting method 'LDA-Km' or 'LDA-GMM' (default).
 'numClusRange'   -  Search range to determine the number of cluster.
                    e.g. [3] to force 3 clusters. default = [2, 6]
          'Dim'   -  LDA subspace dimension. default = 2
      'maxIter'   -  Maximum number of iteration between subspace
                    selection and clustering. default = 30
'minSampleSize'   -  Sample size for initialization.
                     default = min(1000, total number of spikes.
     'SampIter'   -  Number of iterations for initialization. default = 5
  'minClusSize'   -  Minimum number of spikes to form a cluster. defaut = 50
    'earlyStop'   -  Stop the search for the number of clusters on the
                     first occurance of over-clustering. default = false
       'doPlot'   -  Plot the sorting results. default = true

```

## Licence
Copyright (C) 2017, Mohammad Reza Keshtkaran <keshtkaran.github@gmail.com>

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>.




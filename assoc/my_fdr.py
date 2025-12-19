# Copyright 2018, Wenjia Bai. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the 'License');
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an 'AS IS' BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
import numpy as np


def p_adjust_fdr(p):
    # FDR correction for multiple testing
    # The implementation provides consistent results as R p.adjust function.
    # Use a new np array, otherwise p will be modified.
    p2 = np.zeros(p.shape, dtype=np.float32)
    idx = np.argsort(p)
    n = len(p)
    p2[idx] = (p[idx] * n) / np.arange(1, n + 1)
    p2[p2 > 1] = 1
    return p2


def fdr_threshold(p, q):
    # p: vector of p-values
    # q: false discovery rate level
    #
    # return values
    # pID: p-value threshold based on independence or positive dependence
    # pN: nonparametric p-value threshold
    #
    # This function takes a vector of p-values and a FDR rate.
    # It returns two p-value thresholds, one based on an assumption of
    # independence or positive dependence, and one that makes no assumptions
    # about how the tests are correlated. For imaging data, an assumption of
    # positive dependence is reasonable, so it should be OK to use the first
    #  (more sensitive) threshold.
    #
    # This implementation provides consistent results as the FDR function at
    # https://warwick.ac.uk/fac/sci/statistics/staff/academic-research/nichols/software/fdr
    # https://warwick.ac.uk/fac/sci/statistics/staff/academic-research/nichols/software/fdr/FDR.m
    p2 = p[~np.isnan(p)]
    p2 = np.sort(p2)
    n = len(p2)
    I = np.arange(1, n + 1)
    cVID = 1
    cVN = np.sum(1 / I)

    idx = np.nonzero(p2 <= ((I * q) / (n * cVID)))[0]
    pID = p2[np.max(idx)] if len(idx) >= 1 else 0

    idx = np.nonzero(p2 <= ((I * q) / (n * cVN)))[0]
    pN = p2[np.max(idx)] if len(idx) >= 1 else 0
    return pID, pN

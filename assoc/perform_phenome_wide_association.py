# Copyright 2019, Wenjia Bai. All Rights Reserved.
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
"""
    Perform phenome-wide association studies between imaging phenotypes and non-imaging phenotypes.
"""
import os
import numpy as np
import pandas as pd
import datetime
import scipy.stats
import math
import re
import csv
import statsmodels.api as sm
from ukbb_cardiac.data.ukb_field_categories import *
from ukbb_cardiac.assoc.my_fdr import fdr_threshold
import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt


def normalise(x):
    return (x - np.mean(x)) / np.std(x)


def rank_normalise(x):
    # Rank-based inverse normal transform
    # Please refer to the function inormal() in the FSLNets package

    # Get the rank of the values in x
    ri = np.argsort(np.argsort(x))

    # Correct for the ranks of repeated values
    # argsort assign different ranks for these values
    # We fill them with the same value
    u, inv_idx = np.unique(x, return_inverse=True)
    sii = np.sort(inv_idx)
    repeated_idx = np.unique(sii[np.diff(np.append(sii, 1)) == 0])
    for i in repeated_idx:
        ri[inv_idx == i] = np.mean(ri[inv_idx == i])

    # Perform inverse normal transform
    # ri + 1 so that the rank starts from 1, to be consistent with Karla's Matlab code
    # p squashes the rank into the range of [0, 1]
    # erfinv can generate a distribution from 2 * p - 1 with 0 mean and 1 standard deviation
    N = len(x)
    ri = ri + 1
    c = 3.0 / 8
    p = (ri - c) / (N - 2 * c + 1)
    y = math.sqrt(2) * scipy.special.erfinv(2 * p - 1)
    return y


if __name__ == '__main__':
    # # # # # # # # # # # # # # # # # # # #
    # Step 1: read imaging phenotypes
    # # # # # # # # # # # # # # # # # # # #
    # Cardiac imaging phenotypes
    data_path = 'uk_biobank_data_path'
    df_idp = pd.read_csv('{0}/clinical_measures.csv'.format(data_path), index_col=0)

    # # # # # # # # # # # # # # # # # # # #
    # Step 2: read non-imaging phenotypes
    # # # # # # # # # # # # # # # # # # # #
    # Participant characteristics
    info_path = 'uk_biobank_info_path'

    # ukb_catname[1001] = 'Primary demographics'
    # ukb_catname[1002] = 'Early life factors'
    # ukb_catname[1007] = 'Education and employment'
    # ukb_catname[1004] = 'Diet summary'
    # ukb_catname[100051] = 'Alcohol summary'
    # ukb_catname[100058] = 'Smoking summary'
    # ukb_catname[100054] = 'Physical activity'
    # ukb_catname[1006] = 'Physical measure summary'
    # ukb_catname[1003] = 'Self-reported medical conditions'
    # ukb_catname[1018] = 'Mental health'
    # ukb_catname[100026] = 'Cognitive function'
    df = []
    category_of_interest = [1001, 1002, 1007, 1004, 100051, 100058, 100054, 1006, 1003, 1018, 100026]
    for cid in category_of_interest:
        df2 = pd.read_csv('{0}/ukb_cardiac_image_subset_{1}.csv'.format(info_path, ukb_catname[cid]),
                          header=[0, 1], index_col=0)
        df += [df2]
    df = pd.concat(df, axis=1)
    df = df.loc[df_idp.index]

    # A dictionary mapping a field ID to its name
    # Category ID, Field ID, Category, Field description
    field_names = {}
    ukb_in = 'ukb_field_added.txt'
    with open(ukb_in, 'r', encoding='latin-1') as f:
        reader = csv.reader(f, delimiter='\t')
        for row in reader:
            field_names[int(row[1])] = row[3]

    # A dictionary mapping a field ID to a category ID
    d_cat = {}
    for cid in category_of_interest:
        print(ukb_catname[cid])
        for fid in ukb_cat[cid]:
            d_cat[fid] = cid

    # # # # # # # # # # # # # # # # # # # #
    # Step 3: confounding factors
    # # # # # # # # # # # # # # # # # # # #
    sex = df['Sex', '31-0.0'].values
    # Age provided by UK Biobank (21003-2.0) seems to be floored, i.e. with half a year error.
    # To get more accurate age values, we calculate age by date.
    age = np.zeros(len(df))
    for i in range(len(df)):
        # Calculate age
        d1 = datetime.date(df.iloc[i]['Year of birth', '34-0.0'], df.iloc[i]['Month of birth', '52-0.0'], 15)
        s = df.iloc[i]['Date of attending assessment centre', '53-2.0']
        d2 = datetime.date(int(s[:4]), int(s[5:7]), int(s[8:]))
        age[i] = np.round((d2 - d1).days / 365.25, 1)
    weight = df['Weight', '21002-2.0'].values
    bmi = df['Body mass index (BMI)', '21001-2.0'].values
    height = np.round(np.sqrt(weight / bmi) * 100)

    # Keep the rows with age, sex, weight and height information
    valid_idx = ~np.isnan(age) & ~np.isnan(sex) & ~np.isnan(weight) & ~np.isnan(height)
    df = df[valid_idx]
    df_idp = df_idp[valid_idx]
    sex = sex[valid_idx]
    age = age[valid_idx]
    sex_age = sex * age  # Sex and age interaction
    weight = weight[valid_idx]
    height = height[valid_idx]

    # Confounding factors
    conf = np.stack((sex, age, sex_age, weight, height), axis=1)

    df_conf = pd.DataFrame(conf, index=df.index, columns=['Sex', 'Age', 'Sex * Age', 'Weight', 'Height'])
    df_conf.to_csv('confounders.csv')

    # Remove confounding factors from df
    df = df.drop(columns=[('Sex', '31-0.0'),
                          ('Year of birth', '34-0.0'),
                          ('Month of birth', '52-0.0'),
                          ('Date of attending assessment centre', '53-0.0'),
                          ('Date of attending assessment centre', '53-1.0'),
                          ('Date of attending assessment centre', '53-2.0'),
                          ('Age when attended assessment centre', '21003-0.0'),
                          ('Age when attended assessment centre', '21003-1.0'),
                          ('Age when attended assessment centre', '21003-2.0'),
                          ('Weight', '21002-0.0'),
                          ('Weight', '21002-1.0'),
                          ('Weight', '21002-2.0'),
                          ('Body mass index (BMI)', '21001-0.0'),
                          ('Body mass index (BMI)', '21001-1.0'),
                          ('Body mass index (BMI)', '21001-2.0')])

    # # # # # # # # # # # # # # # # # # # #
    # Step 4: clean, de-confound and normalise data
    # # # # # # # # # # # # # # # # # # # #
    # This part of the code was adapted from Karla Miller's Matlab code at
    # https://www.fmrib.ox.ac.uk/ukbiobank/gwaspaper/index.html
    # Step 4.1: cleaning
    n_subj, n_col = df.shape
    bad_vars = []
    for i in range(n_col):
        val = df.iloc[:, i]

        # Discard columns which not numbers
        if not np.issubdtype(df.dtypes[i], np.number):
            bad_vars += [i]
            continue

        # Assume negative values are invalid, set them to NaN
        # There are also a lot of empty values, which are already NaN
        val[val < 0] = np.nan
        df.iloc[:, i] = val

        # Valid indices
        valid_idx = ~np.isnan(val)

        # Discard columns with more than 90% missing data
        if np.sum(valid_idx) < (0.1 * n_subj):
            bad_vars += [i]
            continue

        # Discard columns with over 95% elements with the exactly same value
        val_unique, counts = np.unique(val[valid_idx], return_counts=True)
        if np.max(counts) >= (0.95 * np.sum(valid_idx)):
            bad_vars += [i]
            continue

    for i in range(n_col):
        for j in range(i + 1, n_col):
            if i in bad_vars or j in bad_vars:
                continue

            # Discard columns with very high correlation
            val_i = df.iloc[:, i]
            val_j = df.iloc[:, j]
            valid_idx = ~np.isnan(val_i) & ~np.isnan(val_j)
            if np.sum(valid_idx) == 0:
                continue
            cc, _ = scipy.stats.pearsonr(val_i[valid_idx], val_j[valid_idx])
            if cc > 0.9999:
                # Keep the column with more valid elements
                if np.sum(~np.isnan(val_i)) > np.sum(~np.isnan(val_j)):
                    bad_vars += [j]
                else:
                    bad_vars += [i]

    # The cleaned data
    bad_vars = np.unique(bad_vars)
    keep_vars = sorted(list(set(np.arange(n_col)) - set(bad_vars)))
    df = df.iloc[:, keep_vars]
    print('{0} columns kept after data cleaning.'.format(df.shape[1]))

    # Step 4.2: normalise confounding factors
    conf = (conf - np.mean(conf, axis=0)) / np.std(conf, axis=0)

    # Step 4.3: normalise non imaging phenotypes
    df_cont = pd.read_csv('continuous.csv', index_col=0)

    n_col = df.shape[1]
    for i in range(n_col):
        val = df.iloc[:, i]
        valid_idx = ~np.isnan(val)
        x = val[valid_idx]

        # Field ID
        field_id = int(df.columns[i][1].split('-')[0])
        is_continuous = df_cont.loc[field_id]['continuous']

        if is_continuous:
            # If it is a continuous variable, perform standard normalisation.
            x = normalise(x)
        else:
            # If we are not sure whether it is a continuous or categorical variable, convert it
            # into a continuous variable using rank-based inverse normal transform.
            x = rank_normalise(x)
        df.iloc[:, i][valid_idx] = x
    df.to_csv('normalised_non_IDPs.csv')

    # Step 4.4: de-confound and normalise IDPs
    n_row = conf.shape[1]
    n_col = df_idp.shape[1]
    beta = np.zeros((n_row, n_col))
    for i in range(n_col):
        val = df_idp.iloc[:, i]
        valid_idx = ~np.isnan(val)
        x = val[valid_idx]
        beta[:, i] = np.dot(np.linalg.pinv(conf[valid_idx]), x)
        x = x - np.dot(conf[valid_idx], beta[:, i])
        x = normalise(x)
        df_idp.iloc[:, i][valid_idx] = x
    df_idp.to_csv('normalised_IDPs.csv')

    df_beta = pd.DataFrame(beta,
                           index=['sex', 'age', 'sex * age', 'weight', 'height'],
                           columns=df_idp.columns)
    df_beta.to_csv('beta_IDPs.csv')

    # # # # # # # # # # # # # # # # # # # # #
    # # Step 5: uni-variate correlation studies
    # # # # # # # # # # # # # # # # # # # # #
    M = df_idp.shape[1]
    N = df.shape[1]
    corr = np.zeros((M, N))
    corr_p = np.zeros((M, N))
    for i in range(M):
        for j in range(N):
            # Remove NaNs
            x = df_idp.iloc[:, i]
            y = df.iloc[:, j]
            valid_idx = ~np.isnan(x) & ~np.isnan(y)
            x = x[valid_idx]
            y = y[valid_idx]

            # Pearson correlation
            cc, p_val = scipy.stats.pearsonr(x, y)
            corr[i, j] = cc
            corr_p[i, j] = p_val

    # For p-value of 0, assign it with the mininal positive floating value
    # so that we can calculate the logarithm for the Manhattan plot
    corr_p[corr_p == 0] = np.finfo(np.float64).tiny

    # Logarithm
    log_corr_p = - np.log10(corr_p)

    # Save the tables
    df_corr = pd.DataFrame(corr, index=df_idp.columns, columns=df.columns)
    df_p = pd.DataFrame(corr_p, index=df_idp.columns, columns=df.columns)
    df_log_p = pd.DataFrame(log_corr_p, index=df_idp.columns, columns=df.columns)

    df_corr.to_csv('table_corr.csv')
    df_p.to_csv('table_p.csv')
    df_log_p.to_csv('table_log_p.csv')

    corr = df_corr.values
    corr_p = df_p.values
    log_corr_p = df_log_p.values

    # Bonferroni correction
    M, N = corr.shape
    p_bonf = 0.05 / (M * N)

    # FDR correction
    p_fdr, _ = fdr_threshold(corr_p.flatten(), 0.05)

    # Number of phenotypes that is significantly associated with at least one of the IDPs
    print('p_bonf = {0}'.format(p_bonf))
    print('p_fdr = {0}'.format(p_fdr))
    print('Number of correlations reaching Bonferroni threshold = {0}'.format(np.sum(corr_p < p_bonf)))
    print('Number of correlations reaching FDR threshold = {0}'.format(np.sum(corr_p < p_fdr)))
    print('Number of phenotypes reaching Bonferroni threshold = {0}'.format(np.sum(np.sum(corr_p < p_bonf, axis=0) > 0)))
    print('Number of phenotypes reaching FDR threshold = {0}'.format(np.sum(np.sum(corr_p < p_fdr, axis=0) > 0)))

    # # # # # # # # # # # # # # # # # # # #
    # Step 6: Manhattan plot
    # # # # # # # # # # # # # # # # # # # #
    # The category for each column
    # They should be in ascending order. Otherwise, we need to sort the category IDs.
    category = []
    for field_name, field_id in df.columns:
        category += [d_cat[int(field_id.split('-')[0])]]
    category = np.array(category)

    # If we plot in ascending order, the new colours (LA and RA) will overwrite old colours (LV and RV).
    # Instead, we plot the columns in a random order.
    plt_order = np.arange(M)
    plt_order = list(set(plt_order) - set([0, 6, 10, 14, 18, 21]))
    np.random.seed(0)
    np.random.shuffle(plt_order)
    plt_order = [0, 6, 10, 14, 18, 21] + plt_order

    # The Manhattan plot
    plt.figure()
    table = []
    for i in plt_order:
        if df_idp.columns[i][:1] == 'E':
            s = 'LV'
        elif df_idp.columns[i][:2] == 'WT':
            s = 'LV'
        elif df_idp.columns[i][1:3] == 'Ao':
            s = df_idp.columns[i][:3]
        else:
            s = df_idp.columns[i][:2]

        for j in range(N):
            line = [j, log_corr_p[i, j], corr[i, j], abs(corr[i, j]), s]
            table += [line]
    df_table = pd.DataFrame(table, columns=['x', 'log_p', 'corr', 'Correlation', 'Anatomy'])

    ax = sns.scatterplot(x='x', y='log_p', hue='Anatomy', size='Correlation',
                         sizes=(20, 200), size_norm=mpl.colors.Normalize(vmin=0, vmax=0.3),
                         data=df_table, alpha=0.8)
    plt.plot([0, N], [-math.log10(p_bonf), -math.log10(p_bonf)], 'k--', linewidth=1, alpha=0.8)
    plt.text(-1, -math.log10(p_bonf), 'Bonf', horizontalalignment='right', fontsize=11)
    handles, labels = ax.get_legend_handles_labels()
    labels[-1] = '0.3'
    ax.legend(handles, labels)

    xticks = []
    xticklabels = []
    unique_category = category_of_interest
    c_last = unique_category[-1]
    for c in unique_category:
        cid = np.nonzero(category == c)[0]
        x = np.max(cid) + 0.5
        if c != c_last:
            plt.plot([x, x], [0, 300], 'k--', linewidth=0.5)
        xticks += [np.mean(cid)]

        if c == 1001:
            xticklabels += ['Demo         \ngraphics         ']
        elif c == 1002:
            xticklabels += ['Early  \nlife  ']
        elif c == 1007:
            xticklabels += [' Education\n employment']
        elif c == 1004:
            xticklabels += ['Lifestyle   \ndiet   ']
        elif c == 100051:
            xticklabels += ['Lifestyle  \nalcohol  ']
        elif c == 100058:
            xticklabels += ['       Lifestyle\n       smoking']
        elif c == 100054:
            xticklabels += ['    Lifestyle\n    physical\n    activity']
        elif c == 1003:
            xticklabels += ['Self-reported\nmedical conditions']
        elif c == 1006:
            xticklabels += ['Physical\nmeasures']
        else:
            xticklabels += [re.sub(' ', '\n', ukb_catname[c])]

    plt.xlim(0, N)
    plt.ylim(0, 300)
    plt.xticks(xticks, xticklabels, fontsize=10)
    plt.yticks(fontsize=12)
    plt.xlabel('Non-imaging phenotypes', fontsize=12, fontweight='bold')
    plt.ylabel(r'$-\log_{10}(p)$', fontsize=12, fontweight='bold')
    fig = plt.gcf()
    fig.set_size_inches(16, 8)
    plt.tight_layout()
    plt.savefig('manhattan_plot.pdf', bbox_inches='tight')

    # # # # # # # # # # # # # # # # # # # #
    # Step 7: report top hits in latex format
    # # # # # # # # # # # # # # # # # # # #
    df_log_p = pd.read_csv('table_log_p.csv', index_col=0, header=[0, 1])
    df_corr = pd.read_csv('table_corr.csv', index_col=0, header=[0, 1])

    rows = {'LV': ['LVEDV (mL)', 'LVESV (mL)', 'LVSV (mL)', 'LVEF (%)',
                   'LVCO (L/min)', 'LVM (g)'] + list(df_log_p.index[24:]),
            'RV': ['RVEDV (mL)', 'RVESV (mL)', 'RVSV (mL)', 'RVEF (%)'],
            'LA': ['LAV max (mL)', 'LAV min (mL)', 'LASV (mL)', 'LAEF (%)'],
            'RA': ['RAV max (mL)', 'RAV min (mL)', 'RASV (mL)', 'RAEF (%)'],
            'AAo': ['AAo max area (mm2)', 'AAo min area (mm2)', 'AAo distensibility (10-3 mmHg-1)'],
            'DAo': ['DAo max area (mm2)', 'DAo min area (mm2)', 'DAo distensibility (10-3 mmHg-1)']}

    # Go through all the non-imaging phenotypes
    cols = df_log_p.columns
    for k in rows.keys():
        # For each anatomical structure, find the non-imaging phenotype that is most related
        row = rows[k]
        log_p = df_log_p.loc[row]
        corr = df_corr.loc[row]
        log_p_max = np.max(log_p, axis=0)
        log_p_idx = np.argmax(log_p.values, axis=0)
        top_hits = sorted(range(len(log_p_max)), key=lambda x: log_p_max[x], reverse=True)

        # Print the top hits
        for i in range(20):
            c = top_hits[i]
            r = log_p_idx[c]
            if i == 0:
                line = '\multirow{{5}}{{*}}{{{0}}}'.format(k)
            else:
                line = ''
            line += ' & {0:.1f} & {1:.2f} & {2} & {3} & {4} \\\\'.format(
                log_p_max[c], corr.iloc[r, c], row[r], cols[c][0], cols[c][1])
            line = re.sub('WT_', 'Wall thickness ', line)
            line = re.sub('_', ' ', line)
            line = re.sub(' \(%\)', '', line)
            line = re.sub(' \(g\)', '', line)
            line = re.sub(' \(mL\)', '', line)
            line = re.sub(' \(10-3 mmHg-1\)', '', line)
            line = re.sub(' \(mm2\)', '', line)
            line = re.sub(', automated reading', '', line)
            line = re.sub(' 10\+ minutes', '', line)
            print(line)
        print('\hline')

    # Focus on the mental health category
    df_log_p_cat = df_log_p.iloc[:, category == 1018]
    df_corr_cat = df_corr.iloc[:, category == 1018]

    cols = df_log_p_cat.columns
    for k in rows.keys():
        # For each anatomical structure, find the non-imaging phenotype that is most related
        row = rows[k]
        log_p = df_log_p_cat.loc[row]
        corr = df_corr_cat.loc[row]
        log_p_max = np.max(log_p, axis=0)
        log_p_idx = np.argmax(log_p.values, axis=0)
        top_hits = sorted(range(len(log_p_max)), key=lambda x: log_p_max[x], reverse=True)

        # Print the top hits
        for i in range(20):
            c = top_hits[i]
            r = log_p_idx[c]
            if i == 0:
                line = '\multirow{{5}}{{*}}{{{0}}}'.format(k)
            else:
                line = ''
            line += ' & {0:.1f} & {1:.2f} & {2} & {3} & {4} \\\\'.format(
                log_p_max[c], corr.iloc[r, c], row[r], cols[c][0], cols[c][1])
            line = re.sub('WT_', 'Wall thickness ', line)
            line = re.sub('_', ' ', line)
            line = re.sub(' \(%\)', '', line)
            line = re.sub(' \(g\)', '', line)
            line = re.sub(' \(mL\)', '', line)
            line = re.sub(' \(10-3 mmHg-1\)', '', line)
            line = re.sub(' \(mm2\)', '', line)
            line = re.sub(', automated reading', '', line)
            line = re.sub(' 10\+ minutes', '', line)
            print(line)
        print('\hline')

    # Focus on the cognitive function category
    df_log_p_cat = df_log_p.iloc[:, category == 100026]
    df_corr_cat = df_corr.iloc[:, category == 100026]

    cols = df_log_p_cat.columns
    for k in rows.keys():
        # For each anatomical structure, find the non-imaging phenotype that is most related
        row = rows[k]
        log_p = df_log_p_cat.loc[row]
        corr = df_corr_cat.loc[row]
        log_p_max = np.max(log_p, axis=0)
        log_p_idx = np.argmax(log_p.values, axis=0)
        top_hits = sorted(range(len(log_p_max)), key=lambda x: log_p_max[x], reverse=True)

        # Print the top hits
        for i in range(20):
            c = top_hits[i]
            r = log_p_idx[c]
            if i == 0:
                line = '\multirow{{5}}{{*}}{{{0}}}'.format(k)
            else:
                line = ''
            line += ' & {0:.1f} & {1:.2f} & {2} & {3} & {4} \\\\'.format(
                log_p_max[c], corr.iloc[r, c], row[r], cols[c][0], cols[c][1])
            line = re.sub('WT_', 'Wall thickness ', line)
            line = re.sub('_', ' ', line)
            line = re.sub(' \(%\)', '', line)
            line = re.sub(' \(g\)', '', line)
            line = re.sub(' \(mL\)', '', line)
            line = re.sub(' \(10-3 mmHg-1\)', '', line)
            line = re.sub(' \(mm2\)', '', line)
            line = re.sub(', automated reading', '', line)
            line = re.sub(' 10\+ minutes', '', line)
            line = re.sub(' \(trail #1\)', '', line)
            line = re.sub(' \(trail #2\)', '', line)
            line = re.sub('Ecc', '$E_{cc}$', line)
            line = re.sub('Ell', '$E_{ll}$', line)
            print(line)
        print('\hline')

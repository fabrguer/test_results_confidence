import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.graphics.gofplots import qqplot
from scipy.stats import normaltest
from scipy.stats import sem, t

np.random.seed(1234)


# Creates `pts_nr` random points for a given distribution. 
# =============================================================================

def random_data(distr, pts_nr):
    if distr == 'normal':
        return np.random.normal(loc=50, scale=10, size=pts_nr)
    if distr == 'uniform':
        return np.random.uniform(low=0, high=100, size=pts_nr)
    if distr == 'triangular': 
        return np.random.triangular(left=0, mode=90, right=100, size=pts_nr)
    if distr == 'exponential':
        return np.random.exponential(scale=25, size=pts_nr)
    
    
# Returns "avgs_nr" averages from a distribution created using "random_data"
# =============================================================================

def random_data_avg(distr, pts_nr, avgs_nr):
    avgs = []
    for i in range(avgs_nr):
        avg = np.mean(random_data(distr, pts_nr))
        avgs.append(avg)
    return avgs


# Computes std dev, CI for a distribution. PLots histogram, QQ plot, CI score
# =============================================================================

def CI_vs_samples(distributions, samples_list, samples_to_plot):

    avgs_nr = 10000
    fig, ax = plt.subplots(3,3, figsize=(16,15))
    
    results = []
    
    for idx, distribution in enumerate(distributions):
    
        CI_df_cols = ['n','Normal','Exp. Sigma','Sigma','CI_norm','CI_norm_score',
                      'CI_t','CI_t_score']
        CI_df = pd.DataFrame(columns=CI_df_cols)

        real_avg = np.mean(random_data_avg(distribution, 1000, avgs_nr))

        # Repeat for each fo the samples number
        for i, samples in enumerate(samples_list):

            data_avgs = []
            CI_norm_score = []
            CI_t_score = []

            # Repeat 10k for each samples number
            for j in range(avgs_nr):

                # Generates random data
                data = random_data(distribution, samples)
                data_avg = np.mean(data)
                data_avgs.append(data_avg)

                # Computes the CI assuming normal
                data_std = np.std(data)
                CI = 1.96*data_std/np.sqrt(samples)
                lower = data_avg - CI
                upper = data_avg + CI
                CI_norm_score.append(lower <= real_avg <= upper)

                # Computes the CI assuming t-distribution
                confidence = 0.95
                std_err = sem(data)
                h = std_err * t.ppf((1 + confidence) / 2, samples - 1)
                lower = data_avg - h
                upper = data_avg + h
                CI_t_score.append(lower <= real_avg <= upper)

            # Plots the histogram
            if samples in samples_to_plot:
                label = "n = {}".format(samples)
                color = color_lin_gradient(np.array([1,0,0]), np.array([0.2,0,1]), 
                            len(samples_to_plot))[samples_to_plot.index(samples)]
                ax[idx,0].hist(data_avgs, bins=50, label=label, color=color);
                qqplot(np.array(data_avgs), fit=True, line='45', ax=ax[idx,1], 
                            label=label, color=color);

            # Computes the std deviation
            is_normal = normaltest(data_avgs)[-1] > 0.05
            data_std = np.std(random_data(distribution, samples))
            expected_avgs_std = data_std/np.sqrt(samples)
            real_avgs_std = np.std(data_avgs)

            # Update the series and add to the dataframe
            CI_series = pd.Series(
                index= CI_df_cols,
                data = [samples, is_normal, expected_avgs_std, real_avgs_std, 2*CI, 
                        np.mean(CI_norm_score), 2*h, np.mean(CI_t_score)])
            CI_df = CI_df.append(CI_series, ignore_index=True)

        # Plots the graphs
        ax[idx,0].set_xlabel("Value")
        ax[idx,0].set_ylabel("Count")
        ax[idx,0].set_xlim(0,100);
        ax[idx,0].legend();
        ax[idx,1].legend();
        if distribution == 'exponential':
            ax[idx,1].set_xlim(-4,4);
            ax[idx,1].set_ylim(-4,6);

        ax[idx,2].set_xlabel("n")
        ax[idx,2].set_ylabel("CI score")
        ax[idx,2].plot(samples_list, CI_df['CI_norm_score'].values, 'o', 
                   label="Normal Approx (Eq.2)", color="crimson")
        ax[idx,2].plot(samples_list, CI_df['CI_t_score'].values, 'o', 
                   label="T-Distr Approx (Eq.3)", color="blue")
        ax[idx,2].plot(samples_list, len(samples_list)*[0.95], '--', 
                   label='Theoretical CI', color='black')
        ax[idx,2].set_xscale('log')
        ax[idx,2].legend();
    
        results.append(CI_df)
    
    return results


# Return a list of n colors from start_color to finish_color
# =============================================================================

def color_lin_gradient(start_color=[0,0,0], finish_color=[1,1,1], n=10):

    color_delta = finish_color - start_color
    color_step = color_delta/(n-1)

    list_of_colors = []
    list_of_colors.append(start_color)

    for i in range(1, n):
        color = list_of_colors[i-1] + color_step
        list_of_colors.append(color)

    return list_of_colors
from matplotlib import pyplot as plt
import numpy as np
import random

random.seed(45)

num_coins = 100


def toss(num_trials):
    '''
    num_trials: number of trials to be performed.
    
    return a numpy array of size num_trials with each entry representing the number of heads found in each trial

    Use for loops to generate the numpy array and 'random.choice()' to simulate a coin toss
    
    NOTE: Do not use predefined functions to directly get the numpy array. 
    '''
    global num_coins
    results = []

    ## Write your code here
    for i in range(num_trials):
        results.append(sum(random.choice([0, 1]) for _ in range(num_coins)))  # 1 represents heads and 0 represents
        # tails. So by summing all the values obtained we can get no of heads.
    return results


def plot_hist(trial):
    '''
    trial: vector of values for a particular trial.

    plot the histogram for each trial.
    Use 'axs' from plt.subplots() function to create histograms. You can search about how to use it to plot histograms.

    Save the images in a folder named "histograms" in the current working directory.  
    '''
    fig, axs = plt.subplots()

    ## Write your code here
    plt.hist(trial, edgecolor='black')
    axs.set_title(f'Histogram of Number of Heads for {num_trials} Trials')
    axs.set_xlabel('Number of Heads')
    axs.set_ylabel('Frequency')
    plt.savefig(f'histograms/histogram_{num_trials}_trials.png')
    plt.close(fig)


if __name__ == "__main__":
    num_trials_list = [10, 100, 1000, 10000, 100000]
    for num_trials in num_trials_list:
        heads_array = toss(num_trials)
        plot_hist(heads_array)

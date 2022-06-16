# Will Bourne 14/06/2022
# Louis Barnes 15/06/2022
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import itertools
import time
import os

start_time = time.time()

def flatten(xss):
    return [x for xs in xss for x in xs]

def save(data, file):
    with open(file, 'wb') as f:
        np.save(f, data)

def load(file):
    with open(file, 'rb') as f:
        return np.load(f,allow_pickle=True)


def read_data(filenames):
    byte_files = np.array([], dtype=bool)
    for file_link in filenames:
        with open(file_link, "rb") as file:
            # byte_files.append(list(file.read()))
            byte_file = []
            while byte := file.read(1):
                byte_string = format(int.from_bytes(byte, "little"), '08b')
                byte_file.extend(list(byte_string))
            byte_file = np.array(byte_file).astype(np.int)

            # byte_files.append(byte_file)

            byte_files = np.concatenate((byte_files, byte_file))

    return np.reshape(byte_files, (len(filenames),-1))


def plot_kde(number_files, fname):
    # plot the frequency density of all numbers (1 byte, 0-255) stored in the SRAM
    sns.set_style('whitegrid')
    for numbers in number_files:
        sns.kdeplot(numbers)
        print(len(numbers))
    plt.legend(['1', '2', '3'])
    plt.savefig("figs/{0}".format(fname))

    # find which numbers are different between runs on the same board
    mask = number_files[2] == number_files[1]


def hamming_distance(arr1, arr2):
    different_bits = np.bitwise_xor(arr1, arr2)
    distance = np.sum(different_bits)
    return distance


def hamming_distance_combinations(arr):
    hamming_distances = []
    # how many unique ways can we combine the datasets?
    combinations = [list(comb) for comb in itertools.combinations(range(len(arr[:,0])), 2)]
    for combination in combinations:
        hamming_distances.append(hamming_distance(arr[combination[0],:], arr[combination[1],:]))

    return hamming_distances, combinations


def find_constant_bits(arr):
    chance_bit_is_a_one = np.sum(arr.astype(np.float), axis=0) / len(arr[:,0])
    bits_that_dont_change = [(chance_bit_is_a_one == 1) | (chance_bit_is_a_one == 0)]
    return bits_that_dont_change


def inter_intra_mb_constant_bits(mb_list):
    """ :param mb_list: list of all microbit data arrays
        :return: mask of which bits are always constant across all 
microbits,
                 array of masks of which bits are always constant across 
each microbits"""
    # 1. work out which bits change & which don't between separate tests of the same microbit
    mb_const_bits = []
    for mb_data in mb_list:
        mb_const_bits.append(find_constant_bits(mb_data))
    mb_const_bits = np.array(mb_const_bits)
    save(mb_const_bits, "temp/mb_const_bits.npy")

    # 2. of the bits that don't change in each individual microbit, which remain constant across all the microbits?
    global_const_bits = [np.sum(mb_const_bits, axis=0) == len(mb_const_bits)]
    save(global_const_bits, "temp/global_const_bits.npy")

    return global_const_bits, mb_const_bits


def get_weighted_hamming_distances(mb_arrs, exp_arr, weightings):
    frac_hamming_dists = []
    for mb_arr in mb_arrs:
        different_bits = np.bitwise_xor(mb_arr.astype(bool), exp_arr.astype(bool))
        hamming_dist = np.sum(weightings[different_bits])
        # hamming_dist = np.sum(different_bits.astype(float))

        frac_hamming_dists.append(hamming_dist)
    return np.array(frac_hamming_dists, dtype=float)


def get_expected_hamming_distance(mean_arr, weightings):
    unexpected_bit_chance = mean_arr
    unexpected_bit_chance[mean_arr>0.5] = 1 - mean_arr[mean_arr>0.5]
    exp_hamming_dist = np.sum(weightings * unexpected_bit_chance)
    return exp_hamming_dist



###### RUN FUNCTIONS ######

def read_data_from_files():
    ## load training datafiles into numpy array from files
    dir = "data/train/mb1/"; mb1trdata = read_data(["{0}{1}".format(dir, x) for x in os.listdir(dir)])
    dir = "data/train/mb2/"; mb2trdata = read_data(["{0}{1}".format(dir, x) for x in os.listdir(dir)])
    dir = "data/train/mb3/"; mb3trdata = read_data(["{0}{1}".format(dir, x) for x in os.listdir(dir)])
    dir = "data/train/mb4/"; mb4trdata = read_data(["{0}{1}".format(dir, x) for x in os.listdir(dir)])
    dir = "data/train/mb5/"; mb5trdata = read_data(["{0}{1}".format(dir, x) for x in os.listdir(dir)])
    save([mb1trdata, mb2trdata, mb3trdata, mb4trdata, mb5trdata], "temp/mbtrdata.npy")

    ## load testing datafiles into numpy array from files
    dir = "data/test/mb1/"; mb1tedata = read_data(["{0}{1}".format(dir, x) for x in os.listdir(dir)])
    dir = "data/test/mb2/"; mb2tedata = read_data(["{0}{1}".format(dir, x) for x in os.listdir(dir)])
    dir = "data/test/mb3/"; mb3tedata = read_data(["{0}{1}".format(dir, x) for x in os.listdir(dir)])
    dir = "data/test/mb4/"; mb4tedata = read_data(["{0}{1}".format(dir, x) for x in os.listdir(dir)])
    dir = "data/test/mb5/"; mb5tedata = read_data(["{0}{1}".format(dir, x) for x in os.listdir(dir)])
    save([mb1tedata, mb2tedata, mb3tedata, mb4tedata, mb5tedata], "temp/mbtedata.npy")

    print("data loaded in {0}s".format(round(time.time() - start_time, 2)))


## gets both the expected values, and the rounded expected values for every microbit
def get_expected_values(training_data):
    exp_val = []
    for mb in training_data:
        exp_val.append(np.sum(mb.astype(float), axis=0) / len(mb[:, 0]))
    rounded_exp_val = np.round(exp_val)
    return np.array(exp_val), np.array(rounded_exp_val)

## gets the weightings for every microbit
def get_weightings(exp_val):
    return np.abs(0.5 - exp_val) * 2

def standard_deviation(mb_dist, mb_expected_dist):
    return np.sqrt(np.sum((mb_dist - mb_expected_dist)**2) / len(mb_dist)-1)

def get_weighted_dist_for_all(tr_data, te_data, mask):
    exp_val, round_exp_val = get_expected_values(tr_data)
    weightings = get_weightings(exp_val)
    dists = []
    for i in range(np.array(te_data).shape[0]):
        temp = []
        for data in te_data:
            temp.append(get_weighted_hamming_distances(data[:, mask], round_exp_val[i, mask], weightings[i, mask]))
        dists.append(temp)
    dists_percent = 100 * np.array(dists) / len(tr_data[0][:, 0])
    return dists, dists_percent

def do_plots(dists):
    dists = np.array(dists)
    for i in range(dists.shape[0]):
        legend = []
        for j in range(dists.shape[1]):
            sns.kdeplot(dists[i][j])
            legend.append("microbit {0}".format(j))
        plt.legend(legend)
        plt.savefig("figs/fig{0}.png".format(i))
        plt.clf()

def plot_heatmap(exp_val, mask):
    perc = np.abs(exp_val - 0.5) * 200
    perc[:, ~mask] = None
    sns.heatmap(perc, cmap="viridis")
    plt.savefig("figs/heatmap.png")
    plt.clf()



## load data
# read_data_from_files()
[mb1trdata, mb2trdata, mb3trdata, mb4trdata, mb5trdata], [mb1tedata, mb2tedata, mb3tedata, mb4tedata, mb5tedata]= load("temp/mbtrdata.npy"), load("temp/mbtedata.npy") # read data from pickle


## Hamming distance from expected value, only looking at the volatile bits
volatile_bits_mask = ~load("temp/global_const_bits.npy").reshape(-1) # mask of bits that change between microbits

tr_data = [mb1trdata, mb2trdata, mb3trdata, mb4trdata, mb5trdata]
te_data = [mb1tedata, mb2tedata, mb3tedata, mb4tedata, mb5tedata]
exp_val, round_exp_val = get_expected_values(tr_data)
weightings = get_weightings(exp_val)

mb1_dists = get_weighted_hamming_distances(mb1tedata[:, volatile_bits_mask], round_exp_val[0, volatile_bits_mask], weightings[0, volatile_bits_mask])
dists, dists_percent = get_weighted_dist_for_all(tr_data, te_data, volatile_bits_mask)
do_plots(dists)
plot_heatmap(exp_val, volatile_bits_mask)

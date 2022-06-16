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


def PUF_train(trdata):
    ## work out the expected values & weightings for each bit on microbit 1
    expected_values = np.sum(trdata.astype(np.float), axis=0) / len(trdata[:, 0])
    rounded_expec_vals = np.round(expected_values)
    weightings = np.abs(0.5 - expected_values)*2 # x2 so weightings scale linearly between 0 & 1 (not necessary)

    ## work out what the expected hamming distance is
    exp_hamming_dist = get_expected_hamming_distance(expected_values, weightings)

    ## work out the expected std for microbit 1
    tr_dists = get_weighted_hamming_distances(trdata, rounded_expec_vals, weightings)
    expec_std = np.sqrt(np.sum((tr_dists - exp_hamming_dist)**2) / len(tr_dists)-1)
    Z = 3.4  # to give 99.97% certainty
    x = Z * expec_std + exp_hamming_dist

    return rounded_expec_vals, weightings, x


def plot_distros(arrs, length=0):
    for arr in arrs:
        if length > 0:
            sns.kdeplot(100 * arr / length)
        else:
            sns.kdeplot(arr)
    plt.title("distribution of hamming distances between the expected values for mb1\nand all the microbits")
    plt.xlabel("hamming distance (%)" if length > 0 else "hamming distance (total)")
    plt.show()

if __name__ == "__main__":
    ## load data
    # read_data_from_files()
    [mb1trdata, mb2trdata, mb3trdata, mb4trdata, mb5trdata], [mb1tedata, mb2tedata, mb3tedata, mb4tedata, mb5tedata]= load("temp/mbtrdata.npy"), load("temp/mbtedata.npy") # read data from pickle

    # mask of all bits that do change between microbits
    volatile_bits_mask = ~load("temp/global_const_bits.npy").reshape(-1)

    # train the detector on a microbit
    rounded_exp_values_mb1, mb1_weightings, x = PUF_train(mb1trdata[:, volatile_bits_mask])

    # test a microbit to see if it is the same as the detected one
    mb1_dists = get_weighted_hamming_distances(mb1tedata[:, volatile_bits_mask], rounded_exp_values_mb1, mb1_weightings)
    print("mb1 {0} the microbit your are looking for".format("is" if np.mean(mb1_dists)<= x else "is not"))

    mb2_dists = get_weighted_hamming_distances(mb2tedata[:, volatile_bits_mask], rounded_exp_values_mb1, mb1_weightings)
    print("mb2 {0} the microbit your are looking for".format("is" if np.mean(mb2_dists)<= x else "is not"))

    mb3_dists = get_weighted_hamming_distances(mb3tedata[:, volatile_bits_mask], rounded_exp_values_mb1, mb1_weightings)
    print("mb3 {0} the microbit your are looking for".format("is" if np.mean(mb3_dists)<= x else "is not"))

    mb4_dists = get_weighted_hamming_distances(mb4tedata[:, volatile_bits_mask], rounded_exp_values_mb1, mb1_weightings)
    print("mb4 {0} the microbit your are looking for".format("is" if np.mean(mb4_dists)<= x else "is not"))

    mb5_dists = get_weighted_hamming_distances(mb5tedata[:, volatile_bits_mask], rounded_exp_values_mb1, mb1_weightings)
    print("mb5 {0} the microbit your are looking for".format("is" if np.mean(mb5_dists)<= x else "is not"))

    # plot the weighted hamming distances between the test microbit and expected values
    plot_distros([mb1_dists, mb2_dists, mb3_dists, mb4_dists, mb5_dists], mb2tedata.shape[1])

    print("Time Elapsed: {0}s".format(round(time.time()-start_time, 2)))

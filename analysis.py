# Will Bourne 14/06/2022
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
        :return: mask of which bits are always constant across all microbits,
                 array of masks of which bits are always constant across each microbits"""
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


def get_frac_hamming_distances(mbarr, exp_vals):
    frac_hamming_dist = []
    for mbdataset in mbarr:
        diffs = np.abs(mbdataset - exp_vals)
        frac_hamming_dist.append(np.sum(diffs))
    return np.array(frac_hamming_dist, dtype=float)


# Louis functions
def calculate_mean_bit_values(mbdata):
    mean_bit_values = []
    for mb in mbdata:
        mean_bit_values.append(np.sum(mb[:, :], axis = 0) / mb[:, :].shape[0])
    return np.array(mean_bit_values)

def calculate_inter_mean_bit_values(mean_bit_values):
    return np.sum(mean_bit_values, axis=0) / mean_bit_values.shape[0]

def low_intra_volatility_test(mean_bit_values):
    mask = [[], [], []]
    for i in range(mean_bit_values.shape[1]):
        for j in range(mean_bit_values.shape[0]):
            if mean_bit_values[j, i] < 0.2 or mean_bit_values[j, i] > 0.8:
                mask[j].append(True)
            else:
                mask[j].append(False)
    mask = np.array(mask)
    final_mask = np.empty(mask.shape[1], dtype=bool)
    final_mask[:] = True
    for i in range(mask.shape[1]):
        for j in range(mask.shape[0]):
            if mask[j, i] == False:
                final_mask[i] = False    
    return final_mask     

def high_inter_volatility_test(intra_mask, mean_bit_values_inter):
    mask = intra_mask
    for i in range(mean_bit_values_inter.shape[0]):
        if mean_bit_values_inter[i] < 0.3 or mean_bit_values_inter[i] > 0.7:
            mask[i] = False
    return mask 

def calculate_mask(mbdata):
    mean_bit_values = calculate_mean_bit_values(mbdata)
    mean_bit_values_inter = calculate_inter_mean_bit_values(mean_bit_values)
    intra_mask = low_intra_volatility_test(mean_bit_values)
    mask = high_inter_volatility_test(intra_mask, mean_bit_values_inter)
    return mask


## load datafiles into numpy array from files
# dir = "data/mb1/"; mb1data = read_data(["{0}{1}".format(dir, x) for x in os.listdir(dir)])
# dir = "data/mb2/"; mb2data = read_data(["{0}{1}".format(dir, x) for x in os.listdir(dir)])
# dir = "data/mb3/"; mb3data = read_data(["{0}{1}".format(dir, x) for x in os.listdir(dir)])
# dir = "data/mb4/"; mb4data = read_data(["{0}{1}".format(dir, x) for x in os.listdir(dir)])
# dir = "data/mb5/"; mb5data = read_data(["{0}{1}".format(dir, x) for x in os.listdir(dir)])
# save([mb1data, mb2data, mb3data, mb4data, mb5data], "temp/mbdata.npy")

## load datafiles into numpy array from pickle
[mb1data, mb2data, mb3data, mb4data, mb5data] = load("temp/mbdata.npy")
print("data loaded in {0}s".format(round(time.time()-start_time, 2)))


## get hamming distances between 2 specifc RAM datasets of mb2
# ham_dist = hamming_distance(mb2data[0,:], mb2data[1,:])
# print("Hamming count: {0}, Hamming fraction: {1}% difference".format(ham_dist, round(100*ham_dist/len(data[0]), 2)))


## get hamming distances between every possible RAM data pair for mb2
# hamming_distances, combinations = hamming_distance_combinations(mb2data)
# plt.hist(hamming_distances, bins=50)
# sns.kdeplot(100*np.array(hamming_distances)/len(data[0]))
# plt.savefig("figs/intra_hamming_distances_mb2.pdf")


## make a mask to find which bits never change across all microbits / each microbit
## if all bits were random, there is a (1/2^29)*1048576 % chance that any constant bits are returned (very small!)
# global_const_bits, mb_const_bits = inter_intra_mb_constant_bits([mb1data, mb2data, mb3data, mb4data])
# print("constant bits in: mb1={0}, mb2={1}, mb3={2}, mb4={3}, globally={4}".format(*np.sum(mb_const_bits, axis=1),
#                                                                                   np.sum(global_const_bits)))


## Hamming distance from expected value, only looking at the volatile bits


volatile_bits_mask = ~load("temp/global_const_bits.npy").reshape(-1) # mask of bits that change between microbits

## work out the expected values for each bit on microbit 1
expected_values_mb1 = np.sum(mb1data.astype(np.float), axis=0) / len(mb1data[:, 0])

## find the difference between the volatile bits of the expected values and a correct trial dataset
mb1_dists = get_frac_hamming_distances(mb1data[:, volatile_bits_mask], expected_values_mb1[volatile_bits_mask])

## find the difference between the volatile bits of the expected values and incorrect trial datasets
mb2_dists = get_frac_hamming_distances(mb2data[:, volatile_bits_mask], expected_values_mb1[volatile_bits_mask])
mb3_dists = get_frac_hamming_distances(mb3data[:, volatile_bits_mask], expected_values_mb1[volatile_bits_mask])
mb4_dists = get_frac_hamming_distances(mb4data[:, volatile_bits_mask], expected_values_mb1[volatile_bits_mask])

##  plot the hamming distances of all the microbits
sns.kdeplot(100 * mb1_dists/len(mb1_dists))
sns.kdeplot(100 * mb2_dists/len(mb1_dists))
sns.kdeplot(100 * mb3_dists/len(mb1_dists))
sns.kdeplot(100 * mb4_dists/len(mb1_dists))
plt.legend(["microbit 1", "microbit 2","microbit 3","microbit 4"])
plt.title("distribution of hamming distances between the expected values for mb1\nand all the microbits")
plt.xlabel("hamming distance (%)")
plt.show()


# could extend to exclude bits that are particuarly volative?

##  How to improve the filtering mask:
#   take only the bits that have a low chance of changing intra, and a high chance of changing inter
#   how?
#   can sum the ith bit in the sequence and divide by how many sequences, close to 50 -> high chance of changing

# here's implementation but not very fast, also not sure if correct method
# the range of values accepted can be increased / decreased -> this has effect on the uniqueness of the sample if we accept too few values
# it's important after making the mask that we test it on different microboards than the ones it was trained on, maybe get data from 4 more and use that for testing?
mask = calculate_mask([mb1data, mb2data, mb3data])

expected_values_mb4 = np.sum(mb4data.astype(np.float), axis=0) / len(mb4data[:, 0])

mb4_dists = get_frac_hamming_distances(mb4data[:, mask], expected_values_mb4[mask])
mb5_dists = get_frac_hamming_distances(mb5data[:, mask], expected_values_mb4[mask])
print(mb4_dists)
print(mb5_dists)

# print(np.array(final_mask)[50000:51000])
# print(mean_bit_values[:,final_mask].shape)

print("Time Elapsed: {0}s".format(round(time.time()-start_time, 2)))

# Will Bourne 14/06/2022
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import itertools
import time

start_time = time.time()
def flatten(xss):
    return [x for xs in xss for x in xs]


def save(data, file):
    with open(file, 'wb') as f:
        np.save(f, data)

def load(file):
    with open(file, 'wb') as f:
        return np.loaf(f)


def read_data(filenames):
    byte_files = np.array([], dtype=bool)
    for file_link in data_files:
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
        hamming_distances.append(hamming_distance(data[combination[0],:], data[combination[1],:]))

    return hamming_distances, combinations


def find_constant_bits(arr):
    return np.sum(arr, axis=0) / len(arr[:,0])


## load datafiles into numpy array from files
data_files = ["data/mb3/data1 -29.bin",
              "data/mb3/data2 -29.bin",
              "data/mb3/data3 -29.bin",
              "data/mb3/data4 -29.bin",
              "data/mb3/data5 -29.bin",
              "data/mb3/data6-29.bin",
              "data/mb3/data7-29.bin",
              "data/mb3/data8-29.bin",
              "data/mb3/data9-29.bin",
              "data/mb3/data10-29.bin"
              ]
data = read_data(data_files)


## get hamming distances between 2 specifc RAM datasets
# ham_dist = hamming_distance(data[0,:], data[1,:])
# print("Hamming count: {0}, Hamming fraction: {1}% difference".format(ham_dist, round(100*ham_dist/len(data[0]), 2)))


## get hamming distances between every possible RAM data pair
# hamming_distances, combinations = hamming_distance_combinations(data)
# plt.hist(hamming_distances, bins=50)
# plt.gca().set(title='Frequency Histogram', ylabel='Frequency');
# plt.savefig("figs/intra_hamming_distances.pdf")

## work out which bits change & which don't
print(find_constant_bits(data[:,:50]))







print("Time Elapsed: {0}s".format(round(time.time()-start_time, 2)))









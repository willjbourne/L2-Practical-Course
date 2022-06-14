import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import itertools

"""
# data file locations
data_files = ["data/mb1/datafile-b'26.25'-1.txt",
              "data/mb1/datafile-b'26.25'-2.txt",
              "data/mb1/datafile-b'26.00'-3.txt",
              "data/mb1/datafile-b'25.00'-4.txt"]

# read bytes from files
byte_files, number_files = [], []
for file_link in data_files:
    with open(file_link, "rb") as file:
        byte_files.append(file.read())

        # represent each byte with a base-10 number between 0-255
        number_files.append(np.array(list(byte_files[-1])))
"""

# TEST DATA
data = np.array([[True, False, True, False, True, False],
          [True, True, False, False, True, False],
          [True, True, False, False, True, False]])


def flatten(xss):
    return [x for xs in xss for x in xs]


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


def plot_kde(number_files):
    # plot the frequency density of all numbers (1 byte, 0-255) stored in the SRAM
    sns.set_style('whitegrid')
    for numbers in number_files:
        sns.kdeplot(numbers)
        print(len(numbers))
    plt.legend(['1', '2', '3'])
    plt.savefig("figs/sram-frequency-density.pdf")

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
        hamming_distances.append(hamming_distance(combination[0], combination[1]))
    return hamming_distances, combinations

# ------

data_files = ["data/mb3/data1 -28.bin",
              "data/mb3/data2 -28.bin",
              ]

data = read_data(data_files)

# print(data)

# ham_dist = hamming_distance(data[0,:], data[1,:])
# print("Hamming count: {0}, Hamming fraction: {1}%".format(ham_dist, round(100*ham_dist/len(data[0]), 2)))


print(hamming_distance_combinations(data))








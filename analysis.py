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
    itertools.combinations('ABCD', 2)


print(hamming_distance(data[0,:], data[1,:]))








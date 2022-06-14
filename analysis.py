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
    with open(file, 'wb') as f:
        return np.loaf(f)


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
        hamming_distances.append(hamming_distance(data[combination[0],:], data[combination[1],:]))

    return hamming_distances, combinations


def find_constant_bits(arr):
    chance_bit_is_a_one = np.sum(arr.astype(np.float), axis=0) / len(arr[:,0])
    bits_that_dont_change = [(chance_bit_is_a_one == 1) | (chance_bit_is_a_one == 0)]
    return bits_that_dont_change


## load datafiles into numpy array from files
dir = "test_data/mb2/"
mb2data = read_data(["{0}{1}".format(dir, x) for x in os.listdir(dir)])

dir = "test_data/mb3/"
mb3data = read_data(["{0}{1}".format(dir, x) for x in os.listdir(dir)])

dir = "test_data/mb4/"
mb4data = read_data(["{0}{1}".format(dir, x) for x in os.listdir(dir)])



## get hamming distances between 2 specifc RAM datasets
# ham_dist = hamming_distance(mb2data[0,:], mb2data[1,:])
# print("Hamming count: {0}, Hamming fraction: {1}% difference".format(ham_dist, round(100*ham_dist/len(data[0]), 2)))


## get hamming distances between every possible RAM data pair
# hamming_distances, combinations = hamming_distance_combinations(mb2data)
# plt.hist(hamming_distances, bins=50)
# sns.kdeplot(100*np.array(hamming_distances)/len(data[0]))
# plt.savefig("figs/intra_hamming_distances_mb2.pdf")


## work out which bits change & which don't
# print(find_constant_bits(data[:,5000:5500]))

# Which bits remain constant in microbit 2?
mb2_const_bits = find_constant_bits(mb2data)
save(mb2_const_bits, "temp/inter_mb_const_bits.npy")

# Which bits remain constant in microbit 3?
mb3_const_bits = find_constant_bits(mb3data)
save(mb3_const_bits, "temp/inter_mb_const_bits.npy")

# Which bits remain constant in microbit 4?
mb4_const_bits = find_constant_bits(mb4data)
save(mb4_const_bits, "temp/inter_mb_const_bits.npy")






# of the bits that don't change in each individual microbit, which remain constant across all the microbits?
# inter_mb_array = np.array([mb2_const_bits, mb3_const_bits, mb4_const_bits])
# inter_mb_const_bits = find_constant_bits(inter_mb_array)

# inter_mb_const_bits = mb2_const_bits & mb3_const_bits & mb4_const_bits

inter_mb_const_bits = np.bitwise_and(np.bitwise_and(mb2_const_bits, mb3_const_bits), mb4_const_bits)

save(inter_mb_const_bits, "temp/inter_mb_const_bits.npy")


print("constant bits in mb1:", np.sum(mb2_const_bits))
print("constant bits in mb2:", np.sum(mb3_const_bits))
print("constant bits in mb3:", np.sum(mb4_const_bits))
print("constant bits across microbits:2, 3, 4: ", np.sum(inter_mb_const_bits))



print("Time Elapsed: {0}s".format(round(time.time()-start_time, 2)))

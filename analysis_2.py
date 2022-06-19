# Will Bourne 14/06/2022
# Louis Barnes 15/06/2022
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import itertools
import time
import os
import matplotlib as mpl
from functools import reduce
from scipy.stats import norm

start_time = time.time()


def factors(n):
    return np.sort(list(reduce(list.__add__,
                               ([i, n // i] for i in range(1, int(n ** 0.5) + 1) if n % i == 0))))

def flatten(xss):
    return [x for xs in xss for x in xs]

def save(data, file):
    with open(file, 'wb') as f:
        np.save(f, data)

def load(file):
    with open(file, 'rb') as f:
        return np.load(f,allow_pickle=True)

def listdirs(rootdir):
    dirs = []
    for file in os.listdir(rootdir):
        d = os.path.join(rootdir, file)
        if os.path.isdir(d): dirs.append(d)
    return dirs


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


def plot_hamming_dist_choices():
    lengths = range(100)
    combos = []
    for l in lengths:
        combos.append(len([list(comb) for comb in itertools.combinations(range(l), 2)]))
    plt.plot(lengths, combos)
    plt.xlabel("num. datasets")
    plt.ylabel("num. hamming distances")
    plt.savefig("figs/exponential hamming distances.pdf")


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


def read_full_data_from_files():
    mb_datas = []
    for mb_link in listdirs("data/full"):
        mb_datas.append(read_data(["{0}/{1}".format(mb_link, x) for x in os.listdir(mb_link) if not x.startswith('.')]))
    save(mb_datas, "temp/mbfull_data.npy")


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


def density_plots(arrs, length=0, fname="figs/density_plots.png", i=1):
    for arr in arrs:
        if length > 0:
            sns.kdeplot(100 * arr / length)
        else:
            sns.kdeplot(arr)
    plt.title("distribution of hamming distances between the expected values for mb{0}\nand all the microbits".format(i))
    plt.xlabel("hamming distance (%)" if length > 0 else "hamming distance (total)")
    plt.savefig(fname)
    plt.clf()


def inter_chip_ham_dists(arrs, length=0, fname="figs/inter_chip_hamming_plots.png", i=1):
    hamming_dists = []
    for arr in arrs:
        if length > 0:
            hamming_dists.append(np.mean(100 * arr / length))
        else:
            hamming_dists.append(np.sum(arr))
    del hamming_dists[i - 1]
    # get normal distribution
    mu, std = norm.fit(hamming_dists)

    # Plot the histogram.
    plt.hist(hamming_dists, density=True, alpha=0.6, color='orange')
    sns.kdeplot(hamming_dists)
    # Plot the PDF.
    xmin, xmax = plt.xlim()
    x = np.linspace(xmin, xmax, 100)
    p = norm.pdf(x, mu, std)

    plt.plot(x, p, 'k', linewidth=2)
    title = "Fit Values: {:.2f} and {:.2f}".format(mu, std)
    plt.title(title)
    # plt.title("distribution of hamming distances between the expected values for mb{0}\nand the other microbits".format(i))
    plt.xlabel("inter-chip hamming distance (%)" if length > 0 else "hamming distance (total)")
    plt.savefig(fname)
    plt.clf()


def inter_intra_chip_ham_dists(inter_arrs, length=0, fname="figs/inter_chip_hamming_plots.png", i=1):
    fig, (ax1, ax2) = plt.subplots(2, 1)
    plt.xlabel("inter-chip hamming distance (%)")

    intra_dists = 100 * inter_arrs[i-1] / length
    hamming_dists = []
    for arr in inter_arrs:
        if length > 0:
            hamming_dists.append(np.mean(100 * arr / length))
        else:
            hamming_dists.append(np.sum(arr))
    del hamming_dists[i - 1]

    # normal distributions
    intra_mu, intra_std = norm.fit(intra_dists)
    inter_mu, inter_std = norm.fit(hamming_dists)

    # Plot the histograms



    ax1.hist(intra_dists, density=True, alpha=0.6, color='red')
    ax2.hist(hamming_dists, density=True, alpha=0.6, color='red')

    # Plot KDE's
    sns.kdeplot(intra_dists, color='blue', ax=ax1)
    sns.kdeplot(hamming_dists, color='blue', ax=ax2)


    # Plot the PDF.
    xmin, xmax = plt.xlim()
    x1 = np.linspace(*ax1.get_xlim(), 100)
    x2 = np.linspace(*ax2.get_xlim(), 100)
    ax1.plot(x1, norm.pdf(x1, intra_mu, intra_std), 'k', linewidth=2, color='green')
    ax2.plot(x2, norm.pdf(x2, inter_mu, inter_std), 'k', linewidth=2, color='green')

    title = "Fit Values: {:.2f} and {:.2f}".format(inter_mu, inter_std)
    plt.title(title)
    # plt.title("distribution of hamming distances between the expected values for mb{0}\nand the other microbits".format(i))

    ax1.set_title('intra-chip hamming distance')
    ax2.set_title('inter-chip hamming distance')

    plt.tight_layout()

    plt.savefig(fname)
    plt.clf()


def chip_topology_plots(mbdata, dir="figs/chip_layouts", rl=0):
    mb_exp_bit_vals = np.sum(mbdata.astype(np.float), axis=0).reshape(1, -1) / len(mbdata[:, 0])
    # mb_exp_bit_vals[:,~volatile_bits_mask] = None
    total_bits = mb_exp_bit_vals.shape[1]
    factor_list = factors(total_bits)
    print("total bits:", total_bits, "\t num of factors: ", len(factor_list))

    ## Plots
    cdict1 = {'red': ((0.0, 0.0, 0.0),
                      (0.5, 0.0, 0.1),
                      (1.0, 1.0, 1.0)),

              'green': ((0.0, 0.0, 0.0),
                        (1.0, 0.0, 0.0)),

              'blue': ((0.0, 0.0, 1.0),
                       (0.5, 0.1, 0.0),
                       (1.0, 0.0, 0.0))
              }
    blue_red1 = mpl.colors.LinearSegmentedColormap('BlueRed1', cdict1)
    plt.rcParams['axes.facecolor'] = 'white'
    if rl > 0: factor_list = [rl]
    for factor in factor_list[:]:
        image = mb_exp_bit_vals.reshape(factor, -1)
        fig, ax = plt.subplots()
        im = plt.imshow(image, origin='upper',
                        aspect=image.shape[1]/image.shape[0],
                        cmap=plt.get_cmap(blue_red1),
                        interpolation='nearest',
                        vmin=0, vmax=1)
        fig.colorbar(im)
        if rl == 0: plt.savefig("{1}/{0}-rows.png".format(factor, dir))
        else: plt.savefig("{1}-mb{0}-rows.png".format(factor, dir))


def chip_fingerprints(mbdatas):
    fingerprint_arr = []
    for mbdata in mbdatas:
        mb_exp_bit_vals = np.sum(mbdata.astype(np.float), axis=0).reshape(1, -1) / len(mbdata[:, 0])
        fingerprint_arr.append(mb_exp_bit_vals)
    fingerprint_arr = np.array(fingerprint_arr).reshape(len(mbdatas), -1)

    fingerprint_arr[:,~volatile_bits_mask] = None

    ## Plots
    cdict1 = {'red': ((0.0, 0.0, 0.0),
                      (0.5, 0.0, 0.1),
                      (1.0, 1.0, 1.0)),

              'green': ((0.0, 0.0, 0.0),
                        (1.0, 0.0, 0.0)),

              'blue': ((0.0, 0.0, 1.0),
                       (0.5, 0.1, 0.0),
                       (1.0, 0.0, 0.0))
              }
    blue_red1 = mpl.colors.LinearSegmentedColormap('BlueRed1', cdict1)
    plt.rcParams['axes.facecolor'] = 'white'
    fig, ax = plt.subplots()
    im = plt.imshow(fingerprint_arr, origin='upper',
                    aspect=fingerprint_arr.shape[1]/len(mbdatas),
                    cmap=plt.get_cmap(blue_red1),
                    interpolation='nearest',
                    vmin=0, vmax=1)
    fig.colorbar(im)
    plt.savefig("figs/fingerprints.png")

if __name__ == "__main__":
    ## load data
    # read_data_from_files()
    # read_full_data_from_files()
    # read_full_data_from_files()
    # print("data saved to pickle..")

    # [mb1trdata, mb2trdata, mb3trdata, mb4trdata, mb5trdata]= load("temp/mbtrdata.npy") # read data from pickle
    # [mb1tedata, mb2tedata, mb3tedata, mb4tedata, mb5tedata] = load("temp/mbtedata.npy")

    all_mb_data = load("temp/mbfull_data.npy")

    # mask of all bits that do change between microbits
    volatile_bits_mask = ~load("temp/global_const_bits.npy").reshape(-1)

    all_intra_hamming_distances = []
    inter_mb_datasets = []
    for mb in all_mb_data:

        intra_hamming_distances, intra_combinations = hamming_distance_combinations(mb)
        all_intra_hamming_distances.extend(intra_hamming_distances)

        inter_mb_datasets.append(mb[0,:])

    all_inter_hamming_distances, inter_combinations = hamming_distance_combinations(np.array(inter_mb_datasets))


    plt.hist(np.sort(all_intra_hamming_distances), density=True, color="lightblue")
    plt.hist(np.sort(all_inter_hamming_distances), density=True, color="lightgreen")

    # get normal distribution
    mu1, std1 = norm.fit(all_intra_hamming_distances)
    mu2, std2 = norm.fit(all_inter_hamming_distances)

    # Plot the PDF.
    xmin, xmax = plt.xlim()
    x = np.linspace(xmin, xmax, 100)
    p1 = norm.pdf(x, mu1, std1)
    p2 = norm.pdf(x, mu2, std2)
    plt.plot(x, p1, 'k', linewidth=2, color="blue")
    plt.plot(x, p2, 'k', linewidth=2, color="green")

    plt.xlim([xmin, xmax])
    plt.xlabel("Hamming Distance")
    plt.ylabel("Frequency Density")
    plt.legend(["intra-chip hamming distances", "inter-chip hamming distances"])
    plt.savefig("figs/PUF_viability.pdf")
    print("mean1:", mu1, "std1: ", std1)
    print("mean2:", mu2, "std2: ", std2)




    """
    for i in range(len(all_mb_data)):
        print("mb {0} started..".format(i + 1))

        ## train the detector on a microbit
        # rounded_exp_values_mb1, mb1_weightings, x = PUF_train(all_mb_data[i][:, volatile_bits_mask])

        ## get hamming distances for all the microbits
        # all_mb_dists = []
        # for mbtedata in all_mb_data:
        #     mb_dists = get_weighted_hamming_distances(mbtedata[:, volatile_bits_mask], rounded_exp_values_mb1, mb1_weightings)
        #     all_mb_dists.append(mb_dists)

        ## plot the weighted hamming distances between the test microbit and expected values
        # density_plots(all_mb_dists, np.sum(volatile_bits_mask.astype(int)), "figs/density_plots/base_mb_{0}.png".format(i+1), i+1)
        # inter_intra_chip_ham_dists(all_mb_dists, np.sum(volatile_bits_mask.astype(int)), "figs/inter_hamming_plots/base_mb_{0}.png".format(i+1), i+1)

        ## plot images of expected values for different chip layouts
        # try: os.mkdir("figs/chip_layouts/mb{0}_chip_layouts".format(i+1))
        # except: pass
        # chip_topology_plots(all_mb_data[i], dir="figs/chip_layouts/mb{0}_chip_layouts".format(i+1), rl = 16384)
    """
    ## plot the expected values for each of the microbits in the database linearly
    # chip_fingerprints(all_mb_data)

    print("Time Elapsed: {0}s".format(round(time.time()-start_time, 2)))
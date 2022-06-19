# Will Bourne 16/06/2022
import os
import numpy as np
import time
UPDATE_DATABASE = False

start_time = time.time()


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


def save(data, file):
    with open(file, 'wb') as f:
        np.save(f, data)


def listdirs(rootdir):
    dirs = []
    for file in os.listdir(rootdir):
        d = os.path.join(rootdir, file)
        if os.path.isdir(d): dirs.append(d)
    return dirs


def load_database_from_file():
    mb_datas = []
    for mb_link in listdirs("data/mb_database"):
        mb_datas.append(read_data(["{0}/{1}".format(mb_link, x) for x in os.listdir(mb_link) if not x.startswith('.')]))
    save(mb_datas, "temp/mbdatabase.npy")


def train_database_detector(mbs_database, volatile_bits_mask):
    # train the detector on all the microbits
    training_info_database = []
    for mb in mbs_database:
        [rounded_exp_values_mb1, mb1_weightings, x] = analysis.PUF_train(mb[:, volatile_bits_mask])
        training_info_database.append([rounded_exp_values_mb1, mb1_weightings, x])
    analysis.save(training_info_database, "temp/training_info_database.npy")


def get_expected_hamming_distance(mean_arr, weightings):
    unexpected_bit_chance = mean_arr
    unexpected_bit_chance[mean_arr>0.5] = 1 - mean_arr[mean_arr>0.5]
    exp_hamming_dist = np.sum(weightings * unexpected_bit_chance)
    return exp_hamming_dist


def get_weighted_hamming_distances(mb_arrs, exp_arr, weightings):
    frac_hamming_dists = []
    for mb_arr in mb_arrs:
        different_bits = np.bitwise_xor(mb_arr.astype(bool), exp_arr.astype(bool))
        hamming_dist = np.sum(weightings[different_bits])
        # hamming_dist = np.sum(different_bits.astype(float))

        frac_hamming_dists.append(hamming_dist)
    return np.array(frac_hamming_dists, dtype=float)


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
    Z = 10
    x = Z * expec_std + exp_hamming_dist

    return rounded_expec_vals, weightings, x


# mask of all bits that do change between microbits
volatile_bits_mask = ~load("temp/global_const_bits.npy").reshape(-1)

# load query data from file
query_data = read_data(["{0}/{1}".format("data/mb_query", x) for x in os.listdir("data/mb_query") if not x.startswith('.')])
save(query_data, "temp/mbquery.npy")

# load database data
if UPDATE_DATABASE: load_database_from_file()
mbs_database = load("temp/mbdatabase.npy") # [mb1trdata, mb2trdata, mb3trdata, ...]

# train database detector / load training_info_database
if UPDATE_DATABASE:
    # train the detector on all the microbits
    training_info_database = []
    for mb in mbs_database:
        [rounded_exp_values_mb1, mb1_weightings, x] = PUF_train(mb[:, volatile_bits_mask])
        training_info_database.append([rounded_exp_values_mb1, mb1_weightings, x])
    save(training_info_database, "temp/training_info_database.npy")

mbq = load("temp/mbquery.npy") # read data from pickle
training_info_database = load("temp/training_info_database.npy")

# test a microbit to see if it is the same as the detected one
i, found = 0, False
for tri in training_info_database: # [expected value arrays, weightings, x]
    mbq_dists = get_weighted_hamming_distances(mbq[:, volatile_bits_mask], tri[0], tri[1]) # expected value arrays, weightings
    mb_str = [f for f in os.listdir("data/mb_database") if not f.startswith('.')][i]

    # print("testing microbit {0}\n"
    #       "hamming distance(s): {1}\n"
    #       "x: {2}".format(mb_str, np.round(mbq_dists), round(tri[2])))

    if np.mean(mbq_dists) <= tri[2]: # tri[2]: x
        print("microbit linked to fingerprint: {0}\n"
              "hamming distance(s): {1}\n"
              "x: {2}".format(mb_str, np.round(mbq_dists), round(tri[2])))
        found = True
    i += 1

if not found: print("microbit not recognised")
print("Time Elapsed: {0}s".format(round(time.time()-start_time, 2)))




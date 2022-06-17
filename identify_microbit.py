# Will Bourne 16/06/2022
import analysis_2 as analysis
import os
import numpy as np
import time
UPDATE_DATABASE = False

start_time = time.time()

def listdirs(rootdir):
    dirs = []
    for file in os.listdir(rootdir):
        d = os.path.join(rootdir, file)
        if os.path.isdir(d): dirs.append(d)
    return dirs


def load_database_from_file():
    mb_datas = []
    for mb_link in listdirs("data/mb_database"):
        mb_datas.append(analysis.read_data(["{0}/{1}".format(mb_link, x) for x in os.listdir(mb_link) if not x.startswith('.')]))
    analysis.save(mb_datas, "temp/mbdatabase.npy")


def train_database_detector(mbs_database, volatile_bits_mask):
    # train the detector on all the microbits
    training_info_database = []
    for mb in mbs_database:
        [rounded_exp_values_mb1, mb1_weightings, x] = analysis.PUF_train(mb[:, volatile_bits_mask])
        training_info_database.append([rounded_exp_values_mb1, mb1_weightings, x])
    analysis.save(training_info_database, "temp/training_info_database.npy")


# mask of all bits that do change between microbits
volatile_bits_mask = ~analysis.load("temp/global_const_bits.npy").reshape(-1)

# load query data from file
query_data = analysis.read_data(["{0}/{1}".format("data/mb_query", x) for x in os.listdir("data/mb_query") if not x.startswith('.')])
analysis.save(query_data, "temp/mbquery.npy")

# load database data
if UPDATE_DATABASE: load_database_from_file()
mbs_database = analysis.load("temp/mbdatabase.npy") # [mb1trdata, mb2trdata, mb3trdata, ...]

# train database detector / load training_info_database
if UPDATE_DATABASE:
    # train the detector on all the microbits
    training_info_database = []
    for mb in mbs_database:
        [rounded_exp_values_mb1, mb1_weightings, x] = analysis.PUF_train(mb[:, volatile_bits_mask])
        training_info_database.append([rounded_exp_values_mb1, mb1_weightings, x])
    analysis.save(training_info_database, "temp/training_info_database.npy")

mbq = analysis.load("temp/mbquery.npy") # read data from pickle
training_info_database = analysis.load("temp/training_info_database.npy")

# test a microbit to see if it is the same as the detected one
i, found = 0, False
for tri in training_info_database: # [expected value arrays, weightings, x]
    mbq_dists = analysis.get_weighted_hamming_distances(mbq[:, volatile_bits_mask], tri[0], tri[1]) # expected value arrays, weightings
    if np.mean(mbq_dists) <= tri[2]: # tri[2]: x
        correct_mb_str = [f for f in os.listdir("data/mb_database") if not f.startswith('.')][i]
        analysis.density_plots([mbq_dists], query_data.shape[1])
        print("microbit linked to fingerprint: {0}".format(correct_mb_str))
        found = True
    i += 1

if not found: print("microbit not recognised")
print("Time Elapsed: {0}s".format(round(time.time()-start_time, 2)))




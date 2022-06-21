from analysis_2 import *
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd



## load data
# read_full_data_from_files()
all_mb_data = load("temp/mbfull_data.npy")

# mask of all bits that do change between microbits
volatile_bits_mask = ~load("temp/global_const_bits.npy").reshape(-1)


all_intra_biases = []
inter_mb_datasets = []
for mb in all_mb_data:
    mb = mb[:, volatile_bits_mask]

    intra_exp_value = np.mean(mb, axis=0)
    intra_biases = 2 * np.abs(0.5 - intra_exp_value)

    all_intra_biases.append(intra_biases)

    i = 0
    for mbdata in mb:
        i+= 1
        if i <= 5:
            inter_mb_datasets.append(mbdata)




mean_intra_biases = np.mean(all_intra_biases, axis=0)

inter_mb_datasets = np.array(inter_mb_datasets)

inter_exp_value = np.mean(inter_mb_datasets, axis=0)

inter_bias = 2 * np.abs(0.5 - inter_exp_value)

dist = 0.5 - np.abs(0.5-mean_intra_biases)

dist_from_ideal = np.abs(0.5 - inter_exp_value) + dist

print("data loaded... & plotting started")

plt.scatter(mean_intra_biases, inter_bias, marker="x")

df = pd.DataFrame({"mean intra-chip bit bias":mean_intra_biases,"inter-chip bit bias":inter_bias})
print("dataframe done...")
p = sns.jointplot(data=df,x="mean intra-chip bit bias", y="inter-chip bit bias", kind='kde', color="red")
# plt.show()


# plt.scatter(mean_intra_biases, inter_bias, marker="x", c=dist_from_ideal)
# plt.xlabel("mean intra-chip bit bias")
# plt.ylabel("inter-chip bit bias")
# cbar = plt.colorbar()
# plt.show()
plt.savefig("figs/most_useful_bits_density.png", dpi=300)
print("Finished")

print("Time Elapsed: {0}s".format(round(time.time()-start_time, 2)))
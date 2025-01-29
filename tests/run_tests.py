import h5py
import numpy as np
from pickerxl.pickerxl import Picker
import matplotlib.pyplot as plt
#
if __name__ == "__main__":
    model = Picker()
    #
    fid = h5py.File("example_waveforms.h5", "r")
    data_group = fid["data"]
    example_data = []
    true_p_index = []
    true_s_index = []
    for akey in data_group.keys():
        dataset = data_group[akey]
        example_data.append(dataset[...])
        true_p_index.append(float(dataset.attrs["p_arrival_sample"]))
        true_s_index.append(float(dataset.attrs["s_arrival_sample"]))
    fid.close()
    #
    example_data = np.array(example_data)
    preds = model.predict_probability(example_data)
    p_index, s_index = model.predict_arrivals(example_data)
    print("True P-wave arrival index:", true_p_index)
    print("Predicted P-wave arrival index:", p_index)
    print("True S-wave arrival index:", true_s_index)
    print("Predicted S-wave arrival index:", s_index)
    #
    sampling_rate = 100
    p_prop_index = 1
    s_prop_index = 2
    chan_name_list = ['East', 'North', 'Vertical']
    for isample in range(len(example_data)):
        data_per_station = example_data[isample]
        fig, axs = plt.subplots(4, 1, sharex=True, layout='tight')
        for ichan in range(len(data_per_station)):
            ax = axs[ichan]
            time = np.arange(len(data_per_station[ichan]))/sampling_rate
            ax.plot(time, data_per_station[ichan], 'k-', lw=0.5, label=chan_name_list[ichan])
            #
            ax.axvline(time[int(true_p_index[isample])], c='r', ls='--', lw=0.5, ymin=0.5, ymax=1, label='P True')
            ax.axvline(time[int(true_s_index[isample])], c='b', ls='--', lw=0.5, ymin=0.5, ymax=1, label='S True')
            #
            ax.axvline(time[int(p_index[isample])], c='r', ls='-', lw=0.5, ymin=0, ymax=0.5, label='P Predict')
            ax.axvline(time[int(s_index[isample])], c='b', ls='-', lw=0.5, ymin=0, ymax=0.5, label='S Predict')
            #
            ax.set_xlim(time[0], time[-1])
            ax.set_ylabel('Amp (count)')
            ax.legend(ncols=3, fontsize=6)
        #
        ax = axs[-1]
        pred_per_station = preds[isample].detach().numpy()
        time = np.arange(len(pred_per_station[p_prop_index]))/sampling_rate
        ax.plot(time, pred_per_station[p_prop_index], 'r--', lw=0.5, label='P Predict')
        time = np.arange(len(pred_per_station[s_prop_index]))/sampling_rate
        ax.plot(time, pred_per_station[s_prop_index], 'b--', lw=0.5, label='S Predict')
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Prob.')
        ax.legend()
        #
        fig.savefig(f'example_waveform_{isample}.jpg', dpi=300)
        plt.close()
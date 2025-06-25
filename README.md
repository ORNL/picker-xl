# PickerXL

A large deep learning model to measure arrival times from noisy seismic signals. The model was developed by Chengping Chai, Derek Rose, Scott Stewart, Nathan Martindale, Mark Adams, Lisa Linville, Christopher Stanley, Anibely Torres Polanco and Philip Bingham.


## Introduction

This model is trained on STEAD (Mousavi et al., 2019) for Primary (P) and Secondary (S) wave arrival picking. The model was trained using earthquake data at local distances (0-350 km). The model uses 57s-long three-component seismograms sampled at 100 Hz as input. The output of the model are probability channels corresponding to P wave, S wave, and noise. These probabilities can be used to compute P- and S-wave arrival times.

## Installation

```
pip install pickerxl
```

## Example Usage

```
from pickerxl.pickerxl import Picker
import numpy as np
import h5py
model = Picker()
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
preds = model.predict_probability(example_data)
p_index, s_index = model.predict_arrivals(example_data)
print("True P-wave arrival index:", true_p_index)
print("Predicted P-wave arrival index:", p_index)
print("True S-wave arrival index:", true_s_index)
print("Predicted S-wave arrival index:", s_index)
```

## Test the Package

First, go to the top directory of the package. Then, using the following commands.

```
cd tests
python run_tests.py
```

![example image](images/example_waveform_2.png)


## Known Limitations

* The model may have a less-than-optimal performance for earthquake data outside of a source-receiver distance
range of 10–110 km and a magnitude range of 0–4.5 because of biases in the training data.
* The model may produce false detections when applied to continuous seismic data.
* The model may not perform well for earthquake data at larger distances or for non-earthquake sources.

## Reference

Chengping Chai, Derek Rose, Scott Stewart, Nathan Martindale, Mark Adams, Lisa Linville, Christopher Stanley, Anibely Torres Polanco, Philip Bingham; PickerXL, A Large Deep Learning Model to Measure Arrival Times from Noisy Seismic Signals. Seismological Research Letters 2025; doi: https://doi.org/10.1785/0220240353

## License

GNU GENERAL PUBLIC LICENSE version 3

## Credit

The architecture of the deep learning model was adapted from SeisBench (Woollam et al., 2022).

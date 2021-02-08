# retina_info

* cleaned up code for experiments with high-resolution stimuli (collaboration with CBNU College of Medicine)
* Codes in this repo use [pysta2](https://github.com/ys7yoo/pysta2). Thus, clone this and pysta2 in the same folder like this. 
```
git clone https://github.com/ys7yoo/pysta2.git
git clone https://github.com/ys7yoo/retina_info.git
```

## Other packages
```
pip install jupyter matplotlib scikit-learn scikit-image pandas h5py tqdm
```

# data

* Convert matlab files (*.mat) to npz and csv files using `load_mat.ipynb`
* Then, load the converted data similartly to `load_data.ipynb`

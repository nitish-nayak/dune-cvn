# The DUNE Convolutional Visual Network for neutrino interaction classification

This software is provided as a minimal working example of the DUNE Convolutional Neural Network (CVN) as supplemental material to the article "Neutrino interaction classification with the DUNE Convolutional Visual Network" <insert reference when available>. See the [LICENSE file](LICENSE) for the copyright notice and usage options.

1. [Introduction to the CVN](#intro)
2. [Usage instructions](#usage)
3. [Full contents of this release](#contents)

<a name="intro"></a>
## 1. Introduction to the CVN

The CVN classifies neutrino interaction images from the DUNE far detector. Each event consists of three 500 x 500 pixel images in (wire number, time) parameter space. Twenty such example interactions from the DUNE simulation are provided to demonstrate the usage of the CVN software. Please see the article for a full description of the input and outputs of the CVN algorithm.

<a name="usage"></a>
## 2. Usage instructions

To run the example, simply use:

```
python test.py
```

This will produce a file called results.txt in the ./output directory. This can be compared to [./output/expected_results.txt](output/expected_results.txt) to ensure the code has executed correctly.

There are a number of arguments that can be used to change file paths, etc. The full list of these is given below:

| Parameter                 | Default       | Description   |	
| :------------------------ |:-------------:| :-------------|
| --batch_size 	       |	10          |batch size
| --model         |        'saved_model/model.json'          |JSON model
| --weights         |        'saved_model/weights.h5'          |HDF5 pre-trained model weights
| --dataset         |        'dataset'          |dataset path
| --partition         |        'dataset/partition.p'          |Pickle partition
| --shuffle         |        False          |shuffle partition
| --print_model         |        False          |print model summary
| --output_file         |        'output/results.txt'          |output file

Recommended software versions for use:

```
python==3.6
tensorflow==2.1.0
numpy==1.17.3
pickle==4.0
sklearn==0.22.1
```
<a name="contents"></a>
## 3. Full contents of this release

- **./README.md**
	- The README (this) file.
- **./test.py**
	- Python script that runs the CVN over a sample of 20 events.
- **./dataset/event\<n\>.gz**
	- Twenty (n = 0 to 19) input events from the DUNE simulation in Zlib-compressed array format.
- **./dataset.partition.p**
	- Pickle file containing the event ID numbers and a dictionary linking event number to truth information.
- **./modules/data_generator.py**
	- Python class to load the input files provided in the dataset directory.
- **./modules/dune_cvn.py**
	- Python classes describing the CVN network and its architecture.
- **./modules/opts.py**
	- Code to parse user-configurable options.
- **./output/expected_results.txt**
	- A text file containing the expected results of the inference.
- **./output/results.txt**
	- This file will be produced on running test.py, and should be identical to the expected_results.txt file.
- **./saved_model/model.json**
 	- This is the architecture file describing the CVN.
- **./saved_model/weights.h5**
 	- The internal weights of the CVN.


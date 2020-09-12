## ComPerf

This repository contains code relevant to *ComPerf: Comparative Performance
Analysis via Empirical Modelling* (publication forthcoming), which is related
to our earlier work:

[R. Neill, A. Drebes and A. Pop, "Automated Analysis of
Task-Parallel Execution Behavior Via Artificial Neural Networks," 2018 IEEE
International Parallel and Distributed Processing Symposium Workshops (IPDPSW),
Vancouver, BC, 2018, pp. 647-656, doi:
10.1109/IPDPSW.2018.00105](https://ieeexplore.ieee.org/document/8425472).

ComPerf implements a comparative technique for post-mortem parallel performance
analysis, and aims to automate the identification of dominating features
between different parallel workloads, thereby characterizing their observed
performance variations. To do this, predictive artificial neural networks are
trained to (empirically) capture the complex interactions within profiling data
from parallel executions. Recent advances in artificial neural network
interpretability (namely, [DeepLIFT](https://arxiv.org/abs/1704.02685)) to
determine the variations in program and system features that effect the
greatest impact on the observed performance variations.

### Dependencies

The repository requires Python3 running: 

* Numpy (tested on `1.16.4`)
* [Psutil](https://pypi.org/project/psutil/)
* [TensorFlow](https://www.tensorflow.org/) (tested on `1.12.0-rc2`)
* [Keras](https://keras.io/) (tested on `2.0.0`)
* [DeepLIFT](https://github.com/kundajelab/deeplift) (tested on `0.6.6`)
* [scikit-learn](https://scikit-learn.org/stable/) (tested on `0.19.1`)
* [Hyperopt](https://github.com/hyperopt/hyperopt) (tested on `0.2.3`)

While not a dependency, we used the
[Aftermath](https://www.aftermath-tracing.com/) tracing infrastructure for
fine-grained instrumentation and profiling for parallel executions of
[OpenStream](http://openstream.cs.manchester.ac.uk/) and
[OpenMP](https://www.openmp.org/).

### Usage

ComPerf operates on a working directory that contains a `config.py` file, which
defines the experiment configuration. An example experiment folder and
`config.py` is provided in the `examples/` directory. A single run of ComPerf
(via `ComperfRunner.py`) models and analyses the profiling data specific to a
particular experiment repeat and K-Fold index (with reference to K-Fold Cross
Validation). The experiment directory's results folder will then contain the
results for the requested repeat and data partitioning, which are appended to
as the user runs additional repeats and k-folds.

The profiling data must be supplied as a delimited flat file, with rows that
represent examples and a header that identifies each feature/column. Multiple
configurations can be targeted, supplied in the `config.py` via the
`get_configurations()` function. These configurations (returned as strings)
should help identify the filenames to load for each
configuration. With each configuration executed multiple times to produce
multiple datasets for each configuration, the filename's integer suffix must be
used to identify the repeat. For the example `config.py`, the profiling dataset
for the 5th repeat of the tile size configuration (i,j,k) = (8,32,8) is:

    /home/rneill/workspace/data/matmul_datasets/matmul_tiled_8_32_8_4.csv

Here, `matmul_tiled_` is the end of the dataset prefix, the `8_32_8` string is
the configuration identifier, while `4` is the zero-indexed identifier for the
5th repeat. Finally, `.csv` is the datasets suffix. These can be found in
`config.py`, and the dataset itself has been included in `examples` directory
for reference.

Passing `-h` to the runner provides the usage instructions:

	usage: ComperfRunner.py [-h] -f EXPERIMENT_FOLDER -r EXPERIMENT_REPEAT_INDEX
													-i K_FOLD_IDX [-d LOG_LEVEL] [--reload] [--tee]

	required arguments:
		-f EXPERIMENT_FOLDER, --experiment_folder EXPERIMENT_FOLDER
				Experiment base folder (containing config.py).
		-r EXPERIMENT_REPEAT_INDEX, --experiment_repeat_index EXPERIMENT_REPEAT_INDEX
				Assuming repeating the modelling and analysis multiple
				times, provide the repeat index (to get the correct
				train/val/test partitions).
		-i K_FOLD_IDX, --k_fold_idx K_FOLD_IDX
				For the repeat index, what k-fold index are we using
				in this run?

	optional arguments:
		-h, --help            
				show this help message and exit
		-d LOG_LEVEL, --log_level LOG_LEVEL
				Logging level. Options are:1=INFO, 2=DEBUG, 3=TRACE.
		--reload            
				Reload the dataset, even if a serialised version
				already exists in the experiment folder.
		--tee                 
				Pipe logging messages to stdout as well as the log
				file.



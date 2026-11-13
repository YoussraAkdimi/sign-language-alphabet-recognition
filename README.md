# sign-language-alphabet-recognition
Assignment 3 for the course Key Topics in AI

## Repository

[https://github.com/YoussraAkdimi/sign-language-alphabet-recognition](https://github.com/YoussraAkdimi/sign-language-alphabet-recognition)

## Getting the repository

Open a terminal and run:

```bash
git clone https://github.com/YoussraAkdimi/sign-language-alphabet-recognition.git
cd sign-language-alphabet-recognition
```

Alternatively, on GitHub select **Code → Download ZIP** and extract the downloaded archive.

## Environment setup

Make sure Conda (Miniconda or Anaconda) is installed. Run the following commands from the repository's root directory (the directory containing `environment.yml`).

### Create the environment from `environment.yml`

```bash
conda env create -f environment.yml
```

### Activate the environment

```bash
conda activate ktai_3
```

### Update the environment (optional)

When the `environment.yml` file changes, update an existing environment with:

```bash
conda env update -f environment.yml --prune
```

The environment uses Python 3.12 and installs the packages required by the project, including Jupyter, NumPy, pandas, Matplotlib, scikit-learn, OpenCV, MediaPipe and TensorFlow.

## Running the project

Activate the `ktai_3` environment before running any project code. The repository contains two experiment notebooks and a Python script for webcam-based recognition.

### Run the main notebook

The main notebook compares different feature transformations and evaluates the trained models:

`main_files/test_different_transformations_simple_model.ipynb`

From the repository root, start Jupyter from the `main_files` directory:

```bash
cd main_files
jupyter notebook
```

Alternatively, start JupyterLab:

```bash
jupyter lab
```

Open `test_different_transformations_simple_model.ipynb` in the Jupyter interface. Make sure the notebook is running with the `ktai_3` kernel.

### Run the notebook and obtain the results

Run the notebook cells in order, from top to bottom. You can also select **Kernel → Restart Kernel and Run All Cells**.

The notebook loads the training and validation data, creates several versions of the landmark features, trains a GRU model for each version and evaluates the models on the independent test dataset. It compares the following feature-processing configurations:

- Baseline (no feature transformation)
- Landmark normalization
- Landmark weighting
- Pose-based centering
- Removal of Z coordinates
- Removal of lower-body pose landmarks
- All transformations combined

After the cells have completed, the results are displayed below the relevant cells. These include training and validation curves, confusion matrices, accuracy, balanced accuracy, macro F1 and per-class F1 scores.

### Dataset locations

The notebook expects the following dataset folders in the repository root:

- `mp_data/` — training data
- `mp_data_validation/` — validation data
- `mp_data_new_test/` — independent test data

Each dataset is organized by class and sequence. Every sequence must contain exactly 30 `.npy` files (one feature vector per frame). Keep the class-folder names consistent across all three datasets. The notebook checks that the test classes match the training classes.

The current experiment uses four classes: `idle`, `poes`, `slapen` and `vogel`. It is therefore a four-class gesture-recognition experiment, not a complete alphabet classifier.

### Other experiment notebook (optional)

The repository also contains:

`main_files/test_different_transformations_multi_seed.ipynb`

This notebook is intended for experiments across multiple random seeds. Open it in Jupyter and run the cells in order. It may take longer because model training is repeated.

### Run real-time webcam recognition (optional)

The script `main_files/experiments/experiment.py` records/loads landmark data, trains a model and starts live prediction using a webcam. To run it, return to the repository root and execute:

```bash
cd ..
python main_files/experiments/experiment.py
```

A working webcam and the MediaPipe asset `task_models/holistic_landmarker.task` are required. Review the configuration at the top of the script before running it. Note that `whatIfMode = True` is currently configured, which means the validation capture call does not save newly recorded frames. Press `q` to stop the webcam workflow.

## TensorBoard (optional)

The training code writes TensorBoard logs to the `logs/` directory. From the repository root, run:

```bash
tensorboard --logdir=logs
```

Open the local URL printed in the terminal to inspect the training metrics.

## Troubleshooting

- **Cannot import `modules`:** Start Jupyter from the `main_files/` directory as described above. The notebook derives the project root from its current working directory.
- **Dataset folder not found:** Check that `mp_data/`, `mp_data_validation/` and `mp_data_new_test/` exist in the repository root. The independent test section requires `mp_data_new_test/`.
- **Incorrect number of frames:** Each sequence directory must contain exactly 30 `.npy` files.
- **Conda environment issues:** Make sure Conda is up to date. TensorFlow, MediaPipe and OpenCV compatibility can depend on the operating system and Python package availability.
- **Webcam not detected:** Check camera permissions and make sure another application is not using the camera.

For a detailed description of the project architecture, feature extraction, preprocessing, model, training and evaluation, see [DOCUMENTATION.md](DOCUMENTATION.md).

# Project Setup Guide

This guide provides step-by-step instructions for setting up the project environment using Conda, activating it, and running `app.py`.

## Prerequisites

Ensure you have [Miniconda](https://docs.conda.io/en/latest/miniconda.html) or [Anaconda](https://www.anaconda.com/) installed on your system.

## Install Conda (if not installed)

### Windows:
Download and install Miniconda from [here](https://docs.conda.io/en/latest/miniconda.html). Follow the installation instructions and ensure to select the option to add Conda to the system PATH.

### macOS/Linux:
Use the following command to install Miniconda:

```bash
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh
bash miniconda.sh
```

Follow the installation steps and restart your terminal.

## Recreate Conda Environment

Once Conda is installed, navigate to the project directory and run:

```bash
conda env create -f environment.yml
```

This will create a new Conda environment with the required dependencies.

## Activate the Conda Environment

Activate the newly created environment using:

```bash
conda activate my_env
```

Replace `my_env` with the actual environment name specified in the `environment.yml` file.

## Run the Application

After activating the environment, run the application with:

```bash
python app.py
```

## Deactivating the Environment

To exit the Conda environment, use:

```bash
conda deactivate
```

## Additional Notes
- If you need to update the environment, use:
  ```bash
  conda env update --file environment.yml --prune
  ```
- To remove the environment:
  ```bash
  conda env remove -n my_env
  ```
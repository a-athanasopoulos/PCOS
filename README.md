# Probably Correct Optimal Stable Matching for Two-Sided Market Under Uncertainty

This repository contains the implementation of the algorithms and experiments discussed in the paper titled "Probably Correct Optimal Stable Matching for Two-Sided Market Under Uncertainty." The goal of this work is to develop and analyze algorithms for finding stable matchings in two-sided markets where preferences are uncertain.

## Table of Contents
- [Project Structure](#project-structure)
- [Setup](#setup)
- [Running the Experiments](#running-the-experiments)


## Project Structure
1. `experiment/pac` contains the experiments of the paper.
2. `experiment/pac/generate_instance` generates the two different instances as described in the simulation section.
3. `matching/matching_algo` contains utility matching functions for the case of known preferences, such as the Gale-Shapley algorithm.
4. `matching/centralised/pac` contains our centralised matching algorithms for the paper.
5. `setup/setup.py` contains setup files such as the root directory for the results.

## Setup
1. **Step 1 - Configure file structure**  
   First, create a specific directory for setting up the project:
   ```bash
   mkdir /BASE_PATH/matching_aamas
   mkdir /BASE_PATH/matching_aamas/env
   mkdir /BASE_PATH/matching_aamas/code
   mkdir /BASE_PATH/matching_aamas/workspace
   cd /BASE_PATH/matching_pac
   ```
   Add the code under `/your_path/matching_aamas/code`.  
   The code outputs results to a specific file called `/your_path/matching_aamas/workspace`.  
   
2. **Step 2 - Configure file structure**  
   Set up the directory of your workspace by modifying the file in `setup/setup.py`. You can set `BASE_PATH = /your_path` or modify it according to your wishes.

3. **Step 3 - Install requirements**  
   To run this project, you need the following:
   - Python 3.10
   - Required packages (listed in `requirements.txt`)

   Here is an example of how to properly create the Python environment and install the libraries:
   ```bash
   python3 -m venv /BASE_PATH/matching_aamas/env/matching_env
   source /BASE_PATH/matching_aamas/env/matching_env/bin/activate
   pip install -r /BASE_PATH/matching_aamas/code/matching_bandits/requirements.txt
   ```

## Running the Experiments
1. **Generate instances**  
   First, generate the instances by running:
   ```bash
   python -u -m experiment.pac.generate_instance.generate_instance
   ```
2. **Run simulations** 
   Then we can run the simulation to reproduce the results of the paper.
   ```bash
   python -u -m experiment.pac.run_all
   ```
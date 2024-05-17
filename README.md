# NYC-Rideshare: EDA On High Volume for Hire Vehicles (hvfhv) in July 2022 in New York City. 

This repository contains the Python script for performing Exploratory Data Analysis (EDA) on a dataset of taxi trips. The script is named `main.py`.

## Table of Contents

- [Introduction](#introduction)
- [Requirements](#requirements)
- [Installation](#installation)
- [Usage](#usage)
- [Data](#data)

## Introduction

Exploratory Data Analysis (EDA) is a crucial step in the data science process. It helps to summarize the main characteristics of a dataset, often with visual methods. This script provides a comprehensive EDA process focusing on taxi trip data.

The analysis is conducted in two parts:
1. Exploring the relationship between pre-trip variables and the tip amount (`tips`).
2. Analyzing whether riders tip (`tip_flag`), where `tip_flag` is 1 if the tip amount is greater than 0 and 0 otherwise.

## Requirements

- Python 3.x
- Pandas
- NumPy
- Scikit-learn
- Scipy
- Matplotlib
- Seaborn

## Installation

1. Clone the repository:

    ```bash
    git clone https://github.com/lohzisan/NYC-Rideshare/
    ```

2. Navigate to the project directory:

    ```bash
    cd NYC-Rideshare
    ```

3. Install the required packages:

    ```bash
    pip/pip3 install requirements.txt
    ```

## Usage

Run the `main.py` script with your dataset:

```bash
python main.py
```

## Data

The parquet files used are too big to be uploaded to GitHub. Please navigate to the data folder in the main branch
and in the `data.txt` file, there should be a direct link to the file used. 



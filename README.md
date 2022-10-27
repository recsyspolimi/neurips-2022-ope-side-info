# Off-Policy Evaluation with Deficient Support Using Side Information

This repository contains the code for our paper "Off-Policy Evaluation with Deficient Support Using Side Information", accepted at NeurIPS 2022.
The pre-print of the paper is available [here](paper/NeurIPS_Deficient_Support_Side_Information.pdf).

## Requirements and Setup

The experiments were made with the following configuration:
* **Ubuntu**: version 20.04.
* **Python**: version 3.8. 
* **Pip**: version 22.1.1.
* **RAM**: for the pre-processing phase, we rely on the [pyIEOE](https://github.com/sony/pyIEOE) package. Due to the dataset size, the pre-processing phase may take up to 40 GB of RAM.

The code is mainly built upon the [Open Bandit Pipeline](https://github.com/st-tech/zr-obp) package and the [pyIEOE](https://github.com/sony/pyIEOE) package.


First of all, download the Open Bandit Dataset at this link: https://research.zozo.com/data.html .
Then, create a directory called "logs" (if not already present), and unzip the dataset inside the "logs" directory.
The final structure should look like this:

logs  <br />
└── open_bandit_dataset

Now, we create our virtual environment.

```shell script
sudo apt update
sudo apt install python3.8-venv
python3 -m venv virtualenv
```

To activate the virtual environment, run:
```shell script
source virtualenv/bin/activate
```

After the activation, we can install the requirements:

```shell script
pip install --upgrade pip
pip install --upgrade "jax[cpu]"
pip install -r requirements.txt
```

## Run the experiments

To start the complete experiments, run:

```shell script
python3 def_benchmark_ope.py
```

WARNING: The PseudoInverse estimator is much slower than the others, you may want to run the others in parallel first with:

```shell script
python3 def_benchmark_ope_NO_PI.py
```

If you want to change the number of deficient actions (the default is 8 over a total of 80), you can do it in this way:

```shell script
python3 def_benchmark_ope_NO_PI.py setting.n_deficient_actions=m
```
where **m** can be any number between 0 and 79.


## Citation
If you use this code in your work, please cite our paper:

Bibtex:
```
@incollection{felicioni2022offpolicy,
	author    = {Nicol{\`{o}} Felicioni and Maurizio {Ferrari Dacrema} and Marcello Restelli and Paolo Cremonesi},
	title	  = {Off-Policy Evaluation with Deficient Support Using Side Information},
	booktitle = {Advances in Neural Information Processing Systems 35 (NeurIPS)},
	year      = {2022},
}
```

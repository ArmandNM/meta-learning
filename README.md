# Few shot learning by features adaptation with Graph Neural Networks

PyTorch implementation for [Few shot learning by features adaptation with Graph Neural Networks](https://andreinicolicioiu.github.io/report/few_shot_graphs/few_shot_graphs_eeml.pdf), part of my Bachelor’s thesis.

Work presented at Eastern European Summer School (EEML 2020), winning a Best Poster Award.

<br/>

<p align="center"><img src="https://armandnicolicioiu.github.io/research/few-shot-gnn/featured.png"/width = 69%></p>


## Installation
```shell
conda create -n few-shot python=3.8
conda activate few-shot
conda install pytorch==1.7.1 torchvision==0.8.2 torchaudio==0.7.2 cudatoolkit=11.0 -c pytorch
pip install higher
pip install torchmeta
pip install tensorboard
```

## Running
This will load the hyperparameters saved in `config` and start a new training process. You can stop it any time with CTRL+C.

```Bash
python main.py
```

<!-- <br/> -->

To start the process in the background, run the following command:

```shell
./start_experiment.sh [experiment_name]
```
 It will also create a new experiment directory with all the logs, config and current code changes. CTRL+C now will only stop the log from displaying, but the process continues running.

## Evaluate
Run `python main.py --test` from an experiment's copy of the project directory.
```
cd experiments/experiment__21_november_2021__15_43_36
cd project
python main.py --test
```
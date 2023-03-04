import yaml
import logging
from runner import RunnerFactory
from utils.args_parser import args
from data import *
import os, datetime

def main():
    # Load the configuration file
    
    with open(args.config, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    # Set up the logger
    logging.basicConfig(level=logging.INFO)
    logging.log(logging.INFO, "Starting the experiment: {}".format(args.task_name))
    init_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    task_name = "{}_{}_{}_{}layers_{}width_{}".format(
        args.task_name, config['dataset']['name'], config['model'],
        config['net_params']['num_layers'],config['net_params']['hidden_dim'], init_time
    )
    config['task_name'] = task_name
    config['save_path'] = os.path.join(config['save_path'], task_name)
    if not os.path.exists(config['save_path']):
        os.makedirs(config['save_path'])
    if config['task_level'] == 'node':
        runner = RunnerFactory.create_runner('NodeClassification', config)
    
    runner.run()

main()
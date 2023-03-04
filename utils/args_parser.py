import argparse


parser = argparse.ArgumentParser()

parser.add_argument('--config', required=True,help="Please give a config.yaml file with training/model/data/param details")
parser.add_argument('--task_name', required=True,help="Please give a task name")
args = parser.parse_args()
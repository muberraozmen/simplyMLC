import argparse
import json
from loader import *
from processor import *
from builder import *
from runner import *


def main(opt):
    data = load(opt['data'])
    train_data, test_loader, num_features, num_labels = process(data, **opt['data']['processor'])
    model = Model(num_features, num_labels, **opt['model'])
    run(train_data, test_loader, model, opt['experiment'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--configuration", default="debug.json")
    args = parser.parse_args()
    with open(args.configuration, 'r') as json_file:
        json_data = json_file.read()
        config = json.loads(json_data)
    main(config)

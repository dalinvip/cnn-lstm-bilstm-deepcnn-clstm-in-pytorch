import os
import argparse
import datetime
import Config.config as configurable
import torch
from DataUtils.Common import seed_num

if __name__ == "__main__":
    print("Process ID {}, Process Parent ID {}".format(os.getpid(), os.getppid()))
    parser = argparse.ArgumentParser(description="Neural Networks")
    parser.add_argument('--config_file', default="./Config/config.cfg")
    args = parser.parse_args()

    config = configurable.Configurable(config_file=args.config_file)
    if config.no_cuda is True:
        print("Using GPU To Train......")
        # torch.backends.cudnn.enabled = True
        # torch.backends.cudnn.deterministic = True
        torch.cuda.manual_seed(seed_num)
        torch.cuda.manual_seed_all(seed_num)
    print("torch.cuda.initial_seed", torch.cuda.initial_seed())





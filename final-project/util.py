import logging
import time
from datetime import datetime
import torch
import random
import numpy as np
def fix_seeds(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    np.random.seed(seed)  # Numpy module.
    random.seed(seed)  # Python random module.
    torch.backends.cudnn.benchmark = False  # For pytorhc reproducible
    torch.backends.cudnn.deterministic = True
def model_setting():
	# model setting
	use_cuda=torch.cuda.is_available()
	if use_cuda:
		device=torch.device('cuda')
	else :
		device=torch.device('cpu' )
	return device
def gen_logger(log_file_name):
	# set up logging to file - see previous section for more details
	logging.basicConfig(level=logging.DEBUG,
						format='%(asctime)s %(name)-12s %(levelname)-8s %(message)s',
						datefmt='%m-%d %H:%M',
						filename=log_file_name,
						filemode='w')
	# define a Handler which writes INFO messages or higher to the sys.stderr
	console = logging.StreamHandler()
	console.setLevel(logging.INFO)
	# set a format which is simpler for console use
	formatter = logging.Formatter('%(name)-12s: %(levelname)-8s %(message)s')
	# tell the handler to use this format
	console.setFormatter(formatter)
	# add the handler to the root logger
	logging.getLogger().addHandler(console)
	logger = logging.getLogger(str(datetime.now().strftime("%Y:%H:%M:%S")))
	# quiet PIL log
	logging.getLogger('PIL').setLevel(logging.WARNING)
	return logger
if __name__ == "__main__":
	L = gen_logger("./test.log")
	L.info("{} QQ is my boss QQ".format(87))

""" Sets up training runs w/ regularizations """



"""
Training with scheduled lipschitz evaluation -- what do I need? 
- training parameters 
- lipschitz evaluation parameters 
- how often to do lipschitz evaluation


How to best organize this data? 
What will I use it for exactly? 
+ plotting average lipschitz values across points
+ calculating time for performance across various models 
[i.e., evaluation at very end only]
"""

import sys 
import os 
sys.path.append(os.path.join(os.getcwd(), '..'))
import copy

from neural_nets.train import TrainParameters, training_loop
from lipMIP import LipParameters, LipMIPResult, LipMIPEvaluation


def train_and_evaluate(network, num_epochs, train_params, lip_params, 
					   eval_after_every):
	""" Interleaves training and lipschitz evaluation
	"""
	eval_results = []
	epoch_no = 0

	while True:
		epochs_remaining = num_epochs - epoch_no
		for i in range(eval_after_every):
			new_params = copy.copy(train_params)
			new_params.num_epochs = min([eval_after_every, epochs_remaining])
			training_loop(network, new_params, epoch_start_no=epoch_no)
			epoch_no += new_params.num_epochs

			
		if epochs_remaining == 0:
			break

	return eval_results




	# make sub-training-loops 
	num_loops = num_epochs / eval_after_every + 1 # always eval at end

	main_loop_num_epochs = 
	final_loop_num_epochs = main_loop_num_epochs * eva
	train_params = copy.copy(train_params)



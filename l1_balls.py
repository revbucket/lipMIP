import numpy as np
import torch
import torch.nn as nn
from hyperbox import Domain
import utilities as utils


class L1BallFactory(Domain):
	def __init__(self, radius):
		self.radius = radius 

	def make_l1_ball(center):
		return L1Ball.build_linf_ball(center, self.radius)


class L1Ball(object):
	""" Class representing l1 balls """
	def __init__(self, center, radius):
		self.center = center 
		self.radius = radius

	@classmethod
	def make_unit_ball(dimension):
		""" Makes the unit l1 ball in the correct dimension """
		return L1Ball(np.zeros(dimension), 1.0)


	def encode_as_gurobi_model(squire, key, num_elements):		
		model = squire.model
		pos_name = key + '_pos'
		neg_name = key + '_neg'
		# Create namers
		key_namer = utils.build_var_namer(key)
		pos_namer = utils.build_var_namer(pos_key)
		neg_namer = utils.build_var_namer(neg_key)

		key_vars, pos_vars, neg_vars = [], [], []

		# Add all the variables and constraints
		for i in range(num_elements):
			pos_vars.append(model.addVar(lb=0.0, ub=1.0, name=pos_namer(i)))
			neg_vars.append(model.addVar(lb=0.0, ub=1.0, name=neg_namer(i)))			
			key_vars.append(model.addVar(lb=-1.0, ub=1.0, name=key_namer(i)))

			model.addConstr(key_vars[-1] == pos_vars[-1] - neg_vars[-1])

		model.addConstr(sum(pos_vars) + sum(neg_vars) <= 1.0)
		model.update() 

		for pair in [(key, key_vars), (pos_key, pos_vars), (neg_key, neg_vars)]:
			squire.set_vars(*pair)

		return

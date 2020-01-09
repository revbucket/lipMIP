""" Subfile to split neural nets into multiple levels where we then
	run LipMIP on each.

	Main steps to do here: 
	- 1) Clean way to specify how much splitting to do [CHECK]
	- 2) Generate techniques to split neural nets into multiple neural nets  
	[CHECK]
	- 3) Generate a way to propagate input region bounds throughout 
		 (might mean making a zonotope class later)
	- 4) Be clear about the math for what I'm computing at each step 

"""

# ====================================================================
# =           Split Parameters Object                                =
# ====================================================================

class SplitParameters:
    """ Object to hold 'how should we partition the NN?' for splitMIP.
        This should just pretty simple 
    """
    @classmethod
    def from_dict(cls, dict_input):
        keys = list(dict_input.keys())
        assert [k in {'num_splits', 'every_x', 'manual_splits'} for k in keys]
        assert len(keys) == 1
        return self.__init__(**dict_input)


    def __init__(self, num_splits=None, every_x=None, manual_splits=None):
        """ Three flavors: 
            - num_splits: split into a fixed number of subnetworks
            - every_x : split into subnetworks with a max number of HIDDEN
                        LAYERS per subnetwork 
            - manual_splits: split into subnetworks where each subnetwork 
                             has a specified number of subnetworks 
        num_splits : if not None is an int
        every_x : if not None is an int
        manual_split: if not None is a list of ints with the sum being equal
                      to the number of HIDDEN LAYERS
        """
        assert sum(_ is None for _ in [num_splits, every_x, manual_splits]) == 1
        if num_splits is not None:
            self.num_splits = num_splits
        if every_x is not None:
            self.every_x = num_splits
        if manual_splits is not None:
            self.manual_splits = manual_splits


    def to_dict(self):
        output_dict = {}
        if self.num_splits is not None:
            output_dict['num_splits'] = self.num_splits

        if self.every_x is not None:
            output_dict['every_x'] = self.every_x

        if self.manual_splits is not None:
            output_dict['manual_splits'] = self.manual_splits

        return output_dict


# ===========================================================
# =           SplitMIP                                      =
# ===========================================================


"""
FLESHING THIS OUT:
- want a function to split problems into multiple subproblems
	+ input: network, splitParameters, solveParameters (when we do that)
	+ output: number of subproblems, each associated with a 
		where each subproblem has a (squire, model, preacts)
"""

class SplitLiPProblem:
	def __init__(self, network, lip_params, split_params):
		self.network = network
		self.lip_params = lip_params
		self.split_params = split_params
		self.subproblems = self._make_subproblems()

	def _make_subproblems(self):
		""" Makes LipProblem objects for splitting, based on how we split 
		Three steps: 
			  i) create subnetworks
			 ii) create LipParam objects with new domains reflected
			iii) create new LipParams->LipProblems
		"""
		#   i)
		subnetworks = self.network.make_subnetworks(self.split_params)

		#  ii)
		preact = PreactivationBounds.naive_ia_from_hyperbox(self.lip_params.domain)
		lip_paramses = []
		for bound in self.bound_iter:
			new_domain = Hyperbox.from_twocol(bound)
			new_lip_Params = LipParameters.change_domain(new_domain)
			lip_paramses.append(new_lip_Params)

		# iii)
		return [LipProblem(subnet, param) for (subnet, param) in
				zip(subnetworks, lip_paramses)]


	def upper_bound_lipschitz(self, answer_only=False):
		""" Solves each subproblem in serial and returns the answer.
		ARGS:
			answer_only: bool - if True, the output to this is a float 
		RETURNS:
			if answer_only, the answer as a float 
			otherwise, the {'results': [list of LipMIPResult objects]
						    'answer': answer}
		"""

		# Solve each subproblem in serial
		subresults = [subproblem.compute_max_lipschitz() 
					  for subproblem in subproblems]
		answer = utils.prod(_.value for _ in subresults)

		if answer_only:
			return answer
		else:
			return {'results': subresults, 'answer': answer}
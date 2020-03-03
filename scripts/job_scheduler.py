""" File to pick up jobs from the jobs/scheduled directory and 
	run them and put them and their result in the job/completed directory 
"""
import matlab.engine
import re 
import os
import shutil
import sys 
import pickle
import argparse
sys.path.append('..')
from experiment import Job

JOBS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
					   'jobs')

SCHEDULE_DIR = os.path.join(JOBS_DIR, 'scheduled')
COMPLETED_DIR = os.path.join(JOBS_DIR, 'completed')

def handle_job(filename):
	with open(os.path.join(SCHEDULE_DIR, filename), 'rb') as f:
		job = pickle.load(f)

	job_out = job.run(write_to_file=False, num_threads=4)
	result_file = os.path.splitext(filename)[0] + '.result'
	with open(os.path.join(COMPLETED_DIR, result_file), 'wb') as f:
		pickle.dump(job_out, f)

	shutil.move(os.path.join(SCHEDULE_DIR, filename),
				os.path.join(COMPLETED_DIR, filename))

	return



def main(name=None, local_or_global=None):
	""" General naming convention of jobs is like 
		'<NAME>_<JOB_SPECIFIC_INFO>_(LOCAL|GLOBAL).job'

	So this file will search for all strings like that in the schedule 
	directory and run them 
	"""

	# Step 1: build regex 
	re_suffix = r'\.job'
	re_prefix = r''
	assert local_or_global in ['LOCAL', 'GLOBAL', None]
	if local_or_global is not None:
		re_suffix = r'_' + local_or_global + re_suffix
	if name is not None:
		re_prefix = name
	re_pattern = re_prefix + r'.*?' + re_suffix

	# Step 2: enumerate all jobs matching regex 
	files = sorted([f for f in os.listdir(SCHEDULE_DIR) 
					if re.match(re_pattern, f)])

	# Step 3: enter main scheduler loop 
	for filename in files:
		print('Handling file: %s...' % filename, end='', flush=True)
		try:
			handle_job(filename)
			print("\tSUCCESS!")
		except Exception as err:
			print("\tFAILURE!", err)





if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('-n', '--name', default=None, action='store',
 					    required=False, type=str)
	parser.add_argument('-l', '--local_or_global', default=None, action='store', 
						required=False, type=str)
	args = parser.parse_args()
	main(name=args.name, local_or_global=args.local_or_global)
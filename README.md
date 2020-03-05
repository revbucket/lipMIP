# LipMIP

Mixed-integer programming formulation for evaluating local Lipschitz constants of ReLU networks. We also hope this can be used as a general-purpose repository for estimating Lipschitz constants of ReLU networks, using a medley of techniques.

Check out our paper on arXiv: [Exactly Computing the Local Lipschitz Constant of ReLU Networks](https://arxiv.org/abs/2003.01219). 
--- 
# News
- 03/04/2020: ArXiv Release and Version 0.1 deployed
---

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes.

### Installation Instructions
Requisite python packages are contained within the file `requirements.txt`. We use pytorch for automatic differentiation and handy neural network utilities. We also use Gurobi to solve mixed-integer programs. 

We now provide detailed installation instructions:
**1) Clone this repository and all submodules.**
```
$ git clone https://github.com/revbucket/lipMIP.git
$ git submodule update --init --recursive
```
**2) Install `pip` dependencies.** We recommend that dependencies be installed within a virtual environment. Instructions to set this up are [here](https://docs.python.org/3/library/venv.html). With the virtual environment activated, we start by installing all `pip`-based python packages:
```
$ cd lipMIP
$ pip3 install -r requirements.txt
```

**3) Install Gurobi and Gurobipy.** First download the most recent version of the Gurobi optimizer (accessible with a free academic license). [Gurobi Website.](https://www.gurobi.com/downloads/gurobi-optimizer-eula/)

Then add the following environment variables to a file that runs on terminal startup (e.g. `~/.bash_profile`):
```
export GUROBI_HOME="/your/path/to/gurobi/linux64"
export PATH="${PATH}:${GUROBI_HOME}/bin"
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH}:${GUROBI_HOME}/lib"
```

And install the license obtained from the Gurobi website with the command:
```
$ grbgetkey aaaa0000-0000-0000-0000-000000000000
```

Finally install and test the gurobipy package to your python virtual environment by calling 
```
$ cd $GUROBI_HOME
$ python3 setup.py install 
$ python3 -c "import gurobipy"
```

**4) Install Mosek and Matlab.** To evaluate against LipSDP, we'll need to install Matlab, the python-link to Matlab, and the Mosek solver. Follow the requirements section of [LipSDP](https://github.com/arobey1/LipSDP) for more details.

**5) (optional) Add kernel representing this virtual environment to Jupyter.** We use jupyter notebook quite frequently for sandboxing purposes. To reflect the virtual environment you've built within jupyter, run the following command:
```
$ python3 -m ipykernel install --name lipmip
```

## Running the tutorials
We have included three tutorials for getting started with this repository. The first one describes basic functionality of neural nets, datasets, and training. The second one describes basic and more advanced usage of LipMIP. The third one describes how to compare LipMIP to other leading Lipschitz estimation techniques. 


With the codebase installed, run the ipython notebooks provided in `tutorials/` directory. 
```shell
$ cd lipMIP 
$ jupyter notebook 
```
and navigate to the `tutorials/` directory and interact with each tutorial file.


## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details



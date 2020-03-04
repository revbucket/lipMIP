# LipMIP

Mixed-integer programming formulation for evaluating local Lipschitz constants of ReLU networks. We also hope this can be used as a general-purpose repository for estimating Lipschitz constants of ReLU networks, using a medley of techniques.

Check out our paper on arXiv: [Provable Certificates for Adversarial Examples: Fitting a Ball in the Union of Polytopes](https://arxiv.org/abs/1903.08778).

--- 
# News
- 03/04/2020: ArXiv Release and Version 0.1 deployed
---

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes.

### Dependencies
Requisite python packages are contained within the file `requirements.txt`. The [`mister_ed`](https://github.com/revbucket/mister_ed) adversarial example toolbox is used to compute upper bounds. This is maintained as a subrepository within this one.

GeoCert makes many many calls to linear program solvers (in the $\ell_\infty$ case) or LCQP solvers (in the $\ell_2$) case. We use the [Gurobi Optimizer](https://www.gurobi.com) for this. Visit their homepage to acquire a free academic license.

### Installing

1. Clone the repository:
    ```shell
    $ git clone https://github.com/revbucket/geometric-certificates
    $ cd geometric-certificates
    ```
2. Install requirements:
    ```shell
    $ pip install -r requirements.txt
    ```
---

## Running the tests

With the codebase installed, run the ipython notebooks provided to get your hands on the algorithm and visualize its behaviour. As an example:

```shell
$ cd examples 
$ jupyter notebook 2D_example.ipynb
```	

---

## Authors

* **Matt Jordan-** University of Texas at Austin 


## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details



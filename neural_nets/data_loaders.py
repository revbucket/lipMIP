
DEFAULT_DATASET_DIR = '~/datasets'
import torch 
import torchvision.transforms as transforms 
import torchvision.datasets as datasets 
import os
import numpy as np
import math
import random
import matplotlib.pyplot as plt
import utilities as utils
import csv

# ===============================================================================
# =           MNIST DATA                                                        =
# ===============================================================================

def load_mnist_data(train_or_val, digits=None, batch_size=128, shuffle=False,
                    use_cuda=False, dataset_dir=DEFAULT_DATASET_DIR):
    """ Builds the standard MNIST data loader object for training or evaluation
        of MNIST data
    ARGS:
        train_or_val: string - must be 'train' or 'val' for training or 
                               validation sets respectively 

    """
    assert train_or_val in ['train', 'val']
    use_cuda = torch.cuda.is_available() and use_cuda
    dataloader_constructor = {'batch_size': batch_size, 
                              'shuffle': shuffle, 
                              'num_workers': 4,
                              'pin_memory': use_cuda}
    transform_chain = transforms.ToTensor()
    if digits == None:
        mnist_dataset = datasets.MNIST(root=dataset_dir, 
                                       train=(train_or_val == 'train'), 
                                       download=True, transform=transform_chain)
    else:
        mnist_dataset = SubMNIST(root=dataset_dir, digits=digits,
                                 train=(train_or_val=='train'), 
                                 download=True, transform=transform_chain)

    return torch.utils.data.DataLoader(mnist_dataset, **dataloader_constructor)


class SubMNIST(datasets.MNIST):
    valid_digits = set(range(10))
    def __init__(self, root, digits, train=True, transform=None, 
                 target_transform=None, download=False):
        super(SubMNIST, self).__init__(root, transform=transform, 
                                       target_transform=target_transform)
        assert [digit in self.valid_digits for digit in digits] 
        assert digits == sorted(digits)
        target_map = {digit + 10: i for i, digit in enumerate(digits)}
        
        # --- remap targets to select out only the images we want 
        self.targets = self.targets + 10
        for digit, label in target_map.items():
            self.targets[self.targets== digit] = label

        # --- then select only indices with these new labels 
        self.data = self.data[self.targets < 10]
        self.targets = self.targets[self.targets < 10]

    @property 
    def raw_folder(self):
        return os.path.join(self.root, 'MNIST', 'raw')

    @property 
    def processed_folder(self):
        return os.path.join(self.root, 'MNIST', 'processed')

# ===============================================================================
# =           FMNIST DATA                                                       =
# ===============================================================================
def load_fmnist_data(train_or_val, batch_size=128, shuffle=True, use_cuda=False, 
                     dataset_dir=DEFAULT_DATASET_DIR):
    """ Builds the FMNIST data loader object for training/eval of FMNIST
        Same signature as load_mnist_data (with no 'digits')
    """
    assert train_or_val in ['train', 'val']
    use_cuda = torch.cuda.is_available() and use_cuda
    dataloader_constructor = {'batch_size': batch_size, 
                              'shuffle': shuffle, 
                              'num_workers': 4,
                              'pin_memory': use_cuda}
    transform_chain = transforms.ToTensor()
    
    fmnist_dataset = datasets.FashionMNIST(root=dataset_dir, 
                                    train=(train_or_val == 'train'), 
                                    download=True, transform=transform_chain)

    return torch.utils.data.DataLoader(fmnist_dataset, **dataloader_constructor)

# ===============================================================================
# =           CIFAR10 DATA                                                      =
# ===============================================================================


def load_cifar10_data(train_or_val, batch_size=128, shuffle=True, use_cuda=False, 
                      dataset_dir=DEFAULT_DATASET_DIR):
    """ Builds the FMNIST data loader object for training/eval of FMNIST
        Same signature as load_mnist_data (with no 'digits')
    """
    assert train_or_val in ['train', 'val']
    use_cuda = torch.cuda.is_available() and use_cuda
    dataloader_constructor = {'batch_size': batch_size, 
                              'shuffle': shuffle, 
                              'num_workers': 4,
                              'pin_memory': use_cuda}
    transform_chain = transforms.ToTensor()
    
    cifar10_dataset = datasets.CIFAR10(root=dataset_dir, 
                                       train=(train_or_val == 'train'), 
                                       download=True, transform=transform_chain)

    return torch.utils.data.DataLoader(cifar10_dataset, **dataloader_constructor)



# ===============================================================================
# =           IRIS DATA                                                         =
# ===============================================================================

class IRIS:
    """ Data copied from https://github.com/yangzhangalmo/pytorch-iris/
    Methods in this are:
    1) split_train_val - splits dataset into training/validation sets 
    2) display_2d - picks some random orthogonal directions and plots data 
                    in the projection to the 2D affine subspace
    """
    DATAFILE = os.path.join(os.path.dirname(__file__), 'iris.csv')
    MAPDICT = {'Iris-setosa': 0,
               'Iris-versicolor': 1, 
               'Iris-virginica': 2}
    def __init__(self):
        X, Y = [], []

        with open(self.DATAFILE, newline='\n') as csvfile:
            for i, el in enumerate(csv.reader(csvfile, delimiter=',')):
                if i == 0:
                    continue # Skip header 
                else:
                    X.append([float(_) for _ in el[:-1]])
                    Y.append(self.MAPDICT[el[-1]])

        self.X, self.Y = torch.tensor(X), torch.tensor(Y).long()
        self.train_set = None
        self.test_set = None

    def split_train_val(self, train_prop, shuffle_seed=None):
        """ First shuffles the dataset and then returns a split
            of train/val of size train_prop/(1-train_prop)
        """
        if self.train_set is not None:
            return self.train_set, self.test_set
        if shuffle_seed is not None:
            torch.manual_seed(shuffle_seed)
        randperm = torch.randperm(self.Y.numel())
        torch.manual_seed(random.randint(1, 2 ** 20)) # reset the random seed
        shuffle_X = self.X[randperm]
        shuffle_Y = self.Y[randperm]


        cutoff_idx = int(self.X.shape[0] * train_prop)
        self.train_set = [(shuffle_X[:cutoff_idx], shuffle_Y[:cutoff_idx])]
        self.test_set = [(shuffle_X[cutoff_idx:], shuffle_Y[cutoff_idx:])]

        return self.train_set, self.test_set

    def display_2d(self, rand_dirs=None, ax=None):
        """ Centers the data and plots the projection onto a random 
            (or pre-specified) 2D affine subspace passing through data 
            mean .
            RETURNS: (rand_dirs, axes obj) 
        """
        # Center the data 
        center = self.X.sum(0) / self.X.shape[0]
        center_X = self.X - center 

        # Compute the random subspace
        if rand_dirs is not None:
            rd_1, rd_2 = rand_dirs 
        else:
            rd_1 = torch.randn(self.X.shape[1])
            rd_1 /= torch.norm(rd_1)
            rd_2 = torch.randn(self.X.shape[1])
            rd_2 -= (rd_2 @ rd_1) * rd_1 
            rd_2 /= torch.norm(rd_2)
        rand_mat = torch.stack([rd_1, rd_2]).T 
        x_coords = center_X @ rand_mat 

        # Build axes object 
        if ax is None:
            fig, ax = plt.subplots(figsize=(8,8))

        # Split by class 
        for cls_idx, c in enumerate('rgb'):
            class_x = x_coords[self.Y == cls_idx]
            ax.scatter(class_x[:,0], class_x[:,1        ], c=c)

        return ((rd_1, rd_2), ax)
# ===============================================================================
# =           CIRCLE DATA                                                       =
# ===============================================================================
class CircleSet:
    # Toy Dataset from https://arxiv.org/pdf/1907.05681.pdf
    def __init__(self, N, _type=2001, random_seed=None):
        """ Creates a circle dataset according to the paper from _type:
            _type == 2001: https://arxiv.org/pdf/2001.06263.pdf 
            _type == 1907: https://arxiv.org/pdf/1907.05681.pdf
            Domain: [-4, +4]^2
            X ~ U(Domain)
            Y = 0 if (1 <= x.norm() <= 2), 1 otherwise 
        """
        if random_seed is not None:
            torch.manual_seed(random_seed)
        assert _type in [1907, 2001]

        self._type = _type
        if self._type == 1907:
            self.X = torch.rand(N, 2) * 8 - 4 
            self.Y = ((torch.norm(self.X, dim=1) - 1.5).abs() <= 0.5).long() 
            self.xmin = self.ymin = -4 
            self.xmax = self.ymax = 4
        else:
            self.X = torch.rand(N, 2) * 2 - 1 
            self.Y = ((torch.norm(self.X, dim=1)) <= 2 / math.pi).long()
            self.xmin = self.ymin = -1 
            self.xmax = self.ymax = 1

        self.train_set = None
        self.test_set = None
        torch.manual_seed(random.randint(1, 2 **20)) # reset random seed

    def split_train_val(self, train_prop, shuffle_seed=None):
        """ First shuffles the dataset and then returns a split
            of train/val of size train_prop/(1-train_prop)
        """
        if self.train_set is not None:
            return self.train_set, self.test_set

        if shuffle_seed is not None:
            torch.manual_seed(shuffle_seed)
        randperm = torch.randperm(self.Y.numel())
        torch.manual_seed(random.randint(1, 2 ** 20)) # reset the random seed
        shuffle_X = self.X[randperm]
        shuffle_Y = self.Y[randperm]


        cutoff_idx = int(self.X.shape[0] * train_prop)
        train_set = [(shuffle_X[:cutoff_idx], shuffle_Y[:cutoff_idx])]
        test_set = [(shuffle_X[cutoff_idx:], shuffle_Y[cutoff_idx:])]

        self.train_set = train_set
        self.test_set = test_set
        return train_set, test_set

    def display_2d(self, ax=None):
        if ax is None:
            fig, ax = plt.subplots(figsize=(8,8))
        ax.set_xlim(self.xmin, self.xmax)
        ax.set_ylim(self.ymin, self.ymax)

        for class_idx, c in enumerate('br'):
            class_x = self.X[self.Y == class_idx]

            ax.scatter(class_x[:,0], class_x[:,1], c=c)
        return ax 


# ===============================================================================
# =           RANDOM DATA                                                       =
# ===============================================================================


"""
TODOS:
- Build random dataset generators
  + Eric Wong's random dataset 
  + Random K-clusters
  + Swirly dataset 
"""
class RandomDatasetParams(utils.ParameterObject):
    """ Abstract class to hold generic parameters for training datasets """
    def __init__(num_training, num_val, **kwargs):
        init_args = {k: v for k,v in locals().items()
                     if k not in ['self', '__class__']}
        for k, v in kwargs.items():
            assert k not in ['num_training', 'num_val']
            init_args[k] = v
        super(RandomDatasetParams, self).__init__(**init_args)


class EricParameters(utils.ParameterObject):
    flavor = 'eric'
    """ Eric Wong's 2D dataset:
        https://github.com/locuslab/convex_adversarial/blob/master/examples/2D.ipynb

        Basic gist is to repeat the following process:
        - pick a random point with a random label 
        - if another point exists close to this point, don't add this point 
        - repeat until full of points
    PARAMETERS:
        - num points
        - radius
    """

    def __init__(self, num_points, radius, dimension=2, num_classes=2):
        super(EricParameters, self).__init__(num_points=num_points, 
                                             radius=radius,
                                             dimension=dimension,
                                             num_classes=num_classes)
class RandomKParameters(utils.ParameterObject):
    flavor = 'randomk'
    """ Random K-cluster dataset
        Basic gist is to repeat the following process:
        - pick a bunch of random points 
        - randomly select k of them to be 'leaders', randomly assign 
          labels to these leaders
        - assign labels to the rest of the points by the label of their 
          closest 'leader' 
    PARAMETERS:
        - num points
        - num leaders (k)
    """

    def __init__(self, num_points, k, radius=None, dimension=2, num_classes=2):
        super(RandomKParameters, self).__init__(num_points=num_points, 
                                                k=k, radius=radius, 
                                                dimension=dimension,
                                                num_classes=num_classes)


class SwirlyParameters(utils.ParameterObject):
    flavor = 'swirly'
    """ Random swirly dataset:
        Basic gist is to 
        - define two logarithmic spirals (one for each class)
        - sample random points along these spirals
        - add some amount of noise to these points
    PARAMETERS:
        - a, b : logarithmic spiral parameters 
        - minimum radius
        - noise-value
    """
    def __init__(self, num_points, a, b, min_t, max_t, noise_bound=None):
        super(SwirlyParameters, self).__init__(num_points=num_points, a=a, b=b,
                                               min_t=min_t, max_t=max_t,
                                               noise_bound=noise_bound,
                                               dimension=2)




class RandomDataset:
    """ Builds randomized 2-dimensional, 2-class datasets """
    def __init__(self, parameters, batch_size=128, random_seed=None):
        assert isinstance(parameters, (EricParameters, RandomKParameters, 
                                       SwirlyParameters))
        self.parameters = parameters
        self.dim = self.parameters.dimension
        self.batch_size = int(batch_size)
        if random_seed is None:
            random_seed = random.randint(1, 420 * 69)
        self.random_seed = random_seed
        np.random.seed(self.random_seed)
        self.base_data = self._generate_data()
        self.train_data = None
        self.val_data = None

    def _generate_data(self):
        return {'eric': self._generate_data_eric, 
                'randomk': self._generate_data_randomk, 
                'swirly': self._generate_data_swirly}[self.parameters.flavor]()


    def _minibatchify(self, subset1, subset2=None):
        """ Given a tensor of the shape (N,...), returns a list of 
            tensors where the subset is split into batches of the 
            self.batch_size. If a second subset is supplied, we minibatchify 
            each and zip the result
        ARGS:
            subset1: torch.Tensor (N, ...)
            subset2: if not None, is another torch.Tensor of shape (N, ...)
        RETURNS:
            list of tensors
        """
        num_points = subset1.shape[0]
        num_batches = num_points // self.batch_size
        if num_points % self.batch_size != 0:
            num_batches +=1
        minibatches = []

        for i in range(num_batches):
            mb1 =  subset1[i * self.batch_size: (i+1) * self.batch_size]
            if subset2 is not None:
                mb2 =  subset2[i * self.batch_size: (i+1) * self.batch_size]                
                minibatches.append((mb1, mb2))
            else:
                minibatches.append(mb1)
        return minibatches


    def split_train_val(self, train_prop, resplit=False):
        """ Generates two datasets, a training and validation dataset 
        ARGS:
            train_prop: float in range [0, 1] - proportion of data used 
                        ni the train set 
            resplit: bool - if True, we reshuffle these, otherwise we just 
                     return what we computed last time
        RETURNS:
            (train_set, test_set), where each is an iterable like
                [(examples, labels),...]
                where examples is one minibatch of the 2D Data
                and labels is one minibatch of the labels
        """
        if resplit is False and\
            all(d is not None for d in [self.train_data, self.val_data]) :
            return self.train_data, self.val_data

        perm = torch.randperm(self.parameters.num_points)
        num_train_data = int(train_prop * self.parameters.num_points)
        train_indices = perm[:num_train_data]
        test_indices = perm[num_train_data:]
        base_data, base_labels = self.base_data

        # make training data
        output = []
        for indices in [train_indices, test_indices]:
            examples = base_data.index_select(0, indices)    
            labels = base_labels.index_select(0, indices)
            output.append(self._minibatchify(examples, labels))
        self.train_data = output[0]
        self.val_data = output[1]
        return output


    def plot_2d(self, figsize=(8,8), ax=None):
        """ Plots the data points """
        assert self.dim == 2
        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)
        data, labels = self.base_data 
        data_np = utils.as_numpy(data)
        labels_np = utils.as_numpy(labels)
        ax.scatter(data_np[:,0], data_np[:,1], c=labels_np, cmap='coolwarm')
        return ax

    # ==============================================================
    # =          DATA GENERATION TECHNIQUES                        =
    # ==============================================================
    
    def _generate_data_eric(self):
        """ Generates Eric Wong's 2d training set """
        num_points = self.parameters.num_points
        radius = self.parameters.radius
        num_classes = self.parameters.num_classes
        data_points = self._generate_random_separated_data(num_points, radius, 
                                                           self.dim)
        while len(data_points) < num_points:
            point = np.random.uniform(size=(self.dim))
            if min(np.linalg.norm(point - a) for a in data_points) > 2 * radius:
                data_points.append(point)
        data = torch.Tensor(np.stack(data_points))
        labels = torch.randint(low=0, high=num_classes, size=(num_points,), dtype=torch.uint8)
        #labels = torch.Tensor((np.random.random(num_points) > 0.5).astype(np.uint8))
        return (data, labels.long())

    def _generate_data_randomk(self):
        """ Generates random k-cluster data """
        num_points = self.parameters.num_points
        k = self.parameters.k 
        num_classes = self.parameters.num_classes
        # first make data
        if getattr(self.parameters, 'radius') is not None:
            radius = self.parameters.radius
            data_points = self._generate_random_separated_data(num_points, radius, 
                                                              self.dim)
            data_points = np.stack(data_points)
        else:
            data_points = np.random.uniform(size=(num_points, self.dim))

        # then pick leaders and assign them labels
        leader_indices = np.random.choice(num_points, size=(k), replace=False)
        random_labels = np.random.randint(low=0, high=num_classes, size=k,
                                          dtype=np.uint8)

        # and finally assign labels to everything else 
        all_labels = np.zeros(num_points).astype(np.uint8)
        for i in range(num_points):
            min_leader_dist = np.inf
            min_leader_idx = None
            for j in range(k):
                leader = data_points[leader_indices[j]]
                dist = np.linalg.norm(leader - data_points[i])
                if dist < min_leader_dist:
                    min_leader_dist = dist 
                    min_leader_idx = j
            all_labels[i] = random_labels[min_leader_idx]

        return torch.Tensor(data_points), torch.Tensor(all_labels).long()


    def _generate_data_swirly(self):
        assert self.dim == 2
        num_points = self.parameters.num_points
        a = self.parameters.a 
        b = self.parameters.b
        t_range = self.parameters.max_t - self.parameters.min_t
        each_class = num_points // 2
        t_scale = t_range / float(each_class)
        noise_bound = self.parameters.noise_bound

        # Build method to get right spirals
        def get_data_point(t, flip):
            x = a * math.cos(t) * math.exp(b * t)
            y = a * math.sin(t) * math.exp(b * t)            
            if flip:
                x = -x
                y = -y
            if noise_bound is not None:
                noise = np.random.randn(2) * noise_bound / math.sqrt(2)
                x += noise[0]
                y += noise[1]
            return np.array([x, y])

        # Get data points
        class_0, class_1 = [], []
        for i in range(each_class):
            t = t_scale * i + self.parameters.min_t
            class_0.append(get_data_point(t, False))
            class_1.append(get_data_point(t, True))
        data_points = np.concatenate([np.stack(class_0), np.stack(class_1)])
        labels = np.concatenate([np.zeros(each_class).astype(np.uint8),
                                 np.ones(each_class).astype(np.uint8)])

        return torch.Tensor(data_points), torch.Tensor(labels).long()



    @classmethod
    def _generate_random_separated_data(cls, num_points, radius, dim):
        """ Generates num_points points in 2D at least radius apart 
            from each other 
        OUTPUT IS A LIST OF NUMPY ARRAYS, EACH OF SHAPE (dim,)
        """
        data_points = []
        while len(data_points) < num_points:
            point = np.random.uniform(size=(dim))
            if len(data_points) == 0:
                data_points.append(point)
                continue
            if min(np.linalg.norm(point - a) for a in data_points) > 2 * radius:
                data_points.append(point)
        return data_points

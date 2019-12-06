
DEFAULT_DATASET_DIR = '~/datasets'
import torch 
import torchvision.transforms as transforms 
import torchvision.datasets as datasets 
import os

# ===============================================================================
# =           MNIST DATA                                                        =
# ===============================================================================

def load_mnist_data(train_or_val, digits=None, batch_size=128, shuffle=False,
                    dataset_dir=DEFAULT_DATASET_DIR):
    """ Builds the standard MNIST data loader object for training or evaluation
        of MNIST data
    ARGS:
        train_or_val: string - must be 'train' or 'val' for training or 
                               validation sets respectively 

    """
    assert train_or_val in ['train', 'val']

    dataloader_constructor = {'batch_size': batch_size, 
                              'shuffle': shuffle, 
                              'num_workers': 4,
                              'pin_memory': False}
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
# =           RANDOM DATA                                                       =
# ===============================================================================



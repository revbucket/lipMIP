""" Analogous to the hyperbox.py file, but for zonotopes """ 

import numpy 
import torch 
import torch.nn as nn 
import torch.nn.functional as F
import copy 
import numpy as np 
import numbers 
import utilities as utils 
import gurobipy as gb
import matplotlib.pyplot as plt

from hyperbox import Domain, Hyperbox, BooleanHyperbox


class Zonotope(Domain):
    def __init__(self, dimension, 
                 center=None,
                 generator=None,
                 lbs=None,
                 ubs=None, 
                 shape=None):
        self.dimension = dimension 
        self.center = center # numpy ARRAY
        self.generator = generator # numpy 2D Array (matrix)
        self.lbs = lbs # Array
        self.ubs = ubs # Array 
        self.shape = shape # tuple of 2d-shape (for convs), possibly None
        self._set_lbs_ubs()

    def __getitem__(self, idx):
        return self.lbs[idx], self.ubs[idx]

    @classmethod
    def as_zonotope(cls, abstract_object):
        """ Takes in either a zonotope or hyperbox and returns an equivalent 
            zonotope
        """
        if isinstance(abstract_object, Zonotope):
            return abstract_object
        else:
            return cls.from_hyperbox(abstract_object)

    @classmethod
    def from_hyperbox(cls, hyperbox):
        """ Takes in a Hyperbox object and returns an equivalent zonotope """
        generator = torch.diag(hyperbox.radius)

        return cls(hyperbox.dimension,
                   center=hyperbox.center,
                   generator=generator,
                   lbs=hyperbox.box_low,
                   ubs=hyperbox.box_hi, 
                   shape=hyperbox.shape)

    @classmethod
    def from_vector(cls, vec):
        """ Takes in a vector and makes a hyperbox """
        return cls.from_hyperbox(Hyperbox.from_vector(vec))

    @classmethod
    def cast(cls, obj):
        """ Casts hyperboxes, zonopes, vectors as a zonotope """
        if isinstance(obj, Hyperbox):
            return cls.from_hyperbox(obj)
        elif isinstance(obj, (torch.Tensor, np.ndarray)):
            return cls.from_vector(obj)
        elif isinstance(obj, cls):
            return obj
        else:
            return obj.as_zonotope()            


    def set_2dshape(self, shape):
        self.shape = shape

    def random_point(self, num_points=1, tensor_or_np='tensor', 
                     requires_grad=False):
        assert tensor_or_np in ['np', 'tensor']
        shape = (num_points, self.generator.shape[1])
        rand = torch.rand((num_points, self.generator.shape[1])) *2 - 1
        rand.type(self.center.dtype)

        points = self.y(rand)

        if tensor_or_np == 'tensor':
            points = points.data.requires_grad_(requires_grad)
            if self.shape is not None:
                points = points.view((num_points,) + self.shape)
            return points            
        else:
            return utils.as_numpy(rand_points)

    def y(self, y_tensor): 
        """ Returns a tensor of points in R^n given their generating ys 
        ARGS: 
            y_tensor: tensor shape (k, m) or (m)
        RETURNS:
            tensor of center + E @ y_tensor
        """
        return self.center + (self.generator @ y_tensor.unsqueeze(-1)).squeeze(-1)

    def project_2d(self, dir_matrix):
        """ Projects this object onto the 2 provided directions, 
            can then be used to draw the shape 
        """

        lin = nn.Linear(self.dimension, 2, bias=False)
        lin.weight.data = dir_matrix
        return self.map_linear(lin)


    def map_layer_forward(self, network, i, abstract_params=None):
        layer = network.net[i]
        if isinstance(layer, nn.Linear):
            return self.map_linear(layer, forward=True)
        elif isinstance(layer, nn.Conv2d):
            return self.map_conv2d(network, i, forward=True)
        elif isinstance(layer, nn.ConvTranspose2d):
            return self.map_conv_transpose_2d(network, i, forward=True)            
        elif isinstance(layer, nn.ReLU):
            return self.map_relu()
        elif isinstance(layer, nn.LeakyReLU):
            return self.map_leaky_relu()
        elif isinstance(layer, nn.Tanh):
            if abstract_params is not None and 'deep' in abstract_params:
                return self.map_tanh_deepz()
            if abstract_params is not None and 'box' in abstract_params:
                return self.map_tanh_box()
            return self.map_tanh2()
        elif isinstance(layer, nn.Sigmoid):
            return self.map_sigmoid()
        else:
            raise NotImplementedError("unknown layer type", layer)


    def map_layer_backward(self, network, i, grad_bound, abstract_params=None):
        layer = network.net[-(i + 1)]
        forward_idx = len(network.net) - 1 - i
        if isinstance(layer, nn.Linear):
            return self.map_linear(layer, forward=False)
        elif isinstance(layer, nn.Conv2d):
            if abstract_params is not None and 'old' in abstract_params:
                print("DOING OLD")
                return self.map_conv2d_old(network, forward_idx, forward=False)
            return self.map_conv2d_old(network, forward_idx, forward=False)                
        elif isinstance(layer, nn.ConvTranspose2d):
            return self.map_conv_transpose_2d_old(network, forward_idx, forward=False)                        
        elif isinstance(layer, nn.ReLU):
            if isinstance(grad_bound, Hyperbox):
                return self.map_elementwise_mult(grad_bound)
            else:
                return self.map_switch(grad_bound, **(abstract_params or {}))
        elif isinstance(layer, nn.LeakyReLU):
            return self.map_leaky_switch(layer, grad_bound, 
                                         **(abstract_params or {}))
        elif isinstance(layer, (nn.Sigmoid, nn.Tanh)):
            return self.map_elementwise_mult(grad_bound)
        else:
            return NotImplementedError("Unknown layer type", layer)


    def map_genlin(self, linear_layer, network, layer_num, forward=True):
        if isinstance(linear_layer, nn.Linear):
            return self.map_linear(linear_layer, forward=forward)
        elif isinstance(linear_layer, nn.Conv2d):
            return self.map_conv2d(network, layer_num, forward=forward)
        else:
            raise NotImplementedError("Unknown linear layer", linear_layer)


    def map_linear(self, linear, forward=True):
        """ Takes in a torch.Linear operator and maps this object through 
            the linear map (either forward or backward)
        ARGS:
            linear : nn.Linear object - 
            forward: boolean - if False, we map this 'backward' as if we
                      were doing backprop
        """
        assert isinstance(linear, nn.Linear)
        dtype = linear.weight.dtype 

        if forward:
            new_dimension = linear.out_features
            new_center = linear(self.center)
            new_generator = linear.weight.mm(self.generator)
        else:
            new_dimension = linear.in_features
            new_center = linear.weight.T.mv(self.center)
            new_generator = linear.weight.T.mm(self.generator)
        # Return new zonotope
        new_zono = Zonotope(dimension=new_dimension,
                            center=new_center,
                            generator=new_generator)
        new_zono._set_lbs_ubs()
        return new_zono #.pca_reduction()

    def map_conv2d_old(self, network, index, forward=True):
        layer = network[index]
        assert isinstance(layer, nn.Conv2d)         
        input_shape = network.shapes[index] 
        output_shape = network.shapes[index + 1] 

        if not forward:
            input_shape, output_shape = output_shape, input_shape

        center = self.center.view((1,) + input_shape)
        generator = self.generator.T.view((-1,) + input_shape)
        gen_cols = self.generator.shape[1] 
        if forward: 
            new_center = layer(center).view(-1)
            new_gen = utils.conv2d_mod(generator, layer, 
                                       bias=False, abs_kernel=False)
            new_gen = new_gen.view((gen_cols,) + (-1,)).T 
        else:
            center_in = torch.zeros((1,) + output_shape, requires_grad=True)
            center_out = (layer(center_in) * center).sum() 
            new_center = torch.autograd.grad(center_out, center_in)[0].view(-1)

            gen_in = torch.zeros((gen_cols,) + output_shape, 
                                 requires_grad=True)
            gen_out = utils.conv2d_mod(gen_in, layer, bias=False, 
                                       abs_kernel=False)
            new_gen = torch.autograd.grad((gen_out * generator).sum(), gen_in)[0]
            new_gen = new_gen.view((gen_cols, -1)).T

        new_zono = Zonotope(dimension=new_center.numel(), center=new_center,
                            generator=new_gen, shape=output_shape)
        new_zono._set_lbs_ubs()
        return new_zono

    def map_conv2d(self, network, index, forward=True):
        layer = network[index]
        assert isinstance(layer, nn.Conv2d)         
        input_shape = network.shapes[index] 
        output_shape = network.shapes[index + 1] 

        if not forward:
            input_shape, output_shape = output_shape, input_shape

        center = self.center.view((1,) + input_shape)
        generator = self.generator.T.view((-1,) + input_shape)
        gen_cols = self.generator.shape[1] 
        if forward: 
            new_center = layer(center).view(-1)
            new_gen = utils.conv2d_mod(generator, layer, 
                                       bias=False, abs_kernel=False)
            new_gen = new_gen.view((gen_cols,) + (-1,)).T 
        else:
            new_layer = nn.ConvTranspose2d(layer.out_channels, layer.in_channels, 
                                           kernel_size=layer.kernel_size, 
                                           stride=layer.stride)
            new_layer.weight.data = layer.weight.data 
            new_layer.bias.data = torch.zeros_like(new_layer.bias.data) 
            new_center = new_layer(center).view(-1) 
            new_gen = utils.conv_transpose_2d_mod(generator, new_layer, bias=False, 
                                                  abs_kernel=False)
            new_gen = new_gen.view((gen_cols, -1)).T

        new_zono = Zonotope(dimension=new_center.numel(), center=new_center,
                            generator=new_gen, shape=output_shape)
        return new_zono


    def map_conv_transpose_2d(self, network, index, forward=True):
        layer = network[index]
        assert isinstance(layer, nn.ConvTranspose2d)         
        input_shape = network.shapes[index] 
        output_shape = network.shapes[index + 1] 

        if not forward:
            input_shape, output_shape = output_shape, input_shape

        center = self.center.view((1,) + input_shape)
        generator = self.generator.T.view((-1,) + input_shape)
        gen_cols = self.generator.shape[1] 
        if forward: 
            new_center = layer(center).view(-1)
            new_gen = utils.conv_transpose_2d_mod(generator, layer, 
                                       bias=False, abs_kernel=False)
            new_gen = new_gen.view((gen_cols,) + (-1,)).T 
        else:
            new_layer = nn.Conv2d(layer.out_channels, layer.in_channels, 
                                  kernel_size=layer.kernel_size, 
                                  stride=layer.stride,)
            new_layer.weight.data = layer.weight.data 
            new_layer.bias.data = torch.zeros_like(new_layer.bias.data)            
            new_center = new_layer(center).view(-1) 
            new_gen = utils.conv2d_mod(generator, new_layer, bias=False, 
                                       abs_kernel=False)
            new_gen = new_gen.view((gen_cols,) + (-1,)).T 

        return Zonotope(dimension=new_center.numel(), center=new_center, 
                        generator=new_gen, shape=output_shape)


    def map_conv_transpose_2d_old(self, network, index, forward=True):
        layer = network[index]
        assert isinstance(layer, nn.ConvTranspose2d)         
        input_shape = network.shapes[index] 
        output_shape = network.shapes[index + 1] 

        if not forward:
            input_shape, output_shape = output_shape, input_shape

        center = self.center.view((1,) + input_shape)
        generator = self.generator.T.view((-1,) + input_shape)
        gen_cols = self.generator.shape[1] 
        if forward: 
            new_center = layer(center).view(-1)
            new_gen = utils.conv_transpose_2d_mod(generator, layer, 
                                       bias=False, abs_kernel=False)
            new_gen = new_gen.view((gen_cols,) + (-1,)).T 
        else:
            center_in = torch.zeros((1,) + output_shape, requires_grad=True)
            center_out = (layer(center_in) * center).sum()
            new_center = torch.autograd.grad(center_out, center_in)[0].view(-1) 
            gen_in = torch.zeros((gen_cols,) + output_shape, requires_grad=True)
            gen_out = utils.conv_transpose_2d_mod(gen_in, layer, 
                            bias=False, abs_kernel=False)
            new_gen = torch.autograd.grad((gen_out * generator).sum(), gen_in)[0]
            new_gen = new_gen.view((gen_cols, -1)).T 


        return Zonotope(dimension=new_center.numel(), center=new_center, 
                        generator=new_gen, shape=output_shape)



    def map_avgpool(self, network, index, forward=True):
        layer = network[index]
        assert isinstance(layer, nn.AvgPool2d)         
        input_shape = network.shapes[index] 
        output_shape = network.shapes[index + 1] 

        if not forward:
            input_shape, output_shape = output_shape, input_shape

        center = self.center.view((1,) + input_shape)
        generator = self.generator.T.view((-1,) + input_shape)
        gen_cols = self.generator.shape[1] 
        if forward: 
            new_center = layer(center).view(-1)
            new_gen = layer(generator)
            new_gen = new_gen.view((gen_cols,) + (-1,)).T 
        else:
            center_in = torch.zeros((1,) + output_shape, requires_grad=True)
            center_out = (layer(center_in) * center).sum() 
            new_center = torch.autograd.grad(center_out, center_in)[0].view(-1)

            gen_in = torch.zeros((gen_cols,) + output_shape, 
                                 requires_grad=True)
            gen_out = layer(gen_in)
            new_gen = torch.autograd.grad((gen_out * generator).sum(), gen_in)[0]
            new_gen = new_gen.view((gen_cols, -1)).T

        new_zono = Zonotope(dimension=new_center.numel(), center=new_center,
                            generator=new_gen, shape=output_shape)
        return new_zono
        new_zono._set_lbs_ubs()
        return new_zono        


    def map_nonlin(self, nonlin):
        if nonlin == F.relu: 
            return self.map_relu()
        else: 
            return None # 

    def map_tanh2(self):
        new_trips = [self.get_tanh_hull(self.lbs[i], self.ubs[i]) 
                     for i in range(self.dimension)]
        offsets  = torch.tensor([_[0] for _ in new_trips])
        row_mult = torch.tensor([_[1] for _ in new_trips])
        new_dof  = torch.tensor([_[2] for _ in new_trips])
        center = offsets + row_mult * self.center
        gen = torch.cat([self.generator * row_mult.view(-1, 1), torch.diag(new_dof)], dim=1)
        return Zonotope(dimension=self.dimension, 
                        center=center, 
                        generator=gen, 
                        shape=self.shape)            

    def map_tanh(self, transformer='deep', add_new_cols=True):
        # Do some stupid nonsense and make this a box transformer

        tanh_lbs = F.tanh(self.lbs)
        tanh_ubs = F.tanh(self.ubs)
        new_centers = (tanh_ubs + tanh_lbs) / 2.
        new_ranges = (tanh_ubs - tanh_lbs) / 2.

        new_zono = Zonotope(dimension=self.dimension,
                            center=new_centers, 
                            generator=torch.diag(new_ranges),
                            lbs=tanh_lbs, 
                            ubs=tanh_ubs,
                            shape=self.shape)
        return new_zono

    def map_tanh_box(self):
        lows = torch.tanh(self.lbs)
        his = torch.tanh(self.ubs) 
        twocol = torch.stack([lows, his], dim=1)
        zono = Zonotope.cast(Hyperbox.from_twocol(twocol))
        zono.shape = self.shape 
        return zono


    def map_tanh_deepz(self):
        def deepz_tanh(l, u, ax=None):
            dsig = lambda x: 1- torch.tanh(x)**2
            slope = torch.min(dsig(l), dsig(u))
            mu1 = 0.5* (torch.tanh(u) + torch.tanh(l) - slope * (u + l))
            mu2 = 0.5 * (torch.tanh(u) - torch.tanh(l) - slope * (u - l))
            
            # Now plot the central line 
            upline = lambda x: slope * x + mu1 + mu2
            downline = lambda x: slope * x + mu1 - mu2
            midline = lambda x: slope * x + mu1
            if ax is not None:
                ax.plot((l, u), (upline(l), upline(u)), c='r')
                ax.plot((l, u), (downline(l), downline(u)), c='r')
                ax.plot((l, l), (downline(l), upline(l)), c='r')
                ax.plot((u, u), (downline(u), upline(u)), c='r')
            return mu1, slope, mu2
        trips = [deepz_tanh(self.lbs[i], self.ubs[i]) for i in range(self.dimension)]
        mu1s = torch.tensor([_[0] for _ in trips])
        mu2s = torch.tensor([_[2] for _ in trips])
        slopes = torch.tensor([_[1] for _ in trips])
        deep_gen = torch.cat([self.generator * slopes.view(-1, 1), torch.diag(mu2s)], dim=1)
        new_centers = mu1s + slopes * self.center 
        return Zonotope(dimension=self.dimension, 
                        center=new_centers,
                        generator=deep_gen, 
                        shape=self.shape)



    def map_sigmoid(self):
        new_trips = [self.get_sigmoid_hull(self.lbs[i], self.ubs[i]) 
                     for i in range(self.dimension)]
        offsets  = torch.tensor([_[0] for _ in new_trips])
        row_mult = torch.tensor([_[1] for _ in new_trips])
        new_dof  = torch.tensor([_[2] for _ in new_trips])
        center = offsets + row_mult * self.center
        gen = torch.cat([self.generator * row_mult.view(-1, 1), torch.diag(new_dof)], dim=1)
        return Zonotope(dimension=self.dimension, 
                        center=center, 
                        generator=gen, 
                        shape=self.shape)

    def map_sigmoid_deepz(self):
        def deepz_sigmoid(l, u, ax=None):
            dsig = lambda x: torch.sigmoid(x) * (1 - torch.sigmoid(x))
            slope = torch.min(dsig(l), dsig(u))
            mu1 = 0.5* (torch.sigmoid(u) + torch.sigmoid(l) - slope * (u + l))
            mu2 = 0.5 * (torch.sigmoid(u) - torch.sigmoid(l) - slope * (u - l))
            
            # Now plot the central line 
            upline = lambda x: slope * x + mu1 + mu2
            downline = lambda x: slope * x + mu1 - mu2
            midline = lambda x: slope * x + mu1
            if ax is not None:
                ax.plot((l, u), (upline(l), upline(u)), c='r')
                ax.plot((l, u), (downline(l), downline(u)), c='r')
                ax.plot((l, l), (downline(l), upline(l)), c='r')
                ax.plot((u, u), (downline(u), upline(u)), c='r')
            return mu1, slope, mu2
        trips = [deepz_sigmoid(self.lbs[i], self.ubs[i]) for i in range(self.dimension)]
        mu1s = torch.tensor([_[0] for _ in trips])
        mu2s = torch.tensor([_[2] for _ in trips])
        slopes = torch.tensor([_[1] for _ in trips])
        deep_gen = torch.cat([self.generator * slopes.view(-1, 1), torch.diag(mu2s)], dim=1)
        new_centers = mu1s + slopes * self.center 
        return Zonotope(dimension=self.dimension, 
                        center=new_centers,
                        generator=deep_gen, 
                        shape=self.shape)



    def map_relu(self, transformer='deep', add_new_cols=True):

        single_method = {'box': self.zbox_single,
                         'diag': self.zdiag_single,
                         'switch': self.zswitch_single,
                         'smooth': self.zsmooth_single,
                         'deep': self.deepz_single}[transformer]
        single_outputs = [single_method(i) for i in range(self.dimension)]
        return self._apply_single_outputs(single_outputs, 
                                          add_new_cols=add_new_cols)

    def map_leaky_relu(self, layer, transformer='deep', add_new_cols=True):
        single_method = {'box': self.zbox_single_leaky,
                         'diag': self.zdiag_single_leaky,
                         'switch': self.zswitch_single_leaky,
                         'deep': self.deepz_single_leaky}[transformer]
        single_outputs = [single_method(i, layer) for i in 
                          range(self.dimension)]
        return self._apply_single_outputs(single_outputs, 
                                          add_new_cols=add_new_cols)


    def map_nonlin_backwards(self, nonlin_obj, grad_bound):
        if nonlin_obj == F.relu:
            if isinstance(grad_bound, BooleanHyperbox):
                return self.map_switch(grad_bound)
        elif nonlin_obj == None:
            return self
        else:
            raise NotImplementedError("ONLY RELU SUPPORTED")


    def map_switch(self, bool_box, transformer='deep', add_new_cols=True):
        """ Returns a new zonotope corresponding to a switch function applied 
            to all elements in self, with the given boolean-hyperbox 
        """
        single_method = {'box': self.sBox_single, 
                         'diag': self.sDiag_single,
                         'deep': self.deepS_single}[transformer]
        single_outputs = [single_method(i, bool_box) 
                          for i in range(self.dimension)]
        return self._apply_single_outputs(single_outputs, 
                                          add_new_cols=add_new_cols)


    def map_elementwise_mult(self, hbox, transformer='deep', add_new_cols=True):
        """ Returns a new zonotope corresponding to an elementwise mult fxn 
            applied to all elements in self, where the hbox is the range to 
            multiply by
        """
        single_outputs = [] # (new center coordinate, mult level, none or col)
        for i, (glo, ghi) in enumerate(hbox): 
            # Constant case 
            if glo - ghi == 0:
                single_outputs.append((glo * self.center[i], glo * self.generator[i], None))
            else:
                max_coord = max([abs(self.ubs[i]), abs(self.lbs[i])])
                vert_range =  max_coord * (ghi - glo) / 2
                scale = (glo + ghi) / 2
                new_col = torch.zeros_like(self.center) 
                new_col[i] = vert_range 
                single_outputs.append((scale * self.center[i], scale * self.generator[i], new_col))
        return self._apply_single_outputs(single_outputs, add_new_cols=add_new_cols)


    def map_leaky_switch(self, layer, bool_box, transformer='deep', 
                         add_new_cols=True):
        """ Returns a new zonotope corresponding to a leaky switch function 
            applied to all elements in self, with the given boolean-hyperbox 
        """

        single_outputs = [self.deepS_single_leaky(i, layer, bool_box) 
                          for i in range(self.dimension)]
        return self._apply_single_outputs(single_outputs, 
                                          add_new_cols=add_new_cols)


    def map_abs(self):
        """ Returns a new zonotope that maps all elements through the absolute 
            value operator 

            If l_i > 0, can leave as is 
            If u_i < 0, can negate everything 
            O.w., multiply each row by (u+l)/(u-l) and replace c with the right value 
        """
        new_center = self.center.clone() 
        new_generator = self.generator.clone() 

        # Handle u_i <0 case 
        neg_idxs = self.ubs <= 0 
        new_center[neg_idxs] *=-1 
        new_generator[neg_idxs, :] *= -1 

        # Handle uncertain cases 
        unc = (self.ubs * self.lbs < 0)
        if unc.sum() > 0:
            sum_unc = self.ubs[unc] + self.lbs[unc]
            diff_unc = self.ubs[unc] - self.lbs[unc]

            # Slope is sum/diff 
            # Radius to be added is (u * (1 - slope) / 2)
            # New center is (radius + (sum/2) * slope)

            slope = sum_unc / diff_unc 
            new_rad = self.ubs[unc] * (1 - slope) / 2
            new_center_coords = new_rad + sum_unc * slope / 2


            # Modify center/generator
            new_generator[unc, :] *= slope.view(-1, 1)
            new_center[unc] = new_center_coords

            # Make new cols and append 
            new_cols = torch.zeros(self.dimension, unc.sum().item()) 
            new_cols[unc, :] = new_rad.diag()
            new_generator = torch.cat([new_generator, new_cols], dim=1)


        new_zono = Zonotope(dimension=self.dimension, 
                            center=new_center,
                            generator=new_generator, 
                            shape=self.shape)
        return new_zono


    def map_relu_efficient(self, layer):
        new_center = self.center.clone() 
        new_generator = self.generator.clone() 

        # Handle negative case
        neg_idxs = self.ubs <= 0
        new_center[neg_idxs] = 0.
        new_generator[neg_idxs,:] = 0.

        # Handle uncertain case 
        unc = (self.ubs * self.lbs) < 0 
        if unc.sum() > 0:
            sum_unc = self.ubs[unc] + self.lbs[unc]         
            diff_unc = self.ubs[unc] - self.lbs[unc] 
            prod_unc = self.ubs[unc] * self.lbs[unc]
            slope = self.ubs[unc] / diff_unc 

            new_rad = -prod_unc / (2 * diff_unc)
            new_generator[unc, :] *= slope.view(-1, 1)
            new_center_coords = self.ubs[unc] * slope / 2
            new_center_coords = new_rad + (slope * self.center[unc]) #new_rad + sum_unc * slope / 2
            new_center[unc] = new_center_coords
            new_cols = torch.zeros(self.dimension, unc.sum().item())
            new_cols[unc,:] = new_rad.diag() 
            new_generator = torch.cat([new_generator, new_cols], dim=1)

        new_zono = Zonotope(dimension=self.dimension, 
                            center=new_center, 
                            generator=new_generator,
                            shape=self.shape) 
        return new_zono

    def map_elementwise_mult_efficient(self, hbox):
        new_center = self.center.clone() 
        new_generator = self.generator.clone() 
        """ 
        How many cases? 
        glo = ghi
        """
        const_idxs = (hbox.radius == 0)
        new_center[const_idxs] *= hbox.center[const_idxs]
        new_generator[const_idxs, :] *= hbox.center[const_idxs].view(-1, 1)

        var_idxs = (hbox.radius != 0)
        if var_idxs.sum().item() > 0:
            max_coords = torch.max(abs(self.ubs[var_idxs]), abs(self.lbs[var_idxs]))
            vert_range = max_coords * (hbox.box_hi[var_idxs] - hbox.box_low[var_idxs]) / 2 
            scale = hbox.center[var_idxs]
            new_center[var_idxs] = self.center[var_idxs] * scale
            new_cols = torch.zeros(self.dimension, var_idxs.sum().item())
            new_cols[var_idxs,:] = vert_range.diag() 
            new_generator[var_idxs, :] *= scale
            new_generator = torch.cat([new_generator, new_cols], dim=1)

        new_zono = Zonotope(dimension=self.dimension, 
                            center=new_center, 
                            generator=new_generator,
                            shape=self.shape) 
        return new_zono


    def _apply_single_outputs(self, single_outputs, add_new_cols=True):
        new_center = torch.tensor([_[0] for _ in single_outputs])
        new_generator = torch.stack([_[1] for _ in single_outputs])
        new_cols = [_[2] for _ in single_outputs if _[2] is not None]
        if len(new_cols) > 0 and add_new_cols:
            new_generator = torch.cat([new_generator, 
                                       torch.stack(new_cols).T], dim=1)
        new_zono = Zonotope(dimension=self.dimension,
                            center=new_center,
                            generator=new_generator,
                            shape=self.shape)
        new_zono._set_lbs_ubs()
        return new_zono


    def _set_lbs_ubs(self):
        """ Takes in a Zonotope object without self.lbs, self.ubs set
            and modifies these attributes 
        """
        if self.center is None or self.generator is None:
            return 
        radii = torch.abs(self.generator).sum(1)
        self.lbs = self.center - radii
        self.ubs = self.center + radii 

    def as_hyperbox(self):
        if self.lbs is None or self.ubs is None:
            self._set_lbs_ubs()
        twocol = torch.stack([self.lbs, self.ubs]).T
        box_out = Hyperbox.from_twocol(twocol)
        box_out.set_2dshape(self.shape)
        return box_out

    def pca_reduction(self):
        """ PCA order reduction technique from 
            'Methods for order reduction of zonotopes' (Kopetzki et al)
        """
        # 1) center the generator and compute SVD
        column_mean = torch.mean(self.generator, dim=1, keepdim=True)
        center_gen = self.generator - column_mean
        U, S, V = torch.svd(center_gen, some=True)

        # 2) Map zonotope through the U, and turn into a box 
        new_center = U.T.mv(self.center)
        new_generator = U.T.mm(self.generator)
        intermed_zono = Zonotope(dimension=self.dimension, 
                                 center=new_center,
                                 generator=new_generator, 
                                 shape=self.shape)
        new_zono = Zonotope.from_hyperbox(intermed_zono.as_hyperbox())

        # 3) Go from box back to rotated space
        final_center = U.mv(new_zono.center)
        final_generator = U.mm(new_zono.generator)
        final_zono = Zonotope(dimension=self.dimension,
                              center=final_center,
                              generator=final_generator,
                              shape=self.shape)
        final_zono._set_lbs_ubs()
        return final_zono

    def as_boolean_hbox(self, params=None):
        return BooleanHyperbox.from_zonotope(self)

    def contains(self, points):
        """ runs .contains_point(...) for every point in points """
        if points.dim() == 1:
            points.view(1, -1)
        return self.contains_batch(points)

    def contains_batch(self, points):
        eps = 1e-6
        with utils.silent():
            model = gb.Model() 
        model.setParam('OutputFlag', False)
        gb_vars = [model.addVar(lb=-1.0, ub=1.0) 
                   for i in range(self.generator.shape[1])]
        var_namer = utils.build_var_namer('x')
        x_vars = [model.addVar(lb=self.lbs[i], ub=self.ubs[i], name=var_namer(i))
                  for i in range(self.dimension)] 


        for i in range(self.dimension):
            model.addConstr(x_vars[i] == 
                            gb.LinExpr(self.generator[i], gb_vars) + self.center[i])
        model.update() 

        contain_list = []
        for point in points:
            for i, el in enumerate(point):
                x_vars[i].lb = el-eps 
                x_vars[i].ub = el + eps 
            model.update() 
            model.optimize() 
            contain_list.append(model.Status not in [3,4])
        return contain_list


    def get_2d_boundary(self, num_points): 
        """ For 2d zonotopes, will draw them by rayshooting along coordinates
        ARGS: 
            num_points : int - number of points to check 
        RETURNS: 
            tensor of shape [num_points, 2] which outlines the boundary
        """
        range_matrix = torch.arange(num_points + 1) / float(num_points) * (2 * np.pi)
        cos_els = range_matrix.cos() 
        sin_els = range_matrix.sin() 

        dir_matrix = torch.stack([cos_els, sin_els]).T 
        argmaxs = (dir_matrix @ self.generator).sign()
        points = self.y(argmaxs) 

        return points.detach()

    def draw_2d_boundary(self, ax, num_points=1000, c=None):
        points = self.get_2d_boundary(num_points)
        if c is not None:
            ax.plot(*zip(*points), c=c)
        else:
            ax.plot(*zip(*points))

    def maximize_l1_norm_abs(self):
        sum_operator = nn.Linear(self.dimension, 1, bias=False)
        sum_operator.weight.data = torch.ones_like(sum_operator.weight.data)
        return self.map_abs().map_linear(sum_operator).ubs[0]


    def maximize_l1_norm_mip(self, verbose=False, num_threads=2, time_limit=None):
        """ naive gurobi technique to maximize the l1 norm of this zonotope
        RETURNS:
            opt_val - float, optimal objective value
         """
        model = self._build_l1_mip_model(verbose=verbose, 
                                         num_threads=num_threads)
        if time_limit is not None:
            model.setParam('TimeLimit', time_limit)
        model.optimize()
        return model.ObjBound

    def maximize_l1_norm_lp(self, verbose=False, num_threads=2):
        model = self._build_l1_mip_model(verbose=verbose,
                                         num_threads=num_threads)
        for var in model.getVars():
            if var.VType == gb.GRB.BINARY:
                var.VType = gb.GRB.CONTINUOUS
                var.LB = 0.0 
                var.UB = 1.0
        model.update()
        model.optimize()
        return model.ObjBound

    def maximize_l1_norm_coord(self):
        return torch.max(self.lbs.abs(), self.ubs.abs()).sum()

    def _build_l1_mip_model(self, verbose=False, num_threads=2):
        with utils.silent():
            model = gb.Model() 

        tolerance = 1e-6
        if not verbose: 
            model.setParam('OutputFlag', False)
        model.setParam('Threads', num_threads)
        gen_vars = [model.addVar(lb=-1, ub=1, name='y_%08d' % i)
                    for i in range(self.generator.shape[1])]

        # Now add variables for each coordinate
        x_vars = []
        for i, gen_row in enumerate(self.generator):
            x_vars.append(model.addVar(lb=self.lbs[i], ub=self.ubs[i], 
                                       name='x_%08d' % i))
            model.addConstr(gb.LinExpr(gen_row, gen_vars) + self.center[i] == x_vars[-1])

        # Now consider the absolute value of each x_var 
        t_vars = []
        for i, x_var in enumerate(x_vars):
            if self.ubs[i] <= 0:
                t_vars.append(model.addVar(lb=0.0, ub=abs(self.lbs[i]) + tolerance))
                model.addConstr(t_vars[i] == -x_vars[i])
            elif self.lbs[i] >= 0:
                t_vars.append(model.addVar(lb=0.0, ub=abs(self.ubs[i]) + tolerance))
                model.addConstr(t_vars[i] == x_vars[i])
            else:
                t_vars.append(model.addVar(lb=0.0, 
                                           ub=max([self.ubs[i], -self.lbs[i]])))
                bin_var = model.addVar(lb=0, ub=1, vtype=gb.GRB.BINARY)                
                t_var = t_vars[-1]
                x_var = x_vars[i]
                lb, ub = self.lbs[i], self.ubs[i]
                model.addConstr(t_var >= x_var)
                model.addConstr(t_var >= -x_var)
                model.addConstr(t_var <= x_var - 2 *  lb * (1 - bin_var))
                model.addConstr(t_var <= -x_var + 2 * ub * bin_var)                

        model.setObjective(sum(t_vars), gb.GRB.MAXIMIZE)
        model.update()
        return model     


    def maximize_linf_norm(self):
        """ Returns maximal l_inf norm of any point inside this zono 
        """
        return max([max(abs(self.lbs)), max(abs(self.ubs))])


    def check_orthants(self, coords, orthant_list=None):
        """ Checks which orthants are feasible in this zonotope 
        ARGS:
            coords: int[], which coordinate indices we consider the 
                    restriction to 
            orthant_list: None or list of possible orthants (like 
                             ['00', '01', '11,...] 
        RETURNS:
            dict with all orthants checked, and False if infeasible and 
            a proof of orthant intersection otherwise 
        """
        # Generate orthants to check: 
        def generate_power_set(n, init_list=None):
            if init_list is None:
                init_list = ['0', '1']
            if n == 1:
                return init_list
            return generate_power_set(n - 1, [_ + b for _ in init_list 
                                                    for b in ['0', '1']])

        if orthant_list is None:
            orthant_list = generate_power_set(len(coords))

        # Now check each orthant 
        model = None 
        output_dict = {} 
        for orthant in orthant_list: 
            output, model = self._check_single_orthant(coords, orthant, 
                                                       model=model)
            output_dict[orthant] = output

        return output_dict

    def _check_single_orthant(self, coords, single_orthant, model=None):
        """ Checks feasibility of a single orthant wrt some coords
            of the zonotope 
        ARGS:
            coords: int[] list of coordinates to check only 
            single_orthant: binary string that corresponds to which orthants
                            to check
            model: gurobi model (if it exists), only needs to be changed
                   if a new model is proposed 
        """
        assert len(coords) == len(single_orthant)
        x_namer = utils.build_var_namer('x')
        z_constr_namer = utils.build_var_namer('z_constr')
        # Build model if doesn't exist
        if model is None:
            with utils.silent():
                model = gb.Model() 
                model.setParam('OutputFlag', False)
            # Add variables for x, y 
            y_vars = [model.addVar(lb=-1, ub=1) 
                      for _ in range(self.generator.shape[1])]
            x_vars = [model.addVar(lb=self.lbs[i]-1, ub=self.ubs[i] + 1, name=x_namer(i))
                      for i in coords]

            # Add linExpressions for each x,y 
            for index, i in enumerate(coords):
                model.addConstr(self.center[i].item() + gb.LinExpr(self.generator[i], y_vars) ==\
                                x_vars[index])
            z_var = model.addVar(lb=0.0, name='z')
            model.setObjective(z_var, gb.GRB.MAXIMIZE)

        else: # otherwise, just remove the z-constraints
            x_vars = [model.getVarByName(x_namer(i)) for i in coords]
            z_var = model.getVarByName('z')
            for i in coords:
                model.remove(model.getConstrByName(z_constr_namer(i)))
        model.update()

        # Now add new z constraints and optimize
        for index, i in enumerate(coords):
            sign_var = 1
            if single_orthant[index] == '0':
                sign_var = -1
            model.addConstr(z_var <= sign_var * x_vars[index],
                            name=z_constr_namer(i))
        model.setObjective(z_var, gb.GRB.MAXIMIZE)            
        model.update()
        model.optimize()

        # Process the optimization output:
        if model.Status == 2:
            output = (float(model.objVal), np.array([_.X for _ in x_vars]))
        else:
            output = False
        return output, model 


    def encode_as_gurobi_model(self, squire, key):
        model = squire.model 
        namer = utils.build_var_namer(key)
        gb_vars = []
        raise NotImplementedError("Build this later!")
    

    # =======================================================
    # =           Single ReLU Transformer Methods           =
    # =======================================================
    # Each function here returns a (center_coord, gen_row, gen_col)
    # - center_coord is just a float for the i^th coord of new center
    # - gen_row is a new generator row (none if unchanged)
    # - gen_col is a new generator col (none if not needed)

    def _z_known(self, i):
        if self.lbs[i] >= 0:
            return (self.center[i], self.generator[i], None)
        if self.ubs[i] <= 0:
            return (0, torch.zeros(self.generator.shape[1]), None)

    def zbox_single(self, i):
        if self.lbs[i] * self.ubs[i] >= 0:
            return self._z_known(i)

        center_coord = self.ubs[i] / 2.0
        gen_row = torch.zeros(self.generator.shape[1])
        gen_col = torch.zeros(self.dimension)
        gen_col[i] = self.ubs[i] / 2.0

        return (center_coord, gen_row, gen_col)

    def zdiag_single(self, i):
        if self.lbs[i] * self.ubs[i] >= 0:
            return self._z_known(i)        

        center_coord = self.center[i] - self.lbs[i] / 2.0 
        gen_row = self.generator[i]
        gen_col = torch.zeros(self.dimension)
        gen_col[i] = -self.lbs[i] / 2.0
        return (center_coord, gen_row, gen_col)


    def zswitch_single(self, i):
        if abs(self.lbs[i]) > abs(self.ubs[i]):
            return self.zbox_single(i)
        else:
            return self.zdiag_single(i)

    def zsmooth_single(self, i):
        if self.lbs[i] * self.ubs[i] >= 0:
            return self._z_known(i)

        abs_range = abs(self.lbs[0]) + abs(self.ubs[0])
        zbox_c, zbox_row, zbox_col = self.zbox_single(i)
        zbox_weight = abs(self.lbs[0]) / abs_range

        zdiag_c, zdiag_row, zdiag_col = self.zdiag_single(i)
        zdiag_weight = abs(self.ubs[0]) / abs_range

        # compute center
        new_center = zbox_c * zbox_weight + zdiag_c * zdiag_weight

        # compute row
        new_row = zbox_row * zbox_weight + zdiag_row * zdiag_weight

        # compute col 
        if (zbox_col is None) and (zdiag_col is None):
            new_col = None 
        else:
            if zbox_col is None:
                zbox_col = torch.zeros(self.dimension)
            if zdiag_col is None:
                zdiag_col = torch.zeros(self.dimension)
            new_col = zbox_col * zbox_weight + zdiag_col * zdiag_weight

        return (new_center, new_row, new_col)


    def deepz_single(self, i):
        if self.lbs[i] * self.ubs[i] >= 0:
            return self._z_known(i)
        lambda_ = self.ubs[i] / (self.ubs[i] - self.lbs[i])
        mu_ = -1 * self.ubs[i] * self.lbs[i] / (self.ubs[i] - self.lbs[i])
        new_center = lambda_ * self.ubs[i] / 2.0
        # new_center = self.center[i] * lambda_ + mu_ 
        new_row = self.generator[i] * lambda_
        new_col = torch.zeros(self.dimension)
        new_col[i] = mu_/2.0 +1e-6
        return (new_center, new_row, new_col)

    # ======  End of Single ReLU Transformer Methods  =======

    # =======================================================
    # =           Single LEAKY ReLU Transformer Methods     =
    # =======================================================
    # Each function here returns a (center_coord, gen_row, gen_col)
    # - center_coord is just a float for the i^th coord of new center
    # - gen_row is a new generator row (none if unchanged)
    # - gen_col is a new generator col (none if not needed)
    # All of these take args (i, layer)
    # where i is the coordinate index and layer is the LeakyReLU instance 
    def _z_known_leaky(self, i, layer):
        if self.lbs[i] >= 0:
            return (self.center[i], self.generator[i], None)
        if self.ubs[i] <= 0:
            neg = layer.negative_slope
            return (self.center[i] * neg, self.generator[i] * neg, None)

    def zbox_single_leaky(self, i, layer):
        if self.lbs[i] * self.ubs[i] >= 0:
            return self._z_known_leaky(i, layer)
        neg = layer.negative_slope
        full_range = self.ubs[i] + neg * self.lbs[i]
        center_coord = full_range / 2.0 
        gen_row = torch.zeros(self.generator.shape[1])
        gen_col = torch.zeros(self.dimension)
        gen_col[i] = full_range / 2.0 

        retun (center_coord, gen_row, gen_col)

    def zdiag_single_leaky(self, i, layer):
        if self.lbs[i] * self.ubs[i] >= 0:
            return self._z_known_leaky(i, layer)

        neg = layer.negative_slope 
        assert neg < 1.0 # Assumption: leaky relus need slope < 1
        center_coord = self.center[i] + (neg - 1) * self.lbs[i] / 2.0
        gen_row = self.generator[i] 
        gen_col = torch.zeros(self.dimension)
        gen_col[i] = (neg - 1) * self.lbs[i] / 2.0
        return (center_coord, gen_row, gen_col)

    def zswitch_single_leaky(self, i, layer):
        if abs(self.lbs[i]) > abs(self.ubs[i]):
            return self.zbox_single_leaky(i, layer)
        else:
            return self.zdiag_single_leaky(i, layer)

    def deepz_single_leaky(self, i, layer):
        if self.lbs[i] * self.ubs[i] >= 0:
            return self._z_known_leaky(i, layer)

        neg = layer.negative_slope
        assert neg < 1.0
        # Need to establish:
        # 1) slope 
        # 2) y_max (new y range)
        # 3) new center 
        u, l = self.ubs[i], self.lbs[i]
        lambda_ = (u - neg * l) / (u - l)
        ymax = (1 - neg) * (-u * l) / (u - l)

        center_coord = (u * u - neg * l * l) / (2 * (u - l))
        gen_row = self.generator[i] * lambda_
        gen_col = torch.zeros(self.dimension)
        gen_col[i] = ymax / 2.0 

        return (center_coord, gen_row, gen_col)

    # ======  End of Single LEAKY ReLU Transformer Methods  =======

    # ===========================================
    # =           Single SWITCH ReLU            =
    # ===========================================
    # Each function here returns a (center_coord, gen_row, gen_col)
    # - center_coord is just a float for the i^th coord of new center
    # - gen_row is a new generator row (none if unchanged)
    # - gen_col is a new generator col (none if not needed)
    def _s_known(self, i, bool_box, leaky=None):
        if bool_box[i] == 1:
            return (self.center[i], self.generator[i], None)

        if bool_box[i] == -1:
            return (0.0, torch.zeros(self.generator.shape[1]), None)


    def sBox_single(self, i, bool_box):
        if bool_box[i] != 0:
            return self._s_known(i, bool_box)

        gen_row = torch.zeros(self.generator.shape[1])
        gen_col = torch.zeros(self.dimension)
        if self.lbs[i] >= 0:
            center_coord = self.ubs[i] / 2.0
            gen_col[i] = self.ubs[i] / 2.0
        elif self.ubs[i] <= 0:
            center_coord = self.lbs[i] / 2.0 
            gen_col[i] = -self.lbs[i] / 2.0
        else:
            center_coord = (self.ubs[i] + self.lbs[i]) / 2.0 
            gen_col[i] = (self.ubs[i] - self.lbs[i]) / 2.0

        return (center_coord, gen_row, gen_col)

    def sDiag_single(self, i, bool_box):
        if bool_box[i] != 0:
            return self._s_known(i, bool_box)

        gen_row = self.generator[i]
        gen_col = torch.zeros(self.dimension)

        if self.lbs[i] >= 0: 
            center_coord = self.center[i] - self.ubs[i] / 2.0 
            gen_col[i] = self.ubs[i] / 2.0
        elif self.ubs[i] <= 0: 
            center_coord = self.center[i] - self.lbs[i] / 2.0 
            gen_col[i] = -self.lbs[i] / 2.0
        else:
            center_coord = self.center[i] - (self.ubs[i] + self.lbs[i]) / 2.0 
            gen_col[i] = (self.ubs[i] - self.lbs[i]) / 2.0

        return (center_coord, gen_row, gen_col)

    def deepS_single(self, i, bool_box):
        if bool_box[i] != 0:
            return self._s_known(i, bool_box)
        if self.ubs[i] * self.lbs[i] >= 0:
            return self.sBox_single(i, bool_box)

        range_ = self.ubs[i] - self.lbs[i]
        gen_col = torch.zeros(self.dimension)
        if self.ubs[i] >= -self.lbs[i]:
            lambda_ = -self.lbs[i] / range_
            center_coord = (lambda_ * self.center[i] + self.ubs[i] / 2.0 +
                            self.ubs[i] *self.lbs[i] / range_)
            gen_row = lambda_ * self.generator[i]
            gen_col[i] = self.ubs[i] / 2.0
        else:
            lambda_ = self.ubs[i] / range_
            center_coord = (lambda_ * self.center[i] + self.lbs[i] / 2.0 -
                            self.ubs[i] * self.lbs[i] / range_)
            gen_row = lambda_ * self.generator[i] 
            gen_col[i] = -self.lbs[i] / 2.0 


        return (center_coord, gen_row, gen_col)


    # =======================================================
    # =           Single LEAKY SWITCH Transformer Methods     =
    # =======================================================
    # Each function here returns a (center_coord, gen_row, gen_col)
    # - center_coord is just a float for the i^th coord of new center
    # - gen_row is a new generator row (none if unchanged)
    # - gen_col is a new generator col (none if not needed)
    # All of these take args (i, layer, bool_box)
    # where i is the coordinate index and layer is the LeakyReLU instance 

    def _s_known_leaky(self, i, layer, bool_box):
        mult = 1.0 
        if bool_box[i] == -1:
            mult = layer.negative_slope

        return (mult * self.center[i], mult * self.generator[i], None)


    def deepS_single_leaky(self, i, layer, bool_box):
        # If this LeakyReLU is always OFF of ON -- everything becomes constant
        if bool_box[i] != 0:
            return self._s_known_leaky(i, layer, bool_box)

        neg = layer.negative_slope
        assert neg < 1
        l = self.lbs[i]
        u = self.ubs[i]
        med = (l + u) / 2.0
        ymax = (1 - neg) * max([abs(l), abs(u)])
        gen_col = torch.zeros(self.dimension)
        gen_col[i] = ymax / 2.0


        # essentially four cases 
        # Case 1: both l, u >=0 
        #         + slope is neg
        #         + center passes through (u, (1+neg) * u / 2)

        # Case 2: both l, u <= 0
        #         + slope is neg 
        #         + center passes through (l, (1+neg) * l / 2)

        if l * u >= 0: # cases 1 and 2 here
            lambda_ = neg
            gen_row = self.generator[i] * lambda_
            if u > 0:
                center_coord = (1 + neg) * u / 2 + lambda_ * (l - u) / 2.0
            else:
                center_coord = (1 + neg) * l / 2 + lambda_ * (u - l) / 2.0                

        # Cases 3 and 4: l < 0 < u
        # Case 3 occurs when u > |l|
        #        + max-vert is (1-neg) * u
        #        + slope is min([u-neg*l, neg*u-l])/(u-l)
        #        + center passes through (u, (1+neg) * u / 2)

        #Case 4: u < |l|
        #        + max-vert is (1-neg) * |l|
        #        + slope is min([u-neg*l, neg*u-l]) / (u-l)
        #        + center passes through (l, (1+neg) * l / 2)
        else:
            lambda_ = min([u - neg * l, neg * u - l]) / (u - l)
            gen_row = lambda_ * self.generator[i]
            if u > abs(l):
                cx, cy = (u, (1 + neg) * u / 2.0)
            else:
                cx, cy = (l, (1 + neg) * l / 2.0)
            center_coord = cy + lambda_ * (med - cx)


        return (center_coord, gen_row, gen_col)

    # ===============================================================
    # =           Sigmoid mapping methods                           =
    # ===============================================================
    def secant_sigmoid_grad(self, x0):
        def grad_fxn(x): 
            return (x - x0) * torch.sigmoid(x) * (1 - torch.sigmoid(x)) -\
                    (torch.sigmoid(x) - torch.sigmoid(x0))
        return grad_fxn


    def get_sigmoid_hull(self, l, u, plot=False):
        # Cleaning up... 

        # First step is to generate the convex hull... 
        if l > 0: 
            # If always positive, then lower hull is always a line
            offset, slope, dof = self.sigmoid_lower_line(l, u)
        elif u < 0: 
            # If always negative, then upper hull is always a line 
            offset, slope, dof = self.sigmoid_upper_line(l, u) 
        else:
            # Otherwise, we may have to compute some things... 
            # First consider the points where upper and lower lines happen 
            upline_secant = self.secant_sigmoid_grad(l)
            downline_secant= self.secant_sigmoid_grad(u)
            ut = utils.monotone_down_zeros(upline_secant, torch.tensor(0.), u * 2)
            lt = utils.monotone_down_zeros(downline_secant, l * 2, torch.tensor(0.))
            if lt < l:
                offset, slope, dof = self.sigmoid_lower_line(l, u)
            elif ut > u: 
                offset, slope, dof = self.sigmoid_upper_line(l, u)
            else:
                offset, slope, dof = self.sigmoid_tripart(l, u, lt, ut)

        if plot:
            # First plot the sigmoid 
            fig, ax = plt.subplots(figsize=(8,8))

            xs = torch.arange(1001) / 1000.0 * (u - l) + l
            ax.plot(xs.detach(), torch.sigmoid(xs).detach(), c='b')
            upper_line = lambda x: slope * x + offset + dof 
            lower_line = lambda x: slope * x + offset - dof 
            ax.plot([l, u], [upper_line(l), upper_line(u)], c='g')
            ax.plot([l, u], [lower_line(l), lower_line(u)], c='g')
            ax.plot([l, l], [lower_line(l), upper_line(l)], c='g')
            ax.plot([u, u], [lower_line(u), upper_line(u)], c='g')
            return ax, (offset, slope, dof) 

        return (offset, slope, dof)

        
    def sigmoid_lower_line(self, l, u):
        # When we know the lower hull is a line...
        slope = (torch.sigmoid(u) - torch.sigmoid(l)) / (u - l)
        secline = lambda x: torch.sigmoid(l) + slope * (x  -l)
        xstar = torch.logit((0.5 + torch.sqrt(1 - 4 * slope) / 2))
        altitude = torch.sigmoid(xstar) - secline(xstar) 
        offset = (secline(xstar) + torch.sigmoid(xstar)) / 2 - slope * xstar 
        return (offset, slope, altitude/2)

        
    def sigmoid_upper_line(self, l, u):
        # When we know the upper hull is a line... 
        # When the upper hull is a line 
        # sigma'(x) = sigma(x)(1-sigma(x)) = (sigma(u)-sigma(l)) / (u-l)
        slope = (torch.sigmoid(u) - torch.sigmoid(l)) / (u - l)
        secline = lambda x: torch.sigmoid(l) + slope * (x  -l)
        xstar = torch.logit((0.5 - torch.sqrt(1 - 4 * slope) / 2))
        altitude = secline(xstar) - torch.sigmoid(xstar)
        
        offset = (secline(xstar) + torch.sigmoid(xstar)) / 2 - slope * xstar 
        return (offset, slope, altitude/2)

    def sigmoid_tripart(self, l, u, lt, ut):
        # Altitude is one of the three ranges 
        upslope = (torch.sigmoid(ut) - torch.sigmoid(l)) / (ut - l)
        upline = lambda x: torch.sigmoid(l) + upslope * (x - l)# line connecting (l, sigma(l)), and (ut, sigma(ut))
        uprange = upline(lt) - torch.sigmoid(lt) 
        
        downslope = (torch.sigmoid(lt) - torch.sigmoid(u)) /(lt - u) 
        downline = lambda x: torch.sigmoid(u) + downslope * (x - u)
        downrange = torch.sigmoid(ut) - downline(ut)

        # And then get altitude of each of three points 
        # For [l, lt] does there exist a point with sigma'(x) = upslope 
        lowmax = torch.logit((1 - torch.sqrt(1 - 4 * upslope)) / 2.0)
        himax = torch.logit((1 + torch.sqrt(1 - 4 * downslope))/ 2.0)
        # Now get max altitude amongst these three parts: 
        
        lowmax = torch.min(lowmax, lt)
        low_alt = upline(lowmax) - torch.sigmoid(lowmax)
        
        himax = torch.max(himax, ut) 
        hi_alt = torch.sigmoid(himax) - downline(himax)    
        if low_alt > hi_alt: 
            xstar = lowmax 
            slope = upslope 
            line = upline 
            altitude = low_alt
        else:
            xstar = himax 
            slope =downslope 
            line = downline 
            altitude = hi_alt
            
        offset = (line(xstar) + torch.sigmoid(xstar)) /2 - slope * xstar
        return (offset, slope, altitude/2)

    # =============================================
    # =           Tanh mapping methods            =
    # =============================================

    def secant_tanh_grad(self, x0):
        def grad_fxn(x): 
            return (x - x0) * (1 - torch.tanh(x) **2) - (torch.tanh(x) - torch.tanh(x0))
        return grad_fxn

    def get_tanh_hull(self, l, u, plot=False):

        if l > 0: 
            offset, slope, dof = self.tanh_lower_line(l, u) 
        elif u < 0: 
            offset, slope, dof = self.tanh_upper_line(l, u) 
        else:
            # Otherwise, we may have to compute some things... 
            # First consider the points where upper and lower lines happen 
        
            upline_secant = self.secant_tanh_grad(l)
            downline_secant= self.secant_tanh_grad(u)
            ut = utils.monotone_down_zeros(upline_secant, torch.tensor(0.), u * 2)
            lt = utils.monotone_down_zeros(downline_secant, l * 2, torch.tensor(0.))
        
            if lt < l:
                offset, slope, dof = self.tanh_lower_line(l, u)
            elif ut > u: 
                offset, slope, dof = self.tanh_upper_line(l, u)
            else:
                offset, slope, dof = self.tanh_tripart(l, u, lt, ut)
        if plot: 
            fig, ax = plt.subplots(figsize=(8,8))

            xrange = torch.arange(1001) / 1000. * (u - l) + l 
            ax.plot(xrange, torch.tanh(xrange), c='b')
            upper_line = lambda x: slope * x + offset + dof 
            lower_line = lambda x: slope * x + offset - dof 
            ax.plot([l, u], [upper_line(l), upper_line(u)], c='g')
            ax.plot([l, u], [lower_line(l), lower_line(u)], c='g')
            ax.plot([l, l], [lower_line(l), upper_line(l)], c='g')
            ax.plot([u, u], [lower_line(u), upper_line(u)], c='g')
            return ax
        return (offset, slope, dof)
        
            
    def tanh_lower_line(self, l, u):
        # When we know the lower hull is a line...
        slope = (torch.tanh(u) - torch.tanh(l)) / (u - l)
        secline = lambda x: torch.tanh(l) + slope * (x  -l)
        
        xstar = torch.atanh(torch.sqrt(1 - slope))
        altitude = torch.tanh(xstar) - secline(xstar) 
        offset = (secline(xstar) + torch.tanh(xstar))/2 - slope * xstar 
        return (offset, slope, altitude/2)

    def tanh_upper_line(self, l, u): 
        slope = (torch.tanh(u) - torch.tanh(l)) / (u - l)
        secline = lambda x: torch.tanh(l) + slope * (x  -l)
        
        xstar = torch.atanh(-torch.sqrt(1 - slope))
        altitude = secline(xstar) - torch.tanh(xstar)
        offset = (secline(xstar) + torch.tanh(xstar))/2 - slope * xstar 
        return (offset, slope, altitude/2)



    def tanh_tripart(self, l, u, lt, ut):
        # Altitude is one of the three ranges 
        upslope = (torch.tanh(ut) - torch.tanh(l)) / (ut - l)
        upline = lambda x: torch.tanh(l) + upslope * (x - l)# line connecting (l, sigma(l)), and (ut, sigma(ut))
        uprange = torch.tanh(lt) - upline(lt)
        
        downslope = (torch.tanh(lt) - torch.tanh(u)) /(lt - u) 
        downline = lambda x: torch.tanh(u) + downslope * (x - u)
        downrange = downline(ut) - torch.tanh(ut)

        # And then get altitude of each of three points 
        # For [l, lt] does there exist a point with sigma'(x) = upslope 
        lowmax = torch.atanh(-torch.sqrt(1 - upslope))
        himax = torch.atanh(torch.sqrt(1 - downslope))
        

        # Now get max altitude amongst these three parts: 
        lowmax = torch.min(lowmax, lt)
        low_alt = upline(lowmax) - torch.tanh(lowmax)
        
        himax = torch.max(himax, ut) 
        hi_alt = torch.tanh(himax) - downline(himax) 
        if low_alt > hi_alt: 
            xstar = lowmax 
            slope = upslope 
            line = upline 
            altitude = low_alt
        else:
            xstar = himax 
            slope =downslope 
            line = downline 
            altitude = hi_alt
            
        offset = (line(xstar) + torch.tanh(xstar)) /2 - slope * xstar
        return (offset, slope, altitude/2.0)
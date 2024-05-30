# Importing necessary packages 
import torch
from torch import nn 
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F
import brevitas.nn as qnn


# TODO: after making sure that the model is working fine, remove all unnecessary printing statements from all the classes below 


# Customized Linear layer (multiplying the weight matrix with the sparsity mask to induce pruning)
class SparseLinear(qnn.QuantLinear):
    def __init__(self, in_features: int, out_features: int, mask: nn.Module, bias: bool = True) -> None:
        super(SparseLinear, self).__init__(in_features=in_features, out_features=out_features, bias=bias)
        self.mask = mask

    def forward(self, input: Tensor) -> torch.Tensor:
        #print(f"self.weight shape: {self.weight.shape}")
        #print(self.mask.print_mask_size())
        return F.linear(input, self.weight * self.mask(), self.bias)
    

# Applying the customized Linear forward function defined above along with the input and/or output quantization function from the brevitas module 
# TODO: this class is to be further customized to allow for more functionality when the hardware is intoduced 
class SparseLinearNeq(nn.Module):
    def __init__(self, in_features: int, out_features: int, input_quant, output_quant, mask, apply_input_quant=True, apply_output_quant=True, first_linear=True) -> None:
        super(SparseLinearNeq, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.input_quant = input_quant
        self.fc = SparseLinear(in_features, out_features, mask)
        self.output_quant = output_quant
        self.is_lut_inference = False
        self.neuron_truth_tables = None
        self.apply_input_quant = apply_input_quant
        self.apply_output_quant = apply_output_quant
        self.first_linear = first_linear

    def lut_cost(self):
        """
        Approximate how many 6:1 LUTs are needed to implement this layer using 
        LUTCost() as defined in LogicNets paper FPL'20:
            LUTCost(X, Y) = (Y / 3) * (2^(X - 4) - (-1)^X)
        where:
        * X: input fanin bits
        * Y: output bits 
        LUTCost() estimates how many LUTs are needed to implement 1 neuron, so 
        we then multiply LUTCost() by the number of neurons to get the total 
        number of LUTs needed.
        NOTE: This function (over)estimates how many 6:1 LUTs are needed to implement
        this layer b/c it assumes every neuron is connected to the next layer 
        since we do not have the next layer's sparsity information.
        """
        # Compute LUTCost of 1 neuron
        _, input_bitwidth = self.input_quant.get_scale_factor_bits()
        _, output_bitwidth = self.output_quant.get_scale_factor_bits()
        input_bitwidth, output_bitwidth = int(input_bitwidth), int(output_bitwidth)
        x = input_bitwidth * self.fc.mask.fan_in # neuron input fanin
        y = output_bitwidth 
        neuron_lut_cost = (y / 3) * ((2 ** (x - 4)) - ((-1) ** x))
        # Compute total LUTCost
        return self.out_features * neuron_lut_cost
    
    def forward(self, x: Tensor) -> Tensor:
        if self.apply_input_quant:
            x = self.input_quant(x)

        print('linear forward')
        #print('before view', x.shape)
        if self.first_linear:
            x = x.view(x.size(0), -1)
            print('reshaping done')
        #print('after view', x.shape)
        x = self.fc(x)
        #print('after Fc layer', x.shape)
        if self.apply_output_quant and self.output_quant is not None:
            x = self.output_quant(x)
        print('linear layer done')
        return x
    


# Customized convolutional forward function where the weight matrix is multiplied by the mask to induce sparsity into the layer 
class SparseConv1d(qnn.QuantConv1d):
    def __init__(self, in_channels, out_channels, mask:nn.Module, kernel_size=3, padding=1, bias=False) -> None:
        super(SparseConv1d, self).__init__(in_channels, out_channels, kernel_size, padding=padding, bias=bias)
        self.mask = mask
    def forward(self, input) -> Tensor:
        #print('Input shape:', input.shape)
        #print('Weight shape:', self.weight.shape)
        masked_weights = self.weight * self.mask()
        #print('Masked weights shape:', masked_weights.shape)
        output = F.conv1d(input, masked_weights, self.bias, padding=self.padding)
        #print('Output shape of SparseConv1d:', output.shape)
        return output


# Applying the customized convolutional forward function defined above along with the input and/or output quantization function from the brevitas module 
# TODO: this class is to be further customized to allow for more functionality when the hardware is intoduced 
class SparseConv1dNeq(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size,  input_quant, output_quant, mask, apply_input_quant=True, apply_output_quant=True, padding=1) -> None:
        super(SparseConv1dNeq, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.input_quant = input_quant
        #self.mask = mask
        self.padding = padding
        self.conv = SparseConv1d(in_channels, out_channels, mask, kernel_size, padding=padding, bias=False)
        self.output_quant = output_quant
        self.apply_input_quant = apply_input_quant
        self.apply_output_quant = apply_output_quant 
    
    def lut_cost(self):
        """
        Approximate how many 6:1 LUTs are needed to implement this layer using 
        LUTCost() as defined in LogicNets paper FPL'20:
            LUTCost(X, Y) = (Y / 3) * (2^(X - 4) - (-1)^X)
        where:
        * X: input fanin bits
        * Y: output bits 
        LUTCost() estimates how many LUTs are needed to implement 1 neuron, so 
        we then multiply LUTCost() by the number of neurons to get the total 
        number of LUTs needed.
        NOTE: This function (over)estimates how many 6:1 LUTs are needed to implement
        this layer b/c it assumes every neuron is connected to the next layer 
        since we do not have the next layer's sparsity information.
        """
        # Compute LUTCost of 1 neuron
        _, input_bitwidth = self.input_quant.get_scale_factor_bits()
        _, output_bitwidth = self.output_quant.get_scale_factor_bits()
        input_bitwidth, output_bitwidth = int(input_bitwidth), int(output_bitwidth)
        x = input_bitwidth * self.conv.mask.fan_in # neuron input fanin
        y = output_bitwidth 
        neuron_lut_cost = (y / 3) * ((2 ** (x - 4)) - ((-1) ** x))
        # Compute total LUTCost
        return self.out_channels * neuron_lut_cost
    
    def forward(self, x: Tensor) -> Tensor:
        if self.apply_input_quant:
            #print('before input quant', x.shape)
            x = self.input_quant(x)
            print('before self.conv', x.shape)
        x = self.conv(x)
        print('after self.conv',x.shape)
        if self.apply_output_quant:
            print('applying output quant')
            x = self.output_quant(x)
            print('shape after out quant', x.shape)
            print('output_quantized done')
        return x
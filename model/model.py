# Importing libraries and function 
from torch import nn 
import torch.nn as nn

from brevitas.quant import IntBias
from brevitas.nn import QuantConv1d, QuantLinear
from brevitas.nn import QuantReLU
from brevitas.core.scaling import ScalingImplType
from brevitas.core.quant import QuantType
# Importing functions from the directory 
from utils.mask import Conv1DMask, DenseMask2D, RandomFixedSparsityMask2D, RandomFixedSparsityConv1DMask
from utils.layers import SparseConv1dNeq, SparseLinearNeq
from utils.quant import QuantBrevitasActivation, ScalarBiasScale



# The model definition 
# TODO: remove unnecessary print statements after ensuring that everthing is working 
class QuantizedRadioml(nn.Module):
    def __init__(self, model_config): 
        super(QuantizedRadioml, self).__init__()
        self.model_config = model_config
        self.maxpool = nn.MaxPool1d(2)
        self.num_neurons = [self.model_config['input_length']] + self.model_config['hidden_layers'] + [self.model_config['output_length']]
        print(self.num_neurons)
        layer_list = []
        print(layer_list)

        # QNN model structure 
        for i in range(1, len(self.num_neurons)): 
            print('i, layer number:', i)
            in_features = self.num_neurons[i-1]
            print('in_feature:', in_features)
            out_features = self.num_neurons[i]
            print('out_feature:', out_features)

            # post transfroms 
            pool = nn.MaxPool1d(2) 

            # applying batch norm for the out_features in each layer
            bn = nn.BatchNorm1d(out_features) 

            if  i == 1:   # first layer architecture 
                #quantized_input = qnn.QuantReLU(act_quant= InputQuantizer) #try this if u want to include the class above
                input_quantized = QuantBrevitasActivation(QuantReLU(bit_width=model_config['input_bitwidth'], max_val=2.0, min_val=-2.0, quant_type=QuantType.INT, scaling_imp_type=ScalingImplType.CONST, narrow_range=False), pre_transforms=None, post_transforms=None)
                output_quantized = QuantBrevitasActivation(QuantReLU(bit_width=model_config['hidden_bitwidth'], max_val=2.0, min_val=-2.0, quant_type=QuantType.INT, scaling_imp_type=ScalingImplType.CONST), pre_transforms=[bn], post_transforms=[pool])
                mask1 = Conv1DMask(out_channels=out_features, in_channels=in_features, kernel_size=3 ) # this mask has been used in the first layer as it returns a mask with all elements set to 1
                mask1.print_mask_size()
                layer = SparseConv1dNeq(in_channels=in_features, out_channels=out_features, kernel_size=3, input_quant=input_quantized, output_quant=output_quantized, mask=mask1, padding=1)
                layer_list.append(layer)
            elif  i == len(self.num_neurons)-1:   # last layer architecture 
                output_bias_scale = ScalarBiasScale(bias_init=0.33) # this function will be imported later 
                output_quantized = QuantBrevitasActivation(QuantReLU(bit_width=model_config['output_bitwidth'], max_val=2.0, min_val=-2.0, narrow_range=False, quant_type=QuantType.INT, scaling_imp_type=ScalingImplType.CONST), post_transforms=[output_bias_scale])
                mask2 = DenseMask2D(in_features=in_features, out_features=out_features) # in the last layer a mask with all elements set to 1 is also applied   
                mask2.print_mask_size()
                layer = SparseLinearNeq(in_features=in_features, out_features=out_features, input_quant=layer_list[-1].output_quant, output_quant=output_quantized, mask=mask2, apply_output_quant=False, apply_input_quant=False, first_linear=False)
                layerz = QuantLinear(in_features=in_features, out_features=out_features, weight_bit_width = 8, bias=True, bias_quant=IntBias)

                layer_list.append(layer)
            
            elif i == len(self.num_neurons)-2:   #hidden linear layers architecture (normal)
                output_quantized = QuantBrevitasActivation(QuantReLU(bit_width=model_config['hidden_bitwidth'], max_val=2.0, min_val=-2.0, quant_type=QuantType.INT, scaling_imp_type=ScalingImplType.CONST), pre_transforms=[bn], post_transforms=None)
                mask3 = RandomFixedSparsityMask2D(in_features=in_features, out_features=out_features, fan_in=model_config['hidden_fanin'])
                mask3.print_mask_size()
                layer = SparseLinearNeq(in_features=in_features, out_features=out_features, input_quant=layer_list[-1].output_quant, output_quant=output_quantized, mask=mask3, apply_input_quant=False, first_linear=False)
                layer_list.append(layer)
            
            elif i == len(self.num_neurons)-3:   # first hidden linear layers architecture
                output_quantized = QuantBrevitasActivation(QuantReLU(bit_width=model_config['hidden_bitwidth'], max_val=2.0, min_val=-2.0, quant_type=QuantType.INT, scaling_imp_type=ScalingImplType.CONST), pre_transforms=[bn], post_transforms=None)
                mask44 = RandomFixedSparsityMask2D(in_features=512, out_features=out_features, fan_in=model_config['hidden_fanin'])
                mask44.print_mask_size()
                layer = SparseLinearNeq(in_features=512, out_features=out_features, input_quant=layer_list[-1].output_quant, output_quant=output_quantized, mask=mask44, apply_input_quant=False, first_linear=True)
                layer_list.append(layer)

            elif i == len(self.num_neurons)-4:  # last conv layer architecture 
                output_quantized = QuantBrevitasActivation(QuantReLU(bit_width=model_config['hidden_bitwidth'], max_val=2.0, min_val=-2.0, quant_type=QuantType.INT, scaling_imp_type=ScalingImplType.CONST), pre_transforms=[bn], post_transforms=[pool])
                mask4 = RandomFixedSparsityConv1DMask(out_channels=out_features, in_channels=in_features, kernel_size=3, fan_in=model_config['conv_fanin'] )
                mask4.print_mask_size()
                layer = SparseConv1dNeq(in_channels=in_features, out_channels=out_features, kernel_size=3, input_quant=layer_list[-1].output_quant, output_quant=output_quantized, mask=mask4, padding=1, apply_input_quant=False)
                layer_list.append(layer)
            else:   # hidden conv layers architecture 
                output_quantized = QuantBrevitasActivation(QuantReLU(bit_width=model_config['hidden_bitwidth'], max_val=2.0, min_val=-2.0, quant_type=QuantType.INT, scaling_imp_type=ScalingImplType.CONST), pre_transforms=[bn], post_transforms=[pool])
                mask5 = RandomFixedSparsityConv1DMask(out_channels=out_features, in_channels=in_features, kernel_size=3, fan_in=model_config['conv_fanin'] )
                mask5.print_mask_size()
                layer = SparseConv1dNeq(in_channels=in_features, out_channels=out_features, kernel_size=3, input_quant=layer_list[-1].output_quant, output_quant=output_quantized, mask=mask5, padding=1, apply_input_quant=False)
                layer_list.append(layer)


        self.module_list = nn.ModuleList(layer_list)

    def forward(self, x): # this is the normal forward function with no verlilog included 
        for layer in self.module_list: 
            x = layer(x)
        return x 
import torch as th
from torch import nn
from torch.nn import ConstantPad2d
from braindecode.models.shallow_fbcsp import ShallowFBCSPNet
from braindecode.models.base import BaseModel

from braindecode.torch_ext.functions import identity
from braindecode.models.util import to_dense_prediction_model
from eegnet import EEGNetv4
from torch.nn.functional import elu
from braindecode.torch_ext.modules import Expression
class ShallowEEGNet(BaseModel):
    """
    Wrapper for ShallowEEGNetModule
    """

    def __init__(self, in_chans, n_classes, input_time_length):
        self.in_chans = in_chans
        self.n_classes = n_classes
        self.input_time_length = input_time_length
   

    def create_network(self):
        return ShallowEEGNetModule(
            in_chans=self.in_chans,
            n_classes=self.n_classes,
            input_time_length=self.input_time_length
       
        )


class ShallowEEGNetModule(nn.Module):


    def __init__(self, in_chans, n_classes, input_time_length):
        super(ShallowEEGNetModule, self).__init__()
       

        shallow_model = ShallowFBCSPNet(in_chans=in_chans, n_classes=n_classes,
                                n_filters_time=40,filter_time_length =13,n_filters_spat=40,pool_time_length=35,pool_time_stride=7,
                                input_time_length=input_time_length,
                                final_conv_length=1).create_network()
        eegnet_model = EEGNetv4(in_chans=in_chans, n_classes=n_classes,input_time_length=input_time_length,final_conv_length=1).create_network()
        reduced_shallow_model = nn.Sequential()
        for name, module in shallow_model.named_children():
            if name == "conv_classifier":
                new_conv_layer = nn.Conv2d(
                    module.in_channels,
                    50,
                    kernel_size=module.kernel_size,
                    stride=module.stride,
                )
                reduced_shallow_model.add_module("shallow_final_conv", new_conv_layer)
                break
            reduced_shallow_model.add_module(name, module)

       
        reduced_eegnet_model= nn.Sequential()
        for name, module in eegnet_model.named_children():
            if name == "conv_classifier":
                new_conv_layer = nn.Conv2d(
                    module.in_channels,
                    50,
                    kernel_size=module.kernel_size,
                    stride=module.stride,
                )
                reduced_eegnet_model.add_module(
                    "eegnet_final_conv", new_conv_layer
                )
                reduced_eegnet_model.add_module("permute_back",Expression(lambda x: x.permute(0, 1, 3, 2)))
                break
            reduced_eegnet_model.add_module(name, module)

        to_dense_prediction_model(reduced_shallow_model)

        to_dense_prediction_model(reduced_eegnet_model)
        self.reduced_shallow_model = reduced_shallow_model

        self.reduced_eegnet_model = reduced_eegnet_model
        self.final_conv = nn.Conv2d(
            100, n_classes, kernel_size=(1, 1), stride=1
        )

    def create_network(self):
        return self

    def forward(self, x):
        #print(x.shape)
        shallow_out = self.reduced_shallow_model(x)

        eegnet_out = self.reduced_eegnet_model(x)
        
        n_diff_deep_shallow = shallow_out.size()[2]-eegnet_out.size()[2]
      
        if n_diff_deep_shallow < 0:
            shallow_out = ConstantPad2d((0, 0, -n_diff_deep_shallow, 0), 0)(
                shallow_out
            )
        elif n_diff_deep_shallow > 0:
            eegnet_out = ConstantPad2d((0, 0, n_diff_deep_shallow, 0), 0)(
                eegnet_out
            )

        merged_out = th.cat((shallow_out, eegnet_out), dim=1)
        linear_out = self.final_conv(merged_out)
        softmaxed = nn.LogSoftmax(dim=1)(linear_out)
        squeezed = softmaxed.squeeze(3)
        return squeezed

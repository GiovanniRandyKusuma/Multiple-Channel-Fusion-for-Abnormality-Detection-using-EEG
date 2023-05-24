import torch as th
from torch import nn
from torch.nn import ConstantPad2d

from braindecode.models.base import BaseModel
from braindecode.models.deep4 import Deep4Net
from braindecode.torch_ext.functions import identity
from braindecode.models.util import to_dense_prediction_model
from eegnet import EEGNetv4
from torch.nn.functional import elu
from braindecode.torch_ext.modules import Expression
class DeepEEGNet(BaseModel):
    """
    Wrapper for DeepEEGNetModule
    """

    def __init__(self, in_chans, n_classes, input_time_length):
        self.in_chans = in_chans
        self.n_classes = n_classes
        self.input_time_length = input_time_length

    def create_network(self):
        return DeepEEGNetModule(
            in_chans=self.in_chans,
            n_classes=self.n_classes,
            input_time_length=self.input_time_length,
        )


class DeepEEGNetModule(nn.Module):

    def __init__(self, in_chans, n_classes, input_time_length):
        super(DeepEEGNetModule, self).__init__()
       
        deep_model =  Deep4Net(in_chans, n_classes,
                         n_filters_time=25,
                         filter_time_length=5,
                         pool_time_length=2,
                         n_filters_spat=25,
                         input_time_length=input_time_length,
                         filter_length_2 = 5,
                         filter_length_3 = 5,
                         filter_length_4 = 5,
                         n_filters_2 = int(25 * 2),
                         n_filters_3 = int(25 * (2 ** 2.0)),
                         n_filters_4 = int(25 * (2 ** 3.0)),
                         final_conv_length=1,
                       ).create_network()

        
      

        eegnet_model = EEGNetv4(in_chans=in_chans, n_classes=n_classes,input_time_length=input_time_length,final_conv_length=1).create_network()
        reduced_deep_model = nn.Sequential()
        for name, module in deep_model.named_children():
            if name == "conv_classifier":
                new_conv_layer = nn.Conv2d(
                    module.in_channels,
                    50,
                    kernel_size=module.kernel_size,
                    stride=module.stride,
                )
                reduced_deep_model.add_module("deep_final_conv", new_conv_layer)
                break
            reduced_deep_model.add_module(name, module)

       
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

        to_dense_prediction_model(reduced_deep_model)

        to_dense_prediction_model(reduced_eegnet_model)
        self.reduced_deep_model = reduced_deep_model
    
        self.reduced_eegnet_model = reduced_eegnet_model
        self.final_conv = nn.Conv2d(
            100, n_classes, kernel_size=(1, 1), stride=1
        )

    def create_network(self):
        return self

    def forward(self, x):
        #print(x.shape)
        deep_out = self.reduced_deep_model(x)

        eegnet_out = self.reduced_eegnet_model(x)
        
        n_diff_deep_shallow = deep_out.size()[2]-eegnet_out.size()[2]
      
        if n_diff_deep_shallow < 0:
            deep_out = ConstantPad2d((0, 0, -n_diff_deep_shallow, 0), 0)(
                deep_out
            )
        elif n_diff_deep_shallow > 0:
            eegnet_out = ConstantPad2d((0, 0, n_diff_deep_shallow, 0), 0)(
                eegnet_out
            )

        merged_out = th.cat((deep_out, eegnet_out), dim=1)
        linear_out = self.final_conv(merged_out)
        softmaxed = nn.LogSoftmax(dim=1)(linear_out)
        squeezed = softmaxed.squeeze(3)
        return squeezed


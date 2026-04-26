import torch
import torch.nn as nn
# import layers.layers as layers
import WL_module as modules
import torch.nn.functional as F

class WL_BaseModel(nn.Module):
    def __init__(self, original_features_num, hidden_units, num_classes):
        """
        Build the model computation graph, until scores/values are returned at the end
        """
        super().__init__()
        
        # Second part
        self.fc_layers = nn.ModuleList()
        self.fc_layers.append(modules.FullyConnected(original_features_num, hidden_units))
        self.fc_layers.append(modules.FullyConnected(hidden_units, hidden_units))
        self.fc_layers.append(modules.FullyConnected(hidden_units, num_classes, activation_fn=None))

    def forward(self, input):
        x = input#.unsqueeze(0)
        # x = x.unsqueeze(0)
        # print("x ",x.shape)
        # scores = torch.tensor(0, device=input.device, dtype=x.dtype)

        # for i, block in enumerate(self.reg_blocks):

        #     x = block(x)

        #     if self.config.architecture.new_suffix:
        #         # use new suffix
        #         scores = self.fc_layers[i](layers.diag_offdiag_maxpool(x)) + scores

        # if not self.config.architecture.new_suffix:
        #     # old suffix
        #     # x = layers.diag_offdiag_maxpool(x)  # NxFxMxM -> Nx2F
        for fc in self.fc_layers:
            x = fc(x)
        # scores = x

            
        return F.log_softmax(x, dim=1)
        

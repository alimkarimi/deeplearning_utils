import torch


class FeatureExtractor(nn.Module):
    def __init__(self, model: nn.Module, layers: Iterable[str]): # instantiate class by inputing the model and 
        # the layers from which you want to extract an output
        super().__init__()
        self.model = model
        self.layers = layers
        # create a dictionary of empty tensors in format layer : empty tensor.
        self._features = {layer: torch.empty(0) for layer in layers}
        
        # note that a pytorch model created with nn.Module as the base class will have a named_modules() method.
        # this method is iterable and in each iteration provides a tuple of the layer name as defined in the class 
        # and the constructor used to create that layer (i.e, ReLU() or Conv2d(3, 10, kernel_size=(3, 3), stride=(1, 1))). 

        for layer_id in layers: # iterate through layers we input
            layer = dict([*self.model.named_modules()])[layer_id] # need layer to know which layer to register fwd hook on.
            # the above line creates a dictionary with keys as model layers. the keys are layers named in the model class.
            # for example, a simple CNN will have a self.conv1, self.relu1. conv1 and relu1 will then be in this 
            # dictionary's keys. The values in this dictionary will look like Conv2d(3, 10, kernel_size=(3, 3), stride=(1, 1))
            # or ReLU().
            print('extracting output from', layer) 
            layer.register_forward_hook(self.save_outputs_hook(layer_id)) 

    def save_outputs_hook(self, layer_id: str) -> Callable:
        def fn(_, module_in, output):
            self._features[layer_id] = output # note: layer_id is passed in via constructor - will be something like 'conv1', 'relu1'
            # when class is instantiated, _features is a dictionary of layer: empty tensor
        return fn

    def QKV_weightmatrix_extractor(self, named_param): 
        """Only valid for an attention based model"""
        
        #encoder.layers.encoder_layer_11.self_attention.in_proj_weight has the weights for linearly projecting keys in selfattn.
        embed_dim = self.model.hidden_dim
        in_proj_weight = model.state_dict()[named_param] # must be a named parameter from iterating through model.named_parameters()
        Wq, Wk, Wv = torch.split(in_proj_weight, [embed_dim, embed_dim, embed_dim]) 
        return Wq, Wk, Wv 

    def forward(self, x: torch.tensor) -> Dict[str, torch.tensor]:
        _ = self.model(x)
        return  self._features
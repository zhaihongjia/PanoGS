import torch
import torch.nn as nn
import numpy as np
from nerfstudio.field_components import encodings

def get_encoder(desired_resolution=512, num_components=16):
    '''
    https://github.com/nerfstudio-project/nerfstudio/blob/main/nerfstudio/field_components/encodings.py
    
    https://docs.nerf.studio/reference/api/field_components/encodings.html#nerfstudio.field_components.encodings.TriplaneEncoding
    Learned triplane encoding
    The encoding at [i,j,k] is an n dimensional vector corresponding to the element-wise product of the three n dimensional vectors at plane_coeff[i,j], plane_coeff[i,k], and plane_coeff[j,k].
    This allows for marginally more expressivity than the TensorVMEncoding, and each component is self standing and symmetrical, 
    unlike with VM decomposition where we needed one component with a vector along all the x, y, z directions for symmetry.
    This can be thought of as 3 planes of features perpendicular to the x, y, and z axes, respectively and intersecting at the origin, 
    and the encoding being the element-wise product of the element at the projection of [i, j, k] on these planes.
    The use for this is in representing a tensor decomp of a 4D embedding tensor: (x, y, z, feature_size)
    This will return a tensor of shape (bs:…, num_components)
    TriplaneEncoding(resolution: int = 32, num_components: int = 64, init_scale: float = 0.1, reduce: Literal['sum', 'product'] = 'sum')
    
    plane_coef: Float[Tensor, "3 num_components resolution resolution"]
    Args:
    resolution: Resolution of grid.
    num_components: The number of scalar triplanes to use (ie: output feature size)
    init_scale: The scale of the initial values of the planes
    product: Whether to use the element-wise product of the planes or the sum
    '''

    embed = encodings.TriplaneEncoding(resolution=desired_resolution, num_components=num_components, init_scale=0.1, reduce='sum',)
    out_dim = embed.get_out_dim()

    return embed, out_dim


class FeatureNet(nn.Module):
    def __init__(self, input_ch, dims):
        super(FeatureNet, self).__init__()
        self.input_ch = input_ch
        self.latent_dims = dims
        self.model = self.get_model()
    
    def forward(self, input_feat):
        return self.model(input_feat)
    
    def get_model(self):
        net =  []
        for i in range(len(self.latent_dims)):
            if i == 0:
                in_dim = self.input_ch
            else:
                in_dim = self.latent_dims[i-1]
            
            out_dim = self.latent_dims[i]
            
            net.append(nn.Linear(in_dim, out_dim, bias=False))

            if i != (len(self.latent_dims) - 1):
                net.append(nn.ReLU(inplace=True))

        return nn.Sequential(*nn.ModuleList(net))

class FeatureDecoder(nn.Module):
    def __init__(self, config):
        super(FeatureDecoder, self).__init__()
        self.config = config
        self.latent_dims = config['decoder']['latent_dims']
        self.bounding_box = torch.from_numpy(np.array(self.config['scene']['bound']))
        dim_max = (self.bounding_box[:,1] - self.bounding_box[:,0]).max()

        self.resolutions = []
        for res in self.config['decoder']['resolutions']:
            res_int = int(dim_max / res)
            self.resolutions.append(res_int)
    
        self.encodings = torch.nn.ModuleList()
        # multi-resolution
        # input of triplane should be in range [0, resolution]
        self.num_components = config['decoder']['num_components']
        for res in self.resolutions:
            encoding, embed_dim = get_encoder(desired_resolution=res, num_components=self.num_components)
            self.encodings.append(encoding)

            # we performan add for different resolution, so the embed_dim should be same 
            self.embed_dim = embed_dim 

        print('Parametric encoding resolutions: ', self.resolutions)
        print('Parametric encoding dimensions: ', self.embed_dim)

        self.feature_net = FeatureNet(input_ch=self.embed_dim, dims=self.latent_dims)
    
    def get_embed(self, pos):
        pos = (pos - self.bounding_box[:, 0]) / (self.bounding_box[:, 1] - self.bounding_box[:, 0])
        pos = pos.float().cuda()

        multi_feat = []

        for i in range(len(self.encodings)):
            embed = self.encodings[i](pos).detach()
            multi_feat.append(embed)

        return multi_feat
            
    def forward(self, pos, return_embed=False):
        pos = (pos - self.bounding_box[:, 0]) / (self.bounding_box[:, 1] - self.bounding_box[:, 0])
        pos = pos.float().cuda()

        # embed = self.encodings(pos).cuda()

        # for multi-resolution
        for i in range(len(self.encodings)):
            if i == 0:
                embed = self.encodings[i](pos).cuda()
                # print('embed: ', embed.shape)
            else:
                new_embed = self.encodings[i](pos).cuda()
                embed = embed + new_embed
        
        feature = self.feature_net(embed)     
        return feature
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from utils.graph import Graph
from utils.tgcn import TemporalGraphConvolution

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class Model(nn.Module):
    
    def __init__(self,
                 in_channels, 
                 num_class, 
                 graph_args,
                 edge_inportance_weighting, 
                 **kwargs):
        """Spatial Temporal Graph Convolutional Networks

        Args:
            in_channels (int): number of channels of the input data
            num_class (int): number of classes for the classifation task
            graph_args (dict): the arguments for building the graph
            edge_inportance_weighting (bool): if "True", adds a learnable importance weighting to the edges of the graph
            **kargs (optional): other parameters for graph convolution units 
            
        Shape:
            - Input: math: (N, in_channels, T_{in}, V_{in}, M_{in})
            - Output: math: (N, num_class)
            
            where
                N is the batch size.
                T_{in} is the length of the input sequence,
                V_{in} is the number of graph nodes, 
                M_{in} is the number of instance in a frame
        """
        super().__init__()

        # Load graph
        self.graph = Graph(**graph_args)
        A =torch.tensor(self.graph.A, dtype=torch.float32, requires_grad=False)
        self.register_buffer('A',A)
        
        # bulid networks
        spatial_kernel_size = A.shape[0]
        temporal_kernel_size = 9
        kernel_size = (spatial_kernel_size, temporal_kernel_size)
        self.data_bn = nn.BatchNorm1d(in_channels, A.size[1])
        kwargs0 = {k: v for k,v in kwargs.items() if k !='dropout'}
        self.st_gcn_networks = nn.ModuleList((
            ST_GCN(in_channels, 64, kernel_size, 1, **kwargs),
            ST_GCN(64, 64, kernel_size, 1, **kwargs),
            ST_GCN(64, 64, kernel_size, 1, **kwargs),
            ST_GCN(64, 64, kernel_size, 1, **kwargs),
            ST_GCN(64, 128, kernel_size, 1, **kwargs),
            ST_GCN(128, 128, kernel_size, 1, **kwargs),
            ST_GCN(128, 256, kernel_size, 1, **kwargs),
            ST_GCN(256, 256, kernel_size, 1, **kwargs),
            ST_GCN(256, 256, kernel_size, 1, **kwargs),
        ))
        
        # initialize parameters for edge importance weighting
        if edge_inportance_weighting:
            self.edge_importance = nn.ParameterList([
                nn.Parameter(torch.ones(self.A.size()))
                for i in self.st_gcn_networks
            ])
        else:
            self.edge_importance = [1] * len(self.st_gcn_networks)
            
        # FCN for prediction
        self.fcn = nn.Conv2d(256, num_class, kernel_size=1)
        
    def forward(self, x):
        N, C, T, V, M = x.size()
        x = x.permute(0,4,3,2,1).contiguous()
        x = x.view(N*M, V*C, T)
        x = self.data_bn(x)
        x = x.view(N, M, V, C, T)
        x = x.permute(0,1,3,4,2).contiguous()
        x = x.view(N*M, C, T, V)
        
        # forward
        for gcn, importance in zip(self.st_gcn_networks, self.edge_importance):
            x, _ = gcn(x, self.A * importance)
            
        # global pooling
        x = F.avg_pool2d(x, x.size()[2:])
        x = x.view(N,M,-1,1,1).mean(dim=1)
        
        # prediction
        x = self.fcn(x)
        x = x.view(x.size(0), -1)
        
        return x
        
        
    def feature_extraction(self, x):
        # data normalization
        N, C, T, V, M = x.size()
        x = x.permute(0,4,3,2,1).contiguous()
        x = x.view(N*M, V*C, T)
        x = self.data_bn(x)
        x = x.view(N, M, V, C, T)
        x = x.permute(0,1,3,4,2).contiguous()
        x = x.view(N*M, C, T, V)
        
        # forward
        for gcn, importance in zip(self.st_gcn_networks, self.edge_importance):
            x, _ = gcn(x, self.A * importance)
            
        _, c, t, v = x.size()
        feature = x.view(N, M, c, t, v).permute(0,2,3,4,1)
        
        # prediction
        x = self.fcn(x)
        output = x.view(N, M, -1, t, v).permute(0,2,3,4,1)
        
        return output, feature
    
    


class ST_GCN(nn.Module):
    
    def __init__(self, 
                 in_channels, 
                 out_channels,
                 kernel_size, 
                 stride=1,
                 dropout=0, 
                 residual=True):
        """Implementation of a spatial temporal graph convolution

        Args:
            in_channels (int): number of channels in the input data sequence
            out_channels (int): number of channels in the output data sequence
            kernel_size (tuple): size of the temporal conv and graph conv kernels
            stride (int, optional): stride of the temproal conv. Defaults to 1.
            dropout (int, optional): dropout rate of the final output. Defaults to 0.
            residual (bool, optional): applying a residual mechanism. Defaults to True.
            
        Shape:
            - Input[0]: graph sequence: math: (N, in_channels, T_{in}, V) 
            - Input[1]: graph adjacency matrix: math: (K, V, V) format
            - Output[0]: graph sequence: math: (N, out_channels, T_{out}, V)
            - Output[1]: graph adjacency matrix for output data: math: (K, V, V)
            
            where
                N is the batch size, 
                K is the spatial kernel size, 
                T_{in}/_{out} is the length of input/output sequence,
                V is the number of vertex (graph node)
        """
        super().__init__()
        
        assert len(kernel_size) == 2
        assert kernel_size[0]%2 == 1
        padding = ((kernel_size[0]-1)//2, 0)
        
        self.gcn = TemporalGraphConvolution(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size[1],
        )
        
        self.tcn = nn.Sequential(
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                out_channels,
                out_channels,
                (kernel_size[0], 1),
                (stride, 1),
                padding                
            ),
            nn.BatchNorm2d(out_channels),
            nn.Dropout(dropout, inplace=True)
        )
        
        if not residual:
            self.residual = lambda x:0
            
        elif (in_channels == out_channels) and (stride == 1):
            self.residual = lambda x: x

        else:
            self.residual = nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=1,
                    stride=(stride, 1)),
                nn.BatchNorm2d(out_channels),
            )

        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, x, A):
        res = self.residual(x)
        x, A = self.gcn(x, A)
        x = self.tcn(x) + res
        
        return self.relu(x), A

if __name__ == "__init__":
    model = Model()
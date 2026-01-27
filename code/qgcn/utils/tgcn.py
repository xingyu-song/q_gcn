import torch
import torch.nn as nn

class TemporalGraphConvolution(nn.Module):
    """Implementation of a graph convolution

    Args:
        nn (_type_): _description_
    """
    def __init__(self, 
                 in_channels, 
                 out_channels,
                 kernel_size,
                 t_kernel_size=1, 
                 t_stride=1, 
                 t_padding=1,
                 t_dilation=1, 
                 bias=True):
        """_summary_

        Args:
            in_channels (int): number of channels in the input data sequence
            out_channels (int): number of channels in the output data sequence
            kernel_size (int): size of graph convolution kernel
            t_kernel_size (int, optional): size of temporal convolution kernel. Defaults to 1.
            t_stride (int, optional): stride of temporal convolution. Defaults to 1.
            t_padding (int, optional): temporal zero-padding added to both sides of the input. Defaults to 1.
            t_dilation (int, optional): spacing between temporal kernel elements. Defaults to 1.
            bias (bool, optional): if "True", add a learnable bias to the output. Defaults to True.
            
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
        
        self.kernel_size = kernel_size
        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels*self.kernel_size,
            kernel_size=(t_kernel_size, 1),
            padding=(t_padding,0),
            stride=(t_stride, 1),
            dilation=(t_dilation, 1),
            bias=bias
        )
        
    def forward(self, x, A):
        assert A.size(0) == self.kernel_size
        
        x = self.conv(x)
        
        n, kc, t, v = x.size()
        x = x.view(n, self.kernel_size, kc//self.kernel_size, t, v)
        x = torch.einsum('nkctv,kvw->nctw', (x, A))
        
        return x.contigous(), A
        
        
        
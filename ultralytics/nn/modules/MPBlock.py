import torch
import torch.nn as nn

def autopad(k, p=None, d=1):  # kernel, padding, dilation
    """Pad to 'same' shape outputs."""
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]  # actual kernel-size
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p

class Conv(nn.Module):
    """Standard convolution with args(ch_in, ch_out, kernel, stride, padding, groups, dilation, activation)."""

    default_act = nn.SiLU()  # default activation

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        """Initialize Conv layer with given arguments including activation."""
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        """Apply convolution, batch normalization and activation to input tensor."""
        return self.act(self.bn(self.conv(x)))

    def forward_fuse(self, x):
        """Perform transposed convolution of 2D data."""
        return self.act(self.conv(x))

class MPBlock(nn.Module):

    def __init__(self, c1, c2, k=3, s=1, p=1, g=1, d=1, act=True):

        super().__init__()#p = [(2, 0, 2, 0), (0, 2, 0, 2), (0, 2, 2, 0), (2, 0, 0, 2)]
        self.pad1 = nn.ZeroPad2d(padding=((p, p, p, p)))#(2,1,1,0)
        self.pad2 = nn.ZeroPad2d(padding=((0, 2*p, p, p)))#(1,2,1,0)
        self.pad3 = nn.ZeroPad2d(padding=((p, p, 0, 2*p)))#(1,0,2,1)
        self.pad4 = nn.ZeroPad2d(padding=((0, 2*p,0 , 2*p)))#(0,1,1,2)
        self.bias_conv1 = Conv(c1, c2 // 4, k, s=s, p=0)
        self.bias_conv2 = Conv(c1, c2 // 4, k, s=s, p=0)
        self.bias_conv3 = Conv(c1, c2 // 4, k, s=s, p=0)
        self.bias_conv4 = Conv(c1, c2 // 4, k, s=s, p=0)

    def forward(self, x):
        y1 = self.bias_conv1(self.pad1(x))
        y2 = self.bias_conv2(self.pad2(x))
        y3 = self.bias_conv3(self.pad3(x))
        y4 = self.bias_conv4(self.pad4(x))
        return torch.cat([y1, y2, y3, y4], dim=1)

class MPCBottleneck(nn.Module):

    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):
        super().__init__()
        self.c_ = int(c2 * e)  # hidden channels
        self.pad1 = nn.ZeroPad2d(padding=((1, 1, 1, 1)))#(2,1,1,0)
        self.pad2 = nn.ZeroPad2d(padding=((0, 2, 1, 1)))#(1,2,1,0)
        self.pad3 = nn.ZeroPad2d(padding=((1, 1, 0, 2)))#(1,0,2,1)
        self.pad4 = nn.ZeroPad2d(padding=((0, 2, 0, 2)))#(0,1,1,2)

        self.conv = Conv(c1, self.c_, k[0], s=1)
        self.biasconv1 = Conv(self.c_, c2//4, k[1], s=1,p=0, g=g)
        self.biasconv2 = Conv(self.c_, c2//4, k[1], s=1,p=0, g=g)
        self.biasconv3 = Conv(self.c_, c2//4, k[1], s=1,p=0, g=g)
        self.biasconv4 = Conv(self.c_, c2//4, k[1], s=1,p=0, g=g)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        x1 = self.conv(x)
        y1 = self.biasconv1(self.pad1(x1)) 
        y2 = self.biasconv2(self.pad2(x1)) 
        y3 = self.biasconv3(self.pad3(x1))
        y4 = self.biasconv4(self.pad4(x1))
        result = x + torch.cat([y1, y2, y3, y4], dim=1) if self.add else torch.cat([y1, y2, y3, y4], dim=1)
        return result

class C3(nn.Module):
    """CSP Bottleneck with 3 convolutions."""

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        """Initialize the CSP Bottleneck with given channels, number, shortcut, groups, and expansion values."""
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.cv3 = Conv(2 * c_, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.Sequential(*(MPCBottleneck(c_, c_, shortcut, g, k=((1, 1), (3, 3)), e=1.0) for _ in range(n)))

    def forward(self, x):
        """Forward pass through the CSP bottleneck with 2 convolutions."""
        # s = self.m(self.cv1(x))
        return self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), 1))

class C3k(C3):
    """C3k is a CSP bottleneck module with customizable kernel sizes for feature extraction in neural networks."""

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5, k=3):
        """Initializes the C3k module with specified channels, number of layers, and configurations."""
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)  # hidden channels
        # self.m = nn.Sequential(*(RepBottleneck(c_, c_, shortcut, g, k=(k, k), e=1.0) for _ in range(n)))
        self.m = nn.Sequential(*(MPCBottleneck(c_, c_, shortcut, g, k=(k, k), e=1.0) for _ in range(n)))

class C2f(nn.Module):
    """Faster Implementation of CSP Bottleneck with 2 convolutions."""

    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        """Initializes a CSP bottleneck with 2 convolutions and n Bottleneck blocks for faster processing."""
        super().__init__()
        self.c = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.ModuleList(MPCBottleneck(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n))

    def forward(self, x):
        """Forward pass through C2f layer."""
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))

    def forward_split(self, x):
        """Forward pass using split() instead of chunk()."""
        y = list(self.cv1(x).split((self.c, self.c), 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))

class MPCBlock(C2f):
    """Faster Implementation of CSP Bottleneck with 2 convolutions."""

    def __init__(self, c1, c2, n=1, c3k=False, e=0.5, g=1, shortcut=True):
        """Initializes the C3k2 module, a faster CSP Bottleneck with 2 convolutions and optional C3k blocks."""
        super().__init__(c1, c2, n, shortcut, g, e)
        self.m = nn.ModuleList(
            C3k(self.c, self.c, 2, shortcut, g) if c3k else MPCBottleneck(self.c, self.c, shortcut, g) for _ in range(n)
        )


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    # 加载 starnet_s1 模型
    model = PConv(3, 32, k=3,s=1).to(device)
    input_tensor = torch.randn(1, 3, 64, 64).to(device)
    output = model(input_tensor)
    print(f"Output shape: {output.shape}")
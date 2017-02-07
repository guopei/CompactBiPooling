require 'nn'
require 'libcudafft'

tcbp = {}
tcbp.version = 1.0-1

-- support modules
torch.include('tcbp', 'CompactBiPooling.lua')
torch.include('tcbp', 'SignedSquareRoot.lua')
torch.include('tcbp', 'L2Normalization.lua')

-- prevent likely name conflicts
nn.tcbp = tcbp

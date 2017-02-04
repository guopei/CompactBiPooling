require 'nn'
require 'libcudafft'

tcbp = {}
tcbp.version = 1.0-1

unpack = unpack or table.unpack

-- support modules
torch.include('tcbp', 'CompactBiPooling.lua')
torch.include('tcbp', 'SignedSquareRoot.lua')
torch.include('tcbp', 'BackLayer.lua')

-- prevent likely name conflicts
nn.tcbp = tcbp

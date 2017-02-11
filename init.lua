require 'nn'
require 'libcudafft'

tcbp = {}
tcbp.version = 1.0-1

-- support modules
torch.include('tcbp', 'cbp.lua')
torch.include('tcbp', 'ssr.lua')

-- prevent likely name conflicts
nn.tcbp = tcbp

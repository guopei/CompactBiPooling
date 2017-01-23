require 'nn'

-- create global rnn table:
tcbp = {}
tcbp.version = 1

unpack = unpack or table.unpack

-- for testing:
-- torch.include('cbp', 'test.lua')

-- support modules
torch.include('tcbp', 'CompactBiPooling.lua')
torch.include('tcbp', 'SignedSquareRoot.lua')

-- prevent likely name conflicts
nn.tcbp = tcbp

signal = require 'signal'

in_dim  = 1024
out_dim = 8192

x = torch.rand(in_dim)
y = torch.rand(in_dim)
a = torch.rand(in_dim)
b = torch.rand(in_dim)

xy = torch.ger(x, y)
ab = torch.ger(a, b)

out_p = xy:dot(ab)

h1 = torch.Tensor(in_dim):uniform(0, out_dim):ceil()
h2 = torch.Tensor(in_dim):uniform(0, out_dim):ceil()
s1 = torch.Tensor(in_dim):uniform(0,2):floor():mul(2):add(-1)
s2 = torch.Tensor(in_dim):uniform(0,2):floor():mul(2):add(-1)


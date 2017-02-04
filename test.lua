require 'nn'
require 'tcbp'
require 'cunn'
require 'cudnn'

local mytest = torch.TestSuite()
local tester = torch.Tester()

function mytest.test_forward()
    local use_cuda = true
    torch.manualSeed(0)

    local batch     = 8
    local height    = 6
    local width     = 6
    local in_dim    = 64
    local out_size  = 1024
    local test_dim  = 3
    local iter_num  = 50
    local precision = 0.01
    local x         = torch.rand(batch, in_dim, height, width)
    local y         = torch.rand(batch, in_dim, height, width)
    local w         = torch.rand(batch, in_dim, height, width)
    local z         = torch.rand(batch, in_dim, height, width)

    if use_cuda then
        x  = x:cuda()
        y  = y:cuda()
        w  = w:cuda()
        z  = z:cuda()
    end

    local part = nn.ParallelTable():add(nn.Reshape(in_dim, height*width, true)):add(nn.Reshape(in_dim, height*width, true))
    local bp   = nn.Sequential():add(part):add(nn.MM(false, true)):add(nn.Reshape(in_dim * in_dim, true))

    if use_cuda then bp = bp:cuda() end

    -- (Original) Bilinear Pooling results
    local bp_x = bp({x, y}):clone()
    local bp_y = bp({w, z}):clone()

    local bp_mm = bp_x[test_dim]:dot(bp_y[test_dim])

    -- (Another way to compute) Bilinear Pooling results
    local xy = torch.zeros(batch, in_dim * in_dim)
    local wz = torch.zeros(batch, in_dim * in_dim)

    if use_cuda then
        xy  = xy:cuda()
        wz  = wz:cuda()
    end

    local flatx = x:permute(1,3,4,2):contiguous():view(batch, -1, in_dim)
    local flaty = y:permute(1,3,4,2):contiguous():view(batch, -1, in_dim)
    local flatw = w:permute(1,3,4,2):contiguous():view(batch, -1, in_dim)
    local flatz = z:permute(1,3,4,2):contiguous():view(batch, -1, in_dim)

    for i = 1, batch do
        for j = 1, height*width do
            xy[i]:add(torch.ger(flatx[i][j], flaty[i][j]):view(-1))
            wz[i]:add(torch.ger(flatw[i][j], flatz[i][j]):view(-1))
        end
    end

    local bp_outer = xy[test_dim]:dot(wz[test_dim])

    -- timer = torch.Timer()
    
    -- compact bilinear pooling results
    local cbp_avg = torch.zeros(1)
    for i = 1, iter_num do
        local cbp   = nn.ComBiPooling(out_size)
        if use_cuda then cbp = cbp:cuda() end

        local cbp_x = cbp:forward({x, y}):clone()
        local cbp_y = cbp:forward({w, z}):clone()

        local cbp_one = cbp_x[test_dim]:dot(cbp_y[test_dim])
        cbp_avg = cbp_avg:add(cbp_one)
    end

    -- according to the paper, the inner product
    -- of cbp should match bp.
    cbp_avg = cbp_avg / iter_num
    local diff = (cbp_avg / bp_mm):float()[1] - 1
    tester:assertlt( diff , precision, 'cbp inner product not match bp result')
    tester:eq(bp_mm, bp_outer, 'two bp methods not equal')
    
    --print('average time is: ' .. timer:time().real / iter_num .. 's')
end

function mytest.test_backward()
    local batch     = 2
    local in_dim    = 6
    local height    = 4
    local width     = 4
    local out_size  = 12
    local minval    = -2
    local maxval    = 2
    local jac       = nn.Jacobian
    local input     = torch.rand(batch, in_dim, height, width)
    local cbp       = nn.ComBiPooling(out_size, true)
    
    
    -- because we are using cuda fft, only float precision
    -- so a bigger pertubation.
    local perturbation = 1e-4
    local precision    = 5e-2
    
    -- use cuda
    local use_cuda  = true
    if use_cuda then 
        input = input:cuda()
        cbp   = cbp:cuda() 
    end

    -- jacobian test error
    local err = jac.testJacobian(cbp, input, minval, maxval, perturbation)
    tester:assertlt(err , precision, 'jacobian test fails')
end

function mytest.test_signed_square_root()
    local input  = torch.Tensor({4, 9, -16, -25, 0})
    local target = torch.Tensor({2, 3, -4, -5, 0})
    
    local precision = 1e-5
    
    local ssr = nn.SignedSquareRoot()
    local output = ssr:forward(input)
    tester:eq(output, target, 'SignedSquareRoot layer forward function wrong')
    
    local jac = nn.Jacobian
    local err = jac.testJacobian(ssr, input)
    tester:assertlt(err, precision, 'SignedSquareRoot layer gradient checking fail')
end

tester:add(mytest)
tester:run()

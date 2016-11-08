require 'nn'
require 'cunn'

local cbptest = torch.TestSuite()
local precision = 1e-5
local debug = false

function cbptest.testPsi()
    local batch   = 32
    local dim_1   = 25
    local dim_2   = 25
    local height  = 16
    local width   = 16
    local outSize = 64
    local iters   = 100
    local x       = torch.rand(batch,dim_1,height,width):cuda()
    local y       = torch.rand(batch,dim_2,height,width):cuda()
    local outSum  = 0
    
    local model   = nn.CompactBilinearPooling(outSize):cuda()
    for i=1,iters do
        model:forward({x,y})
        outSum = outSum + model.psi[1][1]:dot(model.psi[2][1])
    end
    local xy = x:permute(1,3,4,2):view(-1,dim_1)[1]:dot(y:permute(1,3,4,2):view(-1,dim_2)[1])
    local diff = math.abs(outSum/iters - xy)
    assert(diff / xy < .1, 'error_ratio='..diff / xy..', E[<phi(x,h,s),phi(y,h,s)>]=<x,y>')
end

function cbptest.testConv()
   local x = torch.CudaTensor{{1,2,3},{2,3,4}}
   local y = torch.CudaTensor{{1,1,1},{2,2,2}}
   local c = nn.CompactBilinearPooling(x:size(2))
   local ans = torch.CudaTensor{{6,6,6},{18,18,18}}
   local output = torch.CudaTensor()
   output = c:conv(x,y)  -- cuda only
   ans:add(-output)
   assert(ans:norm() < precision, ans:norm())

   local x = torch.CudaTensor{{1,2,3,1,1},{2,3,4,1,1}}
   local y = torch.CudaTensor{{1,1,1,1,1},{2,2,2,1,1}}
   local c = nn.CompactBilinearPooling(x:size(2))
   local ans = torch.CudaTensor{{8,8,8,8,8},{17.5,17.5,17.75,17.75,17.5}}
   output = c:conv(x,y)
   ans:add(-output)
   assert(ans:norm() < precision, ans:norm())
end

function cbptest.testForward()
    local batch   = 32
    local dim_1   = 25
    local dim_2   = 25
    local height  = 16
    local width   = 16
    local outSize = 64
    local iters   = 100
    local x       = torch.rand(batch,dim_1,height,width):cuda()
    local y       = torch.rand(batch,dim_2,height,width):cuda()
    local z       = torch.rand(batch,dim_1,height,width):cuda()
    local w       = torch.rand(batch,dim_2,height,width):cuda()
    local cbp     = nn.CompactBilinearPooling(outSize):cuda()
    local part    = nn.ParalleTable():add(nn.Reshape(height*width, true))
    local bp      = nn.Sequential():add(part):add(nn.MM(false, true)):cuda()
    
    # Compact Bilinear Pooling results
    local cbp_xy = cbp({x, y})
    local cbp_zw = cbp({z, w})

    # (Original) Bilinear Pooling results
    local bp_xy = bp({x, y})
    local bp_zw = bp({z, w})
    
    local cbp_kernel = torch.mm(cbp_xy, cbp_zw):sum(2)
    local bp_kernel  = torch.mm(bp_xy, bp_zw):sum(2)
    local ratio      = torch.cdiv(cbp_kernel, bp_kernel)
    print("ratio between Compact Bilinear Pooling (CBP) and Bilinear Pooling (BP):")
    print(ratio)
    assert(torch.all(torch.abs(ratio - 1) < precision))
    print("Passed.")
    
end

mytester = torch.Tester()
mytester:add(cbptest)
math.randomseed(os.time())
mytester:run(tests)


-- loading C library
require 'libcudafft'

local ComBiPooling, parent = torch.class('nn.ComBiPooling', 'nn.Module')

function ComBiPooling:__init(output_size, homo)
    assert(output_size and output_size >= 1, 'missing outputSize...')
    self.output_size = output_size
    self.homo        = homo or false
    self:initVar()
    
end

function ComBiPooling:initVar()
    self.flat_input = torch.Tensor()
    self.hash_input = torch.Tensor()
    self.rand_h_1   = torch.Tensor()
    self.rand_h_2   = torch.Tensor()
    self.rand_s_1   = torch.Tensor()
    self.rand_s_2   = torch.Tensor()
end

-- generate random vectors h1, h2, s1, s2.
-- according to "Algorithm 2 Tensor Sketch Projection" step 1.
function ComBiPooling:genRand(size_1, size_2)
    cutorch.manualSeed(0)
    self.rand_h_1 = self.rand_h_1:resize(size_1):uniform(0,self.output_size):ceil():long()
    self.rand_h_2 = self.rand_h_2:resize(size_2):uniform(0,self.output_size):ceil():long()
    self.rand_s_1 = self.rand_s_1:resize(size_1):uniform(0,2):floor():mul(2):add(-1)
    self.rand_s_2 = self.rand_s_2:resize(size_2):uniform(0,2):floor():mul(2):add(-1)
end

function ComBiPooling:getHashInput()
    self.hash_input:zero()
    self.hash_input[1]:indexAdd(2,self.rand_h_1,
        torch.cmul(self.rand_s_1:repeatTensor(self.flat_size,1),self.flat_input[1]))
    self.hash_input[2]:indexAdd(2,self.rand_h_2,
        torch.cmul(self.rand_s_2:repeatTensor(self.flat_size,1),self.flat_input[2]))
end

function ComBiPooling:checkInput(input)
    if self.homo then
        -- if only one input
        assert(1 == #input, string.format("expect 1 input but get %d...", #input))
        assert(4 == input[1]:nDimension(), 
            string.format("wrong input dimensions, required 4, but get %d", 
            input[1]:nDimension()))
    else
        -- if there are two inputs, #dim and size of each dim is examined.
        assert(2 == #input, string.format("expect 2 inputs but get %d...", #input))
        assert(4 == input[1]:nDimension() and 4 == input[2]:nDimension(), 
            string.format("wrong input dimensions, required (4, 4), but get (%d, %d)", 
            input[1]:nDimension(), input[2]:nDimension()))
        
        for dim = 1, 4 do
            if dim ~= 2 then
                assert(input[1]:size(dim) == input[2]:size(dim), 
                    string.format("input size mismatch, dim %d: %d vs %d", 
                    dim, input[1]:size(dim), input[2]:size(dim)))
            end
        end
    end
end

function ComBiPooling:fftMul(x, y)
    local prod = torch.zeros(x:size()):cuda()
    
    for i = 1, x:size(1) do
        local x1 = x[i][1][1]:select(2,1)
        local x2 = x[i][1][1]:select(2,2)
        local y1 = y[i][1][1]:select(2,1)
        local y2 = y[i][1][1]:select(2,2)
    
        local real = torch.cmul(x1, y1) - torch.cmul(x2, y2)
        local imag = torch.cmul(x1, y2) + torch.cmul(x2, y1)
    
        prod[i]:copy(torch.cat(real, imag, 2))
    end
    
    return prod
end

function ComBiPooling:fft1d(input, output)
    local nSamples = input:size(1)
    local nPlanes = input:size(2)
    local N = input:size(3)
    local M = input:size(4)
    input:resize(nSamples*nPlanes*N, M)
    output:resize(nSamples*nPlanes*N, M/2+1, 2)
    -- calling C function
    cudafft.fft1d_r2c(input, output)
    input:resize(nSamples, nPlanes, N, M)
    output:resize(nSamples, nPlanes, N, M/2+1, 2)
end

function ComBiPooling:ifft1d(input, output)
    local nSamples = output:size(1)
    local nPlanes = output:size(2)
    local N = output:size(3)
    local M = output:size(4)
    input:resize(nSamples*nPlanes*N, M/2+1, 2)
    output:resize(nSamples*nPlanes*N, M)
    -- calling C function
    cudafft.fft1d_c2r(input,output)
    output:div(M)
    input:resize(nSamples, nPlanes, N, M/2+1, 2)
    output:resize(nSamples, nPlanes, N, M)
end

function ComBiPooling:conv(x,y)
    local batchSize = x:size(1)
    local dim = x:size(2)
    local function makeComplex(x,y)
        self.x_ = self.x_ or torch.CudaTensor()
        self.x_:resize(x:size(1),1,1,x:size(2),2):zero()
        self.x_[{{},{1},{1},{},{1}}]:copy(x)
        self.y_ = self.y_ or torch.CudaTensor()
        self.y_:resize(y:size(1),1,1,y:size(2),2):zero()
        self.y_[{{},{1},{1},{},{1}}]:copy(y)
    end
    
    makeComplex(x,y)
    self.fft_x = self.fft_x or torch.CudaTensor(batchSize,1,1,dim,2)
    self.fft_y = self.fft_y or torch.CudaTensor(batchSize,1,1,dim,2)
    
    local output = output or torch.CudaTensor()
    output:resize(batchSize,1,1,dim*2)
    
    self:fft1d(self.x_:view(x:size(1),1,1,-1), self.fft_x)
    self:fft1d(self.y_:view(y:size(1),1,1,-1), self.fft_y)
    
    local prod = self:fftMul(self.fft_x, self.fft_y)
    
    self:ifft1d(prod, output)
    
    return output:resize(batchSize,1,1,dim,2):select(2,1):select(2,1):select(3,1)
end

function ComBiPooling:updateOutput(input)
    
    -- wrap the input into a table if it's homoe
    if self.homo then
        self.input = type(input)=='table' and input or {input}
    else
        self.input = input
    end
    
    -- print(#self.input, type(self.input))
    self:checkInput(self.input)

    
    -- only generate new random vector at the very beginning.
    if 0 == self.rand_h_1:nElement() then 
        if self.homo then
            self:genRand(self.input[1]:size(2), self.input[1]:size(2))
        else
            self:genRand(self.input[1]:size(2), self.input[2]:size(2))
        end
    end
    
    -- convert the input from 4D to 2D and expose dimension 2 outside. 
    self.flat_size = self.flat_size or self.input[1]:size(1) * self.input[1]:size(3) * self.input[1]:size(4)
    self.flat_input:resize(2, self.flat_size, self.input[1]:size(2))
    for i = 1, #self.input do
        local new_input       = self.input[i]:permute(1,3,4,2):contiguous()
        self.flat_input[i]    = new_input:view(-1, self.input[i]:size(2))
    end
    
    if self.homo then self.flat_input[2] = self.flat_input[1]:clone() end
    
    -- get hash input as step 2
    self.hash_input:resize(2, self.flat_size, self.output_size)
    self:getHashInput()
    
    -- step 3
    self.flat_output    = self:conv(self.hash_input[1], self.hash_input[2])
    self.height         = self.input[1]:size(3)
    self.width          = self.input[1]:size(4)
    self.hw_size        = self.hw_size or self.input[1]:size(3) * self.input[1]:size(4)
    self.batch_size     = self.batch_size or self.input[1]:size(1)
    self.channel        = self.channel or self.input[1]:size(2)
    -- reshape output and sum pooling over dimension 2 and 3.
    self.output         = self.flat_output:reshape(self.batch_size, self.hw_size, self.output_size)
    self.output         = self.output:sum(2):squeeze():reshape(self.batch_size, self.output_size)
    
    return self.output
end


function ComBiPooling:updateGradInput(input, gradOutput)
 
    -- input: batch x channel x height x width
    -- gradOutput: batch x output_size
    local gradInput     = gradInput or {}
    local convResult     = convResult or {}
    -- repeatGradOut: flat_size x output_size
    local repeatGradOut = gradOutput:view(self.batch_size, self.output_size, 1):repeatTensor(1, 1, self.hw_size)
    repeatGradOut       = repeatGradOut:permute(1,3,2):reshape(self.flat_size, self.output_size)
    
    
    
    self.gradInput = self.gradInput or {}
    self.convResult     = self.convResult or {}
    
    for k = 1, 2 do
        self.gradInput[k]   = self.gradInput[k] or self.hash_input.new()
        -- self.gradInput[k]: flat_size x output_size
        self.gradInput[k]:resizeAs(self.flat_input[k]):zero()
        self.convResult[k]  = self.convResult[k] or repeatGradOut.new()
        -- self.convResult: flat_size x output_size
        self.convResult[k]:resizeAs(repeatGradOut)

        local index = torch.cat(torch.LongTensor{1}, 
            torch.linspace(self.output_size,2,self.output_size-1):long())
        
        -- self.hash_input: flat_size x output_size
        local reverse_input = self.hash_input[k]:index(
                2 ,index)
        
        -- self.convResult: flat_size x output_size
        self.convResult[k]  = self:conv(repeatGradOut, reverse_input)
        
    end
    
    for k = 1, 2 do
        
        -- self.rand_h_1: 1 x channel, range: [1, output_size]
        -- self.rand_s_1: 1 x channel, range: {1, -1}
        -- self.gradInput: flat_size, channel
        local k_bar = 3-k
        self.gradInput[k_bar]:index(self.convResult[k], 2, self['rand_h_' .. k_bar])
        self.gradInput[k_bar]:cmul(self['rand_s_' .. k_bar]:repeatTensor(self.flat_size,1))

        self.gradInput[k_bar] = self.gradInput[k_bar]:view(
            self.batch_size,self.height,self.width,-1):permute(1,4,2,3):contiguous()

    end

    if type(input)=='table' then 
        return self.gradInput
    else
        return self.gradInput[1] + self.gradInput[2]
    end
end
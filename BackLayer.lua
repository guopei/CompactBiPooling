signal = require 'signal'
local BackLayer, parent = torch.class('nn.BackLayer', 'nn.Module')


function BackLayer:__init(args)
    parent.__init(self)
end

function BackLayer:fftMul(x, y)
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

function BackLayer:fft1d(input, output)
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

function BackLayer:ifft1d(input, output)
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

function BackLayer:conv_cuda(x,y)
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

function BackLayer:fft_mul(x, y)
    local prod = torch.zeros(x:size())
    
    for i = 1, x:size(1) do
        local x1 = x[i]:select(2,1)
        local x2 = x[i]:select(2,2)
        local y1 = y[i]:select(2,1)
        local y2 = y[i]:select(2,2)
    
        local real = torch.cmul(x1, y1) - torch.cmul(x2, y2)
        local imag = torch.cmul(x1, y2) + torch.cmul(x2, y1)
    
        prod[i]:copy(torch.cat(real, imag, 2))
    end
    
    return prod
end

function BackLayer:batch_fft(input, output)
    
    output = torch.zeros(input:size(1), input:size(2), 2)
    for i = 1, input:size(1) do
        output[i] = signal.fft(input[i])
    end
    
    return output
end

function BackLayer:batch_ifft(input, output)
    
    output = torch.zeros(input:size())
    for i = 1, input:size(1) do
        output[i] = signal.ifft(input[i])
    end
    
    return output
end

function BackLayer:conv(x,y)
    
    conv_result = self:fft_mul(self:batch_fft(x), self:batch_fft(y))
    ret = self:batch_ifft(conv_result):select(3,1)
    
    return ret
end

function BackLayer:updateOutput(input)
    
    self.output = self:conv_cuda(input, input)
    
    print(self.output)
    print(self:conv(input, input))
    
    return self.output
end

function BackLayer:updateGradInput(input, gradOutput)
    
    local batch, dim = input:size(1), input:size(2)
    local index = torch.cat(torch.LongTensor{1}, torch.linspace(dim,2,dim-1):long())
    local reverse_input = input:index(2, index)
    local reverse_grad = gradOutput:index(2, torch.linspace(dim,1,dim):long())
    
    self.gradInput = self:conv_cuda(gradOutput, reverse_input)
    
    return self.gradInput * 2
end


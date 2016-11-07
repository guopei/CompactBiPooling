local CompactBilinearPooling, parent = torch.class('nn.CompactBilinearPooling', 'nn.Module')

require 'spectralnet'

function CompactBilinearPooling:__init(outputSize)
    assert(outputSize and outputSize >= 1, 'outputSize(>1) should be specified')
    self.outputSize = outputSize
    self.sum_pool = true
    self:reset()
end

function CompactBilinearPooling:reset()
    self.rand_h_1 = torch.Tensor()
    self.rand_h_2 = torch.Tensor()
    self.rand_s_1 = torch.Tensor()
    self.rand_s_2 = torch.Tensor()
    self.psi = torch.Tensor()
    self.gradInput = {}
    self.convBuf = torch.Tensor()
    self.inputFlatPermute = {}
    self.gradInputFlatPermute = {}
end

--[[
Generate vectors and sketch matrix for tensor count sketch
This is only done once during graph construction, and fixed during each
operation
]]--
function CompactBilinearPooling:sample()
   self.rand_h_1:random(self.outputSize)
   self.rand_h_2:random(self.outputSize)
   self.rand_s_1:random(0, 1):mul(2):add(-1)
   self.rand_s_2:random(0, 1):mul(2):add(-1)
end

--[[
PSI function as shown in "Algorithm 2 Tensor Sketch Projection"
in "Compact Bilinear Pooling" paper.
]]--
function CompactBilinearPooling:psiFunc()
   self.psi:zero()
          self.psi[1]:indexAdd(2,self.rand_h_1,torch.cmul(self.rand_s_1:repeatTensor(self.inputFlatPermute[1]:size(1),1),self.inputFlatPermute[1]))
    self.psi[2]:indexAdd(2,self.rand_h_2,torch.cmul(self.rand_s_2:repeatTensor(self.inputFlatPermute[2]:size(1),1),self.inputFlatPermute[2]))
end

function CompactBilinearPooling:conv(x,y)
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
   cufft.fft1d(self.x_:view(x:size(1),1,1,-1), self.fft_x)
   cufft.fft1d(self.y_:view(y:size(1),1,1,-1), self.fft_y)
   cufft.ifft1d(self.fft_x:cmul(self.fft_y), output)
   return output:resize(batchSize,1,1,dim,2):select(2,1):select(2,1):select(3,1)
end

function CompactBilinearPooling:updateOutput(input)
    -- input size: batch, channel, height, width
    -- two input should match in every dimension except channel.
    assert(#input == 2, 'input must be a pair of tensors')
    assert(input[1]:dim() == 4 and input[2]:dim() == 4, 'input must be 4D tensors')
    for d in #input[1]:dim() do
        if d ~= 2 then
            assert(input[1]:size(d) == input[2]:size(d), 'dimension %d shoud match'.format(d))
        end
    end
    
    if 0==#self.rand_h_1:size() then
        self.rand_h_1:resize(input[1]:size(2))
        self.rand_h_2:resize(input[2]:size(2))
        self.rand_s_1:resize(input[1]:size(2))
        self.rand_s_2:resize(input[2]:size(2))
        self:sample()  -- samples are fixed
    end

    for i = 1, 2 do
        local input_permute = self.input[i]:permute(1,3,4,2):contiguous()
        self.inputFlatPermute[i] = input_permute.view(-1, input_permute:size(input_permute:dim()))
    end
   
    self.psi:resize(2, self.inputFlatPermute[1]:size(1), self.outputSize)
    self:psiFunc()
    
    local output_flat = self:conv(self.psi[1], self.psi[2])
    output_shape = input[1]:size()
    output_shape[4] = self.outputSize
    self.output = output_flat:reshape(output_shape)
        
    if self.sum_pool then
        self.output:sum(2):sum(3):squeeze()
    end
    
    return self.output
end

function CompactBilinearPooling:updateGradInput(input, gradOutput)
    local batchSize = input[1]:size(1)
    local height = input[1]:size(3)
    local width = input[1]:size(4)
    self.gradInput = self.gradInput or {}
    self.gradOutFlatPermute = gradOutput:view(batchSize, 1, 1, self.outputSize):repeatTensor(1, height, width, 1)

    for k=1, 2 do
        self.gradInputFlatPermute[k] = self.gradInputFlatPermute[k] or self.inputFlatPermute[k].new()
        self.gradInputFlatPermute[k]:resizeAs(self.inputFlatPermute[k]):zero()
        self.convBuf = self.convBuf or self.gradOutFlatPermute.new()
        self.convBuf:resizeAs(self.gradOutFlatPermute)

        self.convBuf = self:conv(self.gradOutFlatPermute, self.psi[k%2+1]) -- need more thought
        if k==1 then
            self.gradInputFlatPermute[k]:index(self.convBuf, 2, self.rand_h_1)
            self.gradInputFlatPermute[k]:cmul(self.rand_s_1:repeatTensor(batchSize,1))
        else
            self.gradInputFlatPermute[k]:index(self.convBuf, 2, self.rand_h_2)
            self.gradInputFlatPermute[k]:cmul(self.rand_s_2:repeatTensor(batchSize,1))
        end
    end
   
   
    for i = 1, 2 do
        local gradInputView = self.gradInputFlatPermute[i]:view(batchSize, height, width, -1)
        self.gradInput[i] = gradInputView:permute(1,4,2,3):contiguous()
    end

    return self.gradInput
end

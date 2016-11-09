local CompactBilinearPooling, parent = torch.class('nn.CompactBilinearPooling', 'nn.Module')

require 'spectralnet'

function CompactBilinearPooling:__init(outputSize)
    assert(outputSize and outputSize >= 1, 'missing outputSize!')
    self.outputSize = outputSize
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

-- generate random vectors h1, h2, s1, s2.
-- according to "Algorithm 2 Tensor Sketch Projection" step 1.
function CompactBilinearPooling:sample()
   self.rand_h_1:uniform(0,self.outputSize):ceil()
   self.rand_h_2:uniform(0,self.outputSize):ceil()
   self.rand_s_1:uniform(0,2):floor():mul(2):add(-1)
   self.rand_s_2:uniform(0,2):floor():mul(2):add(-1)
end

-- generate psi function.
-- according to "Algorithm 2 Tensor Sketch Projection" step 2.
function CompactBilinearPooling:psiFunc()
   self.psi:zero()
   self.psi[1]:indexAdd(2,self.rand_h_1,torch.cmul(self.rand_s_1:repeatTensor(self.flatBatchSize,1),self.inputFlatPermute[1]))
   self.psi[2]:indexAdd(2,self.rand_h_2,torch.cmul(self.rand_s_2:repeatTensor(self.flatBatchSize,1),self.inputFlatPermute[2]))
end

-- compute phi(x).
-- according to "Algorithm 2 Tensor Sketch Projection" step 3.
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
    -- check two inputs
    assert(#input == 2, string.format("expect 2 inputs, get %d instead.", #input))
    -- check 4D inputs
    for i = 1, #input do
        assert(input[i]:dim() == 4, string.format("expect 4D input Tensor, get %dD instead for input %d.", input[i]:dim(), i))
    end
    -- check shape match for dimension 1, 3, 4
    for i = 1, input[1]:dim() do
        if i ~= 2 then
            assert(input[1]:size(i) == input[2]:size(i), 
                string.format("dimension %d size mismatch, %d vs %d.", i, input[1]:size(i), input[2]:size(i)))
        end
    end
    self.batchSize = input[1]:size(1)
    self.height    = input[1]:size(3)
    self.width     = input[1]:size(4)
    -- step 1 in algorithm 2.
    if 0==#self.rand_h_1:size() then
        self.rand_h_1:resize(input[1]:size(2))
        self.rand_h_2:resize(input[2]:size(2))
        self.rand_s_1:resize(input[1]:size(2))
        self.rand_s_2:resize(input[2]:size(2))
        self:sample()  -- samples are fixed
    end
    -- convert the input from 4D to 2D and expose dimension 2 outside. 
    for i = 1, #input do
        local input_permute      = input[i]:permute(1,3,4,2):contiguous()
        self.inputFlatPermute[i] = input_permute:view(-1, input[i]:size(2))
    end
    -- step 2 in algorithm 2.
    self.flatBatchSize = self.inputFlatPermute[1]:size(1)
    self.psi:resize(2, self.flatBatchSize, self.outputSize)
    self:psiFunc()
    -- step 3 in algorithm 2.
    local output_flat = self:conv(self.psi[1], self.psi[2])
    -- reshape output and sum pooling over dimension 2 and 3.
    self.output       = output_flat:reshape(self.batchSize, self.height, self.width, self.outputSize)
    self.output       = self.output:sum(2):sum(3):squeeze()
    
    return self.output
end

function CompactBilinearPooling:updateGradInput(input, gradOutput)
    local batchSize = self.inputFlatPermute[1]:size(1)
    self.gradInput = self.gradInput or {}

    for k=1, #input do
        self.gradInputFlatPermute[k] = self.gradInputFlatPermute[k] or self.inputFlatPermute[k].new()
        self.gradInputFlatPermute[k]:resizeAs(self.inputFlatPermute[k]):zero()
        self.convBuf = self.convBuf or gradOutput.new()
        self.convBuf:resizeAs(gradOutput)
        
        self.gradOutputRepeat = gradOutput:view(self.batchSize,1,1,self.outputSize):repeatTensor(1,self.height,self.width, 1):view(-1, self.outputSize):contiguous()

        self.convBuf = self:conv(self.gradOutputRepeat, self.psi[k%2+1])
        if k==1 then
            self.gradInputFlatPermute[k]:index(self.convBuf, 2, self.rand_h_1)
            self.gradInputFlatPermute[k]:cmul(self.rand_s_1:repeatTensor(batchSize,1))
        else
            self.gradInputFlatPermute[k]:index(self.convBuf, 2, self.rand_h_2)
            self.gradInputFlatPermute[k]:cmul(self.rand_s_2:repeatTensor(batchSize,1))
        end
    end
   
   
    for i = 1, #input do
        local gradInputView = self.gradInputFlatPermute[i]:view(self.batchSize, self.height, self.width, -1)
        self.gradInput[i]   = gradInputView:permute(1,4,2,3):contiguous()
    end

    return self.gradInput
end

require 'spectralnet'

local ComBiPooling, parent = torch.class('nn.ComBiPooling', 'nn.Module')

function ComBiPooling:__init(output_size)
    assert(output_size and output_size >= 1, 'missing outputSize...')
    self.output_size = output_size 
    self.flat_input = {}
    self.hash_input = torch.Tensor()
end

-- generate random vectors h1, h2, s1, s2.
-- according to "Algorithm 2 Tensor Sketch Projection" step 1.
function ComBiPooling:genRand(size_1, size_2)
    self.rand_h_1 = torch:Tensor(size_1):uniform(0,self.outputSize):ceil()
    self.rand_h_2 = torch:Tensor(size_2):uniform(0,self.outputSize):ceil()
    self.rand_s_1 = torch:Tensor(size_1):uniform(0,2):floor():mul(2):add(-1)
    self.rand_s_2 = torch:Tensor(size_2):uniform(0,2):floor():mul(2):add(-1)
end

function ComBiPooling:getHashInput()
    self.hash_input:zero()
    self.hash_input[1]:indexAdd(2,self.rand_h_1,
        torch.cmul(self.rand_s_1:repeatTensor(self.flat_size,1),self.flat_input[1]))
    self.hash_input[2]:indexAdd(2,self.rand_h_2,
        torch.cmul(self.rand_s_2:repeatTensor(self.flat_size,1), self.flat_input[2]))    
end

function ComBiPooling:checkInput(input)
    assert(2 == #input, string.format("expect 2 inputs but get %d...", #input))
    assert(4 == input[1].nDimension() and 4 == input[2].nDimension(), 
        string.format("wrong input dimensions, required (4, 4), but get (%d, %d)", 
        input[1].nDimension(), input[2].nDimension()))
    for dim = 1, 4 do
        if dim ~= 2 then
            assert(input[1]:size(dim) == input[2]:size(dim), 
                string.format("input size mismatch, dim %d: %d vs %d", 
                dim, input[1]:size(dim), input[2]:size(dim)))
        end
    end
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
   cufft.fft1d(self.x_:view(x:size(1),1,1,-1), self.fft_x)
   cufft.fft1d(self.y_:view(y:size(1),1,1,-1), self.fft_y)
   cufft.ifft1d(self.fft_x:cmul(self.fft_y), output)
   return output:resize(batchSize,1,1,dim,2):select(2,1):select(2,1):select(3,1)
end

function ComBiPooling:updateOutput(input)
    self:checkInput(input)
    self:genRand(input[1]:size(2), input[2]:size(2))
    
    -- convert the input from 4D to 2D and expose dimension 2 outside. 
    for i = 1, #input do
        local new_input       = input[i]:permute(1,3,4,2):contiguous()
        self.flat_input[i]    = new_input:view(-1, input[i]:size(2))
    end
    
    -- get hash input as step 2
    self.flat_size = self.flat_input[1]:size(1)
    self.hash_input:resize(2, self.flat_size, self.output_size)
    self:getHashInput()
    
    -- step 3
    self.flat_output = self:conv(self.hash_input[1], self.hash_input[2])
    -- reshape output and sum pooling over dimension 2 and 3.
    self.output       = self.flat_output:reshape(input[1]:size(1), input[1]:size(3), input[1]:size(4), self.output_size)
    self.output       = self.output:sum(2):sum(3):squeeze()
    
    return self.output
end
local L2Norm, parent = torch.class('nn.L2Norm', 'nn.Module')

function L2Norm:__init(eps)
    parent.__init(self)
    self.p = 2
    self.eps = eps or 1e-5
end

function L2Norm:updateOutput(input)
    assert(input:dim() <= 2, 'only 1d layer supported')
    local input_size = input:size()
    if input:dim() == 1 then
        input = input:view(1,-1)
    end

    self._output = self._output or input.new()
    self.psum = self.psum or input.new()
    self.power = self.power or input.new()

    self._output:resizeAs(input)

    self.psum:sum(torch.pow(input, 2), 2)
    self.power:pow(self.psum, -0.5):add(self.eps)

    self._output:cmul(input, self.power:view(-1,1):expandAs(input))
    self.output:view(self._output, input_size)
    
    return self.output
end

function L2Norm:updateGradInput(input, gradOutput)
    assert(input:dim() <= 2, 'only 1d layer supported')
    assert(gradOutput:dim() <= 2, 'only 1d layer supported')

    local input_size = input:size()
    if input:dim() == 1 then
        input = input:view(1,-1)
    end

    local n = input:size(1) -- batch size
    local d = input:size(2) -- dimensionality of vectors
    
    local scale = scale or input.new()
    self._gradInput = self._gradInput or input.new()
    scale:resize(n, 1)
    self._gradInput:resizeAs(input)
    
    for i = 1, n do
        scale[i] = self.output[i]:dot(gradOutput[i])
        self._gradInput[i]:cmul(self.output[i], scale[i]:expand(d))
        self._gradInput[i] = gradOutput[i] - self._gradInput[i]
        scale[i] = input[i]:dot(input[i])
        self._gradInput[i]:cmul(self._gradInput[i], torch.pow(scale[i]+self.eps, -0.5):expand(d))
    end
        
    self.gradInput = self._gradInput
    
    return self.gradInput
end
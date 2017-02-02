local SignedSquareRoot, parent = torch.class('nn.SignedSquareRoot', 'nn.Module')

function SignedSquareRoot:__init(args)
    parent.__init(self)
    self.module = nn.Sequential()
        :add(nn.Abs())
        :add(nn.Sqrt())
end

function SignedSquareRoot:updateOutput(input)
    
    self.output = self.module:forward(input)
    -- get sign for each input element
    self.sign = self.sign or input.new()
    self.sign:resizeAs(input)
    torch.sign(self.sign, input)
    self.output:cmul(self.sign)
    
    return self.output
end

function SignedSquareRoot:updateGradInput(input, gradOutput)
    local eps = 1e-11  -- to avoid gradient explosion
    torch.cmul(self.gradInput, gradOutput, 
        torch.pow(self.module:forward(input)+eps,-1)/2)
    
    return self.gradInput
end
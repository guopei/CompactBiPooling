require 'math'
local SignedSquareRoot, parent = torch.class('nn.SignedSquareRoot', 'nn.Module')

function SignedSquareRoot:__init(args)
    parent.__init(self)
    self.module = nn.Sequential()
        :add(nn.Abs())
        :add(nn.Sqrt())
end

function SignedSquareRoot:updateOutput(input)
    self.output = self.output or input.new()
    self.output_ = self.module:forward(input)
    -- get sign for each input element
    self.sign = self.sign or input.new()
    self.sign:resizeAs(input)
    torch.sign(self.sign, input)
    self.output:cmul(self.output_, self.sign)
    
    return self.output
end

function SignedSquareRoot:updateGradInput(input, gradOutput)
    self.gradInput = self.gradInput or inout.new()
    self.gradInput:cdiv(gradOutput, 
        self.output_ * 2)
    -- filtering out nan, avoid 1/0 caused number explosion
    self.gradInput[self.output_:eq(0)] = 0
    
    return self.gradInput
end

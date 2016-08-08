local tnt = require 'torchnet.env'
local argcheck = require 'argcheck'

local FPR95Meter = torch.class('tnt.FPR95Meter', 'tnt.Meter', tnt)

FPR95Meter.__init = argcheck{
   doc = [[
   Compute false positive rate at 95%
   For more information see http://www.mathworks.com/help/nnet/ref/roc.html
   ]],
   {name="self", type="tnt.FPR95Meter"},
   call = function(self)
      self:reset()
   end
 }

FPR95Meter.reset = argcheck{
   {name="self", type="tnt.FPR95Meter"},
   call = function(self)
      self.targets = {}
      self.outputs = {}
      self.tpr, self.fpr = nil
   end
}

FPR95Meter.add = argcheck{
   {name="self", type="tnt.FPR95Meter"},
   {name="output", type="torch.*Tensor"},
   {name="target", type="torch.*Tensor"}, 
   call = function(self, output, target)
      if output:numel() ~= 1 then output = output:squeeze() end
      if target:numel() ~= 1 then target = target:squeeze() end
      table.insert(self.targets, target:float())
      table.insert(self.outputs, output:float())
   end
} 

local roc = function(targets, outputs)
  local L,I = torch.sort(outputs, 1, true)
  local labels = targets:index(1,I)
  local TPR = torch.cumsum(labels:gt(0):float()) / labels:gt(0):float():sum()
  local FPR = torch.cumsum(labels:lt(0):float()) / labels:lt(0):float():sum()
  return TPR, FPR
end


FPR95Meter.value = argcheck{
   {name="self", type="tnt.FPR95Meter"},
   {name="t", type="number", opt=true},
   call = function(self, t)
      local targets = torch.cat(self.targets)
      local outputs = torch.cat(self.outputs)
      self.tpr, self.fpr = roc(targets, outputs)
      local _,k = (self.tpr -0.95):abs():min(1)
      local FPR95 = self.fpr[k[1]]
      return FPR95
   end
}

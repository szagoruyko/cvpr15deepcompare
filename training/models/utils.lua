require 'nn'

local utils = {}

function utils.testModel(net)
   local input = torch.randn(4,2,64,64):typeAs(net.output)
   local output = net:forward(input)
   local gradInput = net:backward(input, output:clone())
   print{net.output, net.gradInput}
end

function utils.MSRinit(net)
   return net:apply(function(v)
      if torch.typename(v):find'Convolution' then
         local n = v.kW*v.kH*v.nOutputPlane
         v.weight:normal(0,math.sqrt(2/n))
         v.bias:zero()
      end
   end)
end

function utils.makeDataParallelTable(model, nGPU)
   if nGPU > 1 then
      local gpus = torch.range(1, nGPU):totable()
      local fastest, benchmark = cudnn.fastest, cudnn.benchmark

      local dpt = nn.DataParallelTable(1, true, true)
         :add(model, gpus)
         :threads(function()
            local cudnn = require 'cudnn'
            cudnn.fastest, cudnn.benchmark = fastest, benchmark
         end)
      dpt.gradInput = nil

      model = dpt:cuda()
   end
   return model
end

return utils

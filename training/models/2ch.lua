require 'nn'
local utils = paths.dofile'./utils.lua'

local model = nn.Sequential()

local fSize = {2, 96, 192, 256, 256, 1}

model:add(nn.SpatialConvolution(fSize[1], fSize[2], 7, 7, 3, 3))
model:add(nn.ReLU())
model:add(nn.SpatialMaxPooling(2,2,2,2))
model:add(nn.SpatialConvolution(fSize[2], fSize[3], 5, 5))
model:add(nn.ReLU())
model:add(nn.SpatialMaxPooling(2,2,2,2))
model:add(nn.SpatialConvolution(fSize[3], fSize[4], 3, 3))
model:add(nn.ReLU())
model:add(nn.View(-1):setNumInputDims(3))
model:add(nn.Linear(fSize[4], fSize[5]))
model:add(nn.ReLU())
model:add(nn.Linear(fSize[5], fSize[6]))

utils.MSRinit(model)
utils.testModel(model)

return model

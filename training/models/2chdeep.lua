require 'nn'
local utils = paths.dofile'./utils.lua'

local model = nn.Sequential()

local fSize = {2, 96, 192}
local featuresOut = fSize[3]*2*2

model:add(nn.SpatialConvolution(fSize[1], fSize[2], 4,4, 3,3))
model:add(nn.ReLU())
model:add(nn.SpatialConvolution(fSize[2], fSize[2], 3,3))
model:add(nn.ReLU())
model:add(nn.SpatialConvolution(fSize[2], fSize[2], 3,3))
model:add(nn.ReLU())
model:add(nn.SpatialConvolution(fSize[2], fSize[2], 3,3))
model:add(nn.ReLU())
model:add(nn.SpatialMaxPooling(2,2,2,2):ceil())
model:add(nn.SpatialConvolution(fSize[2], fSize[3], 3,3))
model:add(nn.ReLU())
model:add(nn.SpatialConvolution(fSize[3], fSize[3], 3,3))
model:add(nn.ReLU())
model:add(nn.SpatialConvolution(fSize[3], fSize[3], 3,3))
model:add(nn.ReLU())
model:add(nn.View(-1):setNumInputDims(3))
model:add(nn.Linear(featuresOut,featuresOut))
model:add(nn.ReLU())
model:add(nn.Linear(featuresOut,1))

utils.MSRinit(model)
utils.testModel(model)

return model

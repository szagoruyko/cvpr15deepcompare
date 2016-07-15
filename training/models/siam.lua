require 'nn'
local utils = paths.dofile'./utils.lua'

local fSize = {1, 96, 192, 256}
local featuresOut = fSize[4]

local desc = nn.Sequential()
desc:add(nn.Reshape(1,64,64))
desc:add(nn.SpatialConvolution(fSize[1], fSize[2], 7, 7, 3, 3))
desc:add(nn.ReLU())
desc:add(nn.SpatialMaxPooling(2,2,2,2))
desc:add(nn.SpatialConvolution(fSize[2], fSize[3], 5, 5))
desc:add(nn.ReLU())
desc:add(nn.SpatialMaxPooling(2,2,2,2))
desc:add(nn.SpatialConvolution(fSize[3], fSize[4], 3, 3))
desc:add(nn.ReLU())
desc:add(nn.View(-1):setNumInputDims(3))
desc:add(nn.Contiguous())

local siamese = nn.Parallel(2,2)
local siam = desc:clone()
desc:share(siam, 'weight', 'bias', 'gradWeight', 'gradBias')
siamese:add(desc)
siamese:add(siam)

local top = nn.Sequential()
top:add(nn.Linear(featuresOut*2, featuresOut*2))
top:add(nn.ReLU())
top:add(nn.Linear(featuresOut*2, 1))

local model = nn.Sequential()
   :add(siamese)
   :add(top)

utils.MSRinit(model)
utils.testModel(model)

return model

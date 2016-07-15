require 'nn'

local fSize = {2, 96, 192, 256, 256}
local featuresOut = fSize[4]

local model = nn.Sequential()
model:add(nn.SpatialConvolution(fSize[1], fSize[2], 7, 7, 3, 3))
model:add(nn.ReLU())
model:add(nn.SpatialMaxPooling(2,2,2,2))
model:add(nn.SpatialConvolution(fSize[2], fSize[3], 5, 5))
model:add(nn.ReLU())
model:add(nn.SpatialConvolution(fSize[3], fSize[4], 3, 3))
model:add(nn.ReLU())
model:add(nn.SpatialConvolution(fSize[4], fSize[5], 1, 1))
model:add(nn.ReLU())
model:add(nn.SpatialAveragePooling(2,2,2,2))
model:add(nn.Reshape(featuresOut*2*2))
model:add(nn.Linear(featuresOut*2*2,1))

return model

require 'nn'

local fSize = {1, 96, 192, 256}
local featuresOut = fSize[4]

local iris = nn.Sequential()
iris:add(nn.Reshape(1,64,64))
iris:add(nn.SpatialZeroPadding(-16, -16, -16, -16))
iris:add(nn.SpatialConvolution(fSize[1], fSize[2], 4, 4, 2, 2))
iris:add(nn.ReLU())
iris:add(nn.SpatialMaxPooling(2,2,2,2))
iris:add(nn.SpatialConvolution(fSize[2], fSize[3], 3, 3))
iris:add(nn.ReLU())
iris:add(nn.SpatialConvolution(fSize[3], fSize[4], 3, 3))
iris:add(nn.ReLU())
iris:add(nn.SpatialConvolution(fSize[4], fSize[4], 3, 3))
iris:add(nn.Reshape(featuresOut))

local fovea = nn.Sequential()
fovea:add(nn.Reshape(1,64,64))
fovea:add(nn.SpatialAveragePooling(2,2,2,2))
fovea:add(nn.SpatialConvolution(fSize[1], fSize[2], 4, 4, 2, 2))
fovea:add(nn.ReLU())
fovea:add(nn.SpatialMaxPooling(2,2,2,2))
fovea:add(nn.SpatialConvolution(fSize[2], fSize[3], 3, 3))
fovea:add(nn.ReLU())
fovea:add(nn.SpatialConvolution(fSize[3], fSize[4], 3, 3))
fovea:add(nn.ReLU())
fovea:add(nn.SpatialConvolution(fSize[4], fSize[4], 3, 3))
fovea:add(nn.Reshape(featuresOut))

local eye = nn.Concat(2,2)
eye:add(iris):add(fovea)

local siamese = nn.Parallel(2,2)
local siam = eye:clone()
eye:share(siam, 'weight', 'bias', 'gradWeight', 'gradBias')
siamese:add(eye):add(siam)

local model = nn.Sequential()
model:add(siamese)
model:add(nn.Linear(featuresOut*4, featuresOut*4))
model:add(nn.ReLU())
model:add(nn.Linear(featuresOut*4, featuresOut*4))
model:add(nn.ReLU())
model:add(nn.Linear(featuresOut*4, 1))

return model

require 'nn'
local utils = paths.dofile'./utils.lua'

local fSize = {2, 96, 192}
local featuresOut = fSize[3]*2*2

local fovea = nn.Sequential()
   :add(nn.Reshape(2,32,32))
   :add(nn.SpatialConvolution(fSize[1], fSize[2], 5,5))
   :add(nn.ReLU())
   :add(nn.SpatialMaxPooling(2,2,2,2)) -- 15
   :add(nn.SpatialConvolution(fSize[2], fSize[2], 3,3))
   :add(nn.ReLU())
   :add(nn.SpatialMaxPooling(2,2,2,2)) -- 15
   :add(nn.SpatialConvolution(fSize[2], fSize[3], 3,3))
   :add(nn.ReLU())
   :add(nn.SpatialConvolution(fSize[3], fSize[3], 3,3))
   :add(nn.ReLU())
   :add(nn.Reshape(featuresOut))

local iris = fovea:clone()

local model = nn.Sequential()
   :add(nn.Concat(2,2)
      :add(nn.Sequential()
         :add(nn.SpatialAveragePooling(2,2,2,2))
         :add(fovea)
      )
      :add(nn.Sequential()
         :add(nn.SpatialZeroPadding(-16,-16,-16,-16))
         :add(iris)
      )
   )
   :add(nn.Linear(featuresOut*2,featuresOut))
   :add(nn.ReLU())
   :add(nn.Linear(featuresOut,1))

utils.MSRinit(model)
utils.testModel(model)

return model

require 'optim'
require 'image'
require 'xlua'

local json = require 'cjson'
local tnt = require 'torchnet'
local utils = require 'models.utils'
require 'FPR95Meter'

local opt = {
   save_folder = 'logs',
   batchSize = 256,
   learningRate = 0.1,
   weightDecay = 0.0005,
   momentum = 0.9,
   t0 = 1e+4,
   eta0 = 0.1,
   max_epoch = 1200,
   model = '2ch',
   optimMethod = 'asgd',
   backend = 'cudnn',
   train_set = 'notredame',
   test_set = 'liberty',
   train_matches = 'm50_500000_500000_0.txt',
   test_matches = 'm50_100000_100000_0.txt',
   nDonkeys = 12,
   manualSeed = 555,
   grad_clamp = 2,
   data_type = 'torch.CudaTensor',
   testOnly = false,
   nGPU = 1,
}
opt = xlua.envparams(opt)
print(opt)


-- Data loading --

local function loadProvider()
   print'Loading train and test data'
   local p = {
      train = torch.load(opt.train_set),
      test = torch.load(opt.test_set),
   }

   -- assign matches and nonmatches
   -- add 1 because original indexes are 0-based
   for i,v in ipairs{'matches', 'nonmatches'} do
      p.train[v] = p.train[1][opt.train_matches][v] + 1
      p.test[v] = p.test[1][opt.test_matches][v] + 1
   end
   return p
end
local provider = loadProvider()


local function random_transform(x)
   local a = torch.random(6)
   for i=1,2 do
      local dst = x[i]
      if a == 1 then     -- do nothing
      elseif a == 2 then image.rotate(dst, dst:clone(), math.pi/2)
      elseif a == 3 then image.rotate(dst, dst:clone(), math.pi)
      elseif a == 4 then image.rotate(dst, dst:clone(), -math.pi/2)
      elseif a == 5 then image.hflip(dst, dst:clone())
      elseif a == 6 then image.vflip(dst, dst:clone())
      end
   end
   if torch.random(0,1) == 1 then
      local x_hat = x:clone()
      x[1]:copy(x_hat[2])
      x[2]:copy(x_hat[1])
   end
   return x
end

local function getIterator(mode)
   return tnt.ParallelDatasetIterator{
      nthread = opt.nDonkeys,
      init = function()
         require 'torchnet'
         require 'image'
      end,
      closure = function()
         local dataset = provider[mode]

         local function getListDataset(pair_type)
            local list_dataset = tnt.ListDataset{
               list = dataset[pair_type],
               load = function(idx)
                  local im = torch.FloatTensor(2,64,64)
                  for i=1,2 do
                     local mean = dataset.patches_mean[idx[i]][1]
                     im[i]:copy(dataset.patches[idx[i]]):add(-mean):div(256)
                  end
                  return {
                     input = im,
                     target = torch.LongTensor{pair_type == 'matches' and 1 or -1},
                  }
               end,
            }
            if mode == 'train' then
               list_dataset = list_dataset:shuffle():transform{input = random_transform}
            end
            return tnt.BatchDataset{
               dataset = list_dataset,
               policy = mode == 'train' and 'skip-last' or 'include-last',
               batchsize = opt.batchSize / 2,
            }
         end

         local concat = tnt.ConcatDataset{
            datasets = {
               getListDataset'matches',
               getListDataset'nonmatches',
            }
         }

         local n = concat:size()
         local multi_idx = torch.range(1,n):view(2,-1):t():reshape(n)

         return tnt.ResampleDataset{
            dataset = concat,
            sampler = function(dataset, idx)
               return multi_idx[idx]
            end,
         }:batch(2):transform{
            input = function(x) return x:view(-1,2,64,64) end,
            target = function(y) return y:view(-1) end,
         }
      end,
   }
end


-- Network setup --

local function cast(x) return x:type(opt.data_type) end
function log(t) print('json_stats: '..json.encode(tablex.merge(t,opt,true))) end

local model = opt.testOnly and torch.load(opt.model) or paths.dofile('models/'..opt.model..'.lua')
if opt.data_type:match'torch.Cuda.*Tensor' then
   require 'cudnn'
   require 'cunn'
   cudnn.convert(model, cudnn):cuda()
   cudnn.benchmark = true
   model = utils.makeDataParallelTable(model, opt.nGPU)
end
cast(model)
print(model)

local engine = tnt.OptimEngine()
local criterion = cast(nn.MarginCriterion())

-- timers
local train_timer = torch.Timer()
local test_timer = torch.Timer()
local data_timer = torch.Timer()

-- meters
local meter = tnt.AverageValueMeter()
local confusion = optim.ConfusionMatrix(2)
local data_time_meter = tnt.AverageValueMeter()
local fpr95meter = tnt.FPR95Meter()
local test_enabled = false

local inputs = cast(torch.Tensor())
local targets = cast(torch.Tensor())
engine.hooks.onSample = function(state)
   if not test_enabled then
      data_time_meter:add(data_timer:time().real)
   end
   inputs:resize(state.sample.input:size()):copy(state.sample.input)
   targets:resize(state.sample.target:size()):copy(state.sample.target)
   state.sample.input = inputs
   state.sample.target = targets
end

engine.hooks.onForwardCriterion = function(state)
   meter:add(state.criterion.output)
   confusion:batchAdd(state.network.output:gt(0):add(1), state.sample.target:gt(0):add(1))
   if test_enabled then
      fpr95meter:add(state.network.output, state.sample.target)
   end
end

local function test()
   test_enabled = true

   engine:test{
      network = model,
      iterator = getIterator'test',
      criterion = criterion,
   }

   test_enabled = false
   confusion:updateValids()
   return fpr95meter:value()
end

engine.hooks.onStartEpoch = function(state)
   local epoch = state.epoch + 1
   print('==>'.." online epoch # " .. epoch .. ' [batchSize = ' .. opt.batchSize .. ']')
   meter:reset()
   confusion:zero()
   train_timer:reset()
   data_time_meter:reset()
end

engine.hooks.onEndEpoch = function(state)
   local train_loss = meter:value()
   confusion:updateValids()
   local train_acc = confusion.totalValid * 100
   local train_time = train_timer:time().real
   meter:reset()
   print(confusion)
   confusion:zero()
   test_timer:reset()
   fpr95meter:reset()

   local cache = state.params:clone()
   state.params:copy(state.optim.ax)

   local testFPR95 = test()

   state.params:copy(cache)

   log{
      loss = train_loss,
      train_loss = train_loss,
      train_acc = train_acc,
      data_loading_time = data_time_meter:value(),
      epoch = state.epoch,
      test_acc = confusion.totalValid * 100,
      testFPR95 = testFPR95,
      lr = opt.learningRate,
      train_time = train_time,
      test_time = test_timer:time().real,
   }
end

engine.hooks.onUpdate = function(state)
   data_timer:reset()
end

if opt.testOnly then
   print('Testing on '..opt.test_set)
   print('FPR95:',test())
   os.exit()
end

engine:train{
   network = model,
   iterator = getIterator'train',
   criterion = criterion,
   optimMethod = optim[opt.optimMethod],
   config = tablex.deepcopy(opt),
   maxepoch = opt.max_epoch,
}

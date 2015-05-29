#include "loader.h"

struct Impl {
  cunn::Sequential::Ptr handle;
  THCState *state;
};

std::shared_ptr<Impl> init(const char* filename)
{
  THCState *state = (THCState*)malloc(sizeof(THCState));
  THCudaInit(state);

  std::shared_ptr<Impl> impl = std::make_shared<Impl>();
  impl->state = state;
  //impl->handle = loadNetwork(state, filename);
  return impl;
}

void reset(std::shared_ptr<Impl> ptr)
{
  THCudaShutdown(ptr->state);
  ptr.reset();
}

#include "mex.h"
#include <memory>
#include <sstream>
#include "wrapper.h"

#define MEX_ARGS int nlhs, mxArray **plhs, int nrhs, const mxArray **prhs

using namespace std;

static std::shared_ptr<Network> net_;

// Log and throw a Mex error
inline void mex_error(const std::string &msg) {
  mexErrMsgTxt(msg.c_str());
}


static void init(MEX_ARGS)
{
  if(nrhs != 1)
  {
    ostringstream error_msg;
    error_msg << "Expected 1 arguments, got " << nrhs;
    mex_error(error_msg.str());
  }

  char* param_file = mxArrayToString(prhs[0]);

  net_.reset(new Network(param_file));
  mexPrintf(net_->tostring().c_str());

  mxFree(param_file);
}

static void set_device(MEX_ARGS)
{
  if (nrhs != 1) {
    ostringstream error_msg;
    error_msg << "Expected 1 argument, got " << nrhs;
    mex_error(error_msg.str());
  }

  int device_id = static_cast<int>(mxGetScalar(prhs[0]));
  net_->setDevice(device_id - 1);
}

static void forward(MEX_ARGS)
{
  if (nrhs != 1) {
    ostringstream error_msg;
    error_msg << "Expected 1 argument, got " << nrhs;
    mex_error(error_msg.str());
  }

  const mxArray *input_array = prhs[0];
  if(!mxIsSingle(input_array))
    mex_error("Expected single array");

  const mwSize ndim = mxGetNumberOfDimensions(input_array);

  if(ndim > 4)
    mex_error("2D, 3D or 4D array expected");

  // it is impossible to have a N1xN2xN3x1 array in Matlab!
  // but cunnproduction only expects 2d or 4d tensors
  // so have to do this:
  // 2d: N x B			->	B x N
  // 3d: 64 x 64 x N		->	1 x N x 64 x 64
  // 4d: 64 x 64 x N x B	->	B x N x 64 x 64

  mwSize tndim = ndim == 3 ? tndim : ndim;

  THLongStorage *input_sizes = THLongStorage_newWithSize(tndim);

  // have to invert the dimensions because torch is row-major
  // and matlab is col-major
  {
    const mwSize* size = mxGetDimensions(input_array);
    long *input_sizes_data = THLongStorage_data(input_sizes);
    if(ndim ==3)  {
      input_sizes_data[0] = 1;
      for(mwSize i=0; i<ndim; ++i)
	input_sizes_data[ndim - i] = size[i];
    }
    else {
      for(mwSize i=0; i<ndim; ++i)
	input_sizes_data[ndim - i - 1] = size[i];
    }
  }

  THFloatTensor *input = THFloatTensor_newWithSize(input_sizes, NULL);

  memcpy(THFloatTensor_data(input),
      mxGetData(input_array),
      THFloatTensor_nElement(input) * sizeof(float));

  THFloatTensor *output = THFloatTensor_new();
  net_->forward(input, output);

  mwSize dims[4];
  const long noutput_dim = THFloatTensor_nDimension(output);
  for(long i=0; i < noutput_dim; ++i)
    dims[noutput_dim - i - 1] = output->size[i];
  mxArray* output_array = mxCreateNumericArray(noutput_dim, dims, mxSINGLE_CLASS, mxREAL);

  memcpy(mxGetData(output_array),
      THFloatTensor_data(output),
      THFloatTensor_nElement(output) * sizeof(float));

  plhs[0] = output_array;

  THFloatTensor_free(input);
  THFloatTensor_free(output);
}

static void print(MEX_ARGS)
{
  if(net_) {
    mexPrintf(net_->tostring().c_str());
  }
}

static void reset(MEX_ARGS)
{
  if(net_) {
    net_.reset();
  }
}

struct handler_registry {
  string cmd;
  void (*func)(MEX_ARGS);
};

static handler_registry handlers[] = {
  // Public API functions
  { "forward",            forward         },
  { "init",               init            },
  { "set_device",         set_device      },
  { "reset",              reset           },
  { "print",              print           },
  { "END",                NULL            },
};


void mexFunction(MEX_ARGS) {
  mexLock();
  if (nrhs == 0) {
    mex_error("No API command given");
    return;
  }

  { // Handle input command
    char *cmd = mxArrayToString(prhs[0]);
    bool dispatched = false;
    // Dispatch to cmd handler
    for (int i = 0; handlers[i].func != NULL; i++) {
      if (handlers[i].cmd.compare(cmd) == 0) {
        handlers[i].func(nlhs, plhs, nrhs-1, prhs+1);
        dispatched = true;
        break;
      }
    }
    if (!dispatched) {
      ostringstream error_msg;
      error_msg << "Unknown command '" << cmd << "'";
      mex_error(error_msg.str());
    }
    mxFree(cmd);
  }
}

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
  handle->setDevice(device_id);
}

static void forward(MEX_ARGS)
{
  if (nrhs != 1) {
    ostringstream error_msg;
    error_msg << "Expected 1 argument, got " << nrhs;
    mex_error(error_msg.str());
  }

  //plhs[0] = do_forward(prhs[0]);
}

static void reset(MEX_ARGS)
{
  if (net_) {
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
  //handle = init("../networks/2ch/2ch_liberty.bin");
  //mexPrintf(tostring(handle).c_str());
  //reset(handle);
}

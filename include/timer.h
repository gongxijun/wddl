/*
----------------------------------
Version    : ??
File Name :     timer.cpp
Description :
Author  :       xijun1
Date    :       2018/1/24
-----------------------------------
Change Activity  :   2018/1/24
-----------------------------------
__author__ = 'xijun1'
*/

//

#ifndef MXNET_WDDL_TIMER_H
#define MXNET_WDDL_TIMER_H




#include <chrono>
#ifndef DISALLOW_COPY_AND_ASSIGN

#define DISALLOW_COPY_AND_ASSIGN(TypeName)              \
  TypeName(const TypeName&);                            \
  void operator=(const TypeName&)

#endif  //DISALLOW_COPY_AND_ASSIGN

//------------------------------------------------------------------------------
// We can use the Timer class like this:
//
//   Timer timer();
//   timer.tic();
//
//     .... /* code we want to evaluate */
//
//   float time = timer.toc();  // (sec)
//
// This class can be used to evaluate multi-thread code.
//------------------------------------------------------------------------------

class Timer {
 public:
    Timer();
    // Reset start time
    void reset();
    // Code start
    void tic();
    // Code end
    float toc();
    // Get the time duration
    float get();

 protected:
    std::chrono::high_resolution_clock::time_point begin;
    std::chrono::milliseconds duration;

 private:
  DISALLOW_COPY_AND_ASSIGN(Timer);
};

#endif  //MXNET_WDDL_TIMER_H
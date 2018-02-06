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

#include "timer.h"

Timer::Timer() {
  reset();
}

// Reset code start
void Timer::reset() {
  begin = std::chrono::high_resolution_clock::now();
  duration =
     std::chrono::duration_cast<std::chrono::milliseconds>(begin-begin);
}

// Code start
void Timer::tic() {
  begin = std::chrono::high_resolution_clock::now();
}

// Code end
float Timer::toc() {
  duration += std::chrono::duration_cast<std::chrono::milliseconds>
              (std::chrono::high_resolution_clock::now()-begin);
  return get();
}

// Get the time duration (seconds)
float Timer::get() {
  return static_cast<float>((float)duration.count() / 1000.);
}
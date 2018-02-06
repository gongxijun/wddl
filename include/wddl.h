
/*
----------------------------------
Version    : ??
File Name :     wddl
Description :
Author  :       xijun1
Date    :       2018/1/24
-----------------------------------
Change Activity  :   2018/1/24
-----------------------------------
__author__ = 'xijun1'
*/

//

#ifndef MXNET_WDDL_WDDL_H
#define MXNET_WDDL_WDDL_H
#include <iostream>
#include <fstream>
#include <map>
#include <string>
#include <vector>
#include <mxnet/mxnet-cpp/MxNetCpp.h>
#include <mxnet/mxnet-cpp/op.h>
#include <glog/logging.h>
#include "../include/feature.h"
#include "../include/timer.h"

using namespace std;
using namespace mxnet::cpp;

namespace  wdl {

    class Wddl {

    public:

        Wddl();
        void Run();
        void print(mxnet::cpp::NDArray nd);

    private:
        Context ctx_dev;
        map <string, NDArray> args_map;
        map<string, NDArray> aux_map;
        std::vector<  vector<float>  > w_train_data;
        std::vector<  vector<float>  > d_train_data;
        std::vector<  vector<float>  > train_label;
        std::vector<  vector<float>  > w_val_data;
        std::vector<  vector<float>  > d_val_data;
        std::vector< vector<float> > val_label;
        void ValAccuracy(int batch_size,size_t iter,  Executor *executor);

        size_t  GetData(const char * train_data_path,
                        const unsigned int batch_size,
                        std::vector< vector<float> > &w_data,
                        std::vector< vector<float> > &d_data,
                        std::vector< vector<float> > &label);
    };

}
#endif //MXNET_WDDL_WDDL_H

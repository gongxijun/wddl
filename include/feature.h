
/*
----------------------------------
Version    : ??
File Name :     feature
Description :
Author  :       xijun1
Date    :       2018/1/24
-----------------------------------
Change Activity  :   2018/1/24
-----------------------------------
__author__ = 'xijun1'
*/

//

#ifndef MXNET_WDDL_FEATURE_H
#define MXNET_WDDL_FEATURE_H

#include <iostream>
#include <string>
#include <vector>
#include <map>
#include <glog/logging.h>
#include <stack>
#include <stdarg.h>
#include <functional>
#include <fstream>

using namespace std;

namespace  wdl {
    //_CSV_COLUMNS = [
    //'age', 'workclass', 'fnlwgt', 'education', 'education_num',
    //'marital_status', 'occupation', 'relationship', 'race', 'gender',
    //'capital_gain', 'capital_loss', 'hours_per_week', 'native_country',
    //'income_bracket'
    //]

    class Feature {
    public:
        Feature();

        float op_age(const float age);   //对年龄进行分桶

        float  op_class(const std::string key ,
                     const std::string name ,
                     std::map<std::string , int >classmap);

        int   cross_feature(const int hash_bucket_size , const int argc, ...);

        int   embedding( std::string value , const  int hash_bucket_size );

        std::map<std::string, size_t > GetData(const char * train_data_path, vector< vector<float > > *wide_data ,
                                               vector<vector<float >> *dl_data,
                                               vector<float> *label) ;

        std::vector<std::string> Separator(const std::string& line,
                       const std::string& tag);

    private:
        std::vector< int > vec_age_stage;
        std::map<std::string ,int> workclassmap;
        std::map<std::string , int > educationmap;
        std::map<std::string , int > marital_statusmap;
        std::map<std::string , int > relationshipmap;
        std::map<std::string , int > gendermap;
        std::map<std::string , int > _columns;
        int hash_bucket_size ;

    };
} // end  wddl

#endif //MXNET_WDDL_FEATURE_H

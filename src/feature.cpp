
/*
----------------------------------
Version    : ??
File Name :     feature.py
Description :
Author  :       xijun1
Date    :       2018/1/24
-----------------------------------
Change Activity  :   2018/1/24
-----------------------------------
__author__ = "xijun1"
*/

//

#include "../include/feature.h"
#define LEN( A )  (sizeof(A)/sizeof(A[0]))

namespace wdl{

    Feature::Feature() {


        //init columns
        std::string vec_columns[] = {
                "age", "workclass", "fnlwgt", "education", "education_num",
                "marital_status", "occupation", "relationship", "race", "gender",
                "capital_gain", "capital_loss", "hours_per_week", "native_country",
                "income_bracket"
        };

        for (int i = 0; i < LEN(vec_columns) ; ++i) {
            _columns[vec_columns[i]]=i;
        }

        //对age进行赋值
        int  age_stage[] = {18, 25, 30, 35, 40, 45, 50, 55, 60, 65};
        vec_age_stage.assign ( age_stage , age_stage + LEN( age_stage ) );

        //对workclass
        std::string vec_workclass[] ={"Self-emp-not-inc", "Private", "State-gov", "Federal-gov",
                                        "Local-gov", "?", "Self-emp-inc", "Without-pay", "Never-worked"};

        for (int i = 0; i < LEN(vec_workclass) ; ++i) {
            workclassmap[vec_workclass[i]]=i;
        }

        //对 education
        std::vector<std::string> vec_education={"Bachelors", "HS-grad", "11th", "Masters", "9th", "Some-college",
                                        "Assoc-acdm", "Assoc-voc", "7th-8th", "Doctorate", "Prof-school",
        "5th-6th", "10th", "1st-4th", "Preschool", "12th"};
        for (int i = 0; i < vec_education.size(); ++i) {
            educationmap[vec_education[i]]=i;
        }

        //marital_status
        std::vector<std::string > vec_marital_status={"Married-civ-spouse", "Divorced", "Married-spouse-absent",
                                         "Never-married", "Separated", "Married-AF-spouse", "Widowed"};
        for (int i = 0; i < vec_marital_status.size(); ++i) {
            marital_statusmap[vec_marital_status[i]]=i;
        }

        //relationship
        std::vector< std::string > vec_relationship = {"Husband", "Not-in-family", "Wife", "Own-child", "Unmarried",
                                           "Other-relative"};

        for (int i = 0; i < vec_relationship.size(); ++i) {
            relationshipmap[vec_relationship[i]]=i;
        }

        //gender
        std::vector<std::string > vec_gender{"Female", "Male"};
        for (int i = 0; i < vec_gender.size(); ++i) {
            gendermap[vec_gender[i]]= i;
        }

        //init
        this->hash_bucket_size = 1000;
    }

    float Feature::op_age(const float age) {
        size_t vec_size =vec_age_stage.size();
        for(int i = 0 ; i < vec_size ; ++i ) {
            if (age < vec_age_stage[i]) {
                return i;
            }
        }
        return vec_size;
    }

    float Feature::op_class(const std::string key ,
                            const std::string name ,
                            std::map<std::string , int >classmap) {

        auto iter = classmap.find(key);
        if(iter == classmap.end()){
            LOG(WARNING)<<" can not find the "<<name<<" : { "<<key<< " } in "<<name<<"\n";
            return -1;
        }
        return  iter->second;
    }

    //交叉特征
    int  Feature::cross_feature(
            const int hash_bucket_size,
            const int argc, ...) {
        CHECK_GE(argc,2)<<" features size can not less than  2 \n";
        va_list args;
        va_start(args,argc);
        //拼接格式
        std::string format="%s";
        for (int j = 1; j <argc; ++j)
            format += "_%s";
        const size_t buff_size= 1024;
        char buff[buff_size]="";
        vsnprintf(buff, (size_t) buff_size, format.c_str(), args);
        std::string str_feature =buff;
        va_end(args);
        //计算.
        return static_cast<int>((std::hash<std::string>{}(str_feature)) % hash_bucket_size);
    }

    int Feature::embedding(std::string value , const  int hash_bucket_size) {

        return static_cast<int>(std::hash<std::string>{}(value) % hash_bucket_size);
    }

    std::vector<std::string> Feature::Separator(const std::string& line,
                            const std::string& tag) {

        std::string::size_type pos_pre, pos_cur;
        pos_cur = line.find(tag);
        pos_pre = 0;
        std::vector<std::string> container;
        while(std::string::npos != pos_cur)
        {
            container.push_back(line.substr(pos_pre, pos_cur - pos_pre));
            pos_pre = pos_cur + tag.size();
            pos_cur = line.find(tag, pos_pre);
        }
        if(pos_cur != line.length())
            container.push_back(line.substr(pos_pre));

        return container;
    }

    std::map<std::string, size_t > Feature::GetData(const char * train_data_path, vector< vector<float > > *wide_data ,
                            vector<vector<float >> *dl_data,
                            vector<float> *label) {
        CHECK_NOTNULL(train_data_path);
        ifstream inf(train_data_path);
        string line;
        size_t _N = 0 , column=0;
        int hash_value=0;
        std::string indicator_columns []  ={"workclass", "education", "gender", "relationship"};
        std::string   embedding_columns [] ={"native_country", "occupation"};
        std::string continuous_columns[] = {"age", "education_num", "capital_gain", "capital_loss", "hours_per_week"};
        std::vector< std::map<std::string , int> > alg_map={this->workclassmap,
                                                            this->educationmap,
                                                            this->gendermap,
                                                            this->relationshipmap};

        //计算dl 的维度
        unsigned long  dl_vec_len = LEN(continuous_columns) + LEN(embedding_columns);

        for (int j = 0; j < LEN(indicator_columns); ++j) {
            dl_vec_len += alg_map[j].size();
        }
        unsigned long w_vec_len =3000;

        while (inf >> line) {
            //一条数据一条数据来处理.
            vector<float> wide_line_data(w_vec_len,0.f);
            vector<float> dl_line_data(dl_vec_len , 0.f);
            std::vector< std::string > attributes =this->Separator(line,",");
            //制作wide部分数据
            //交叉特征 wide data
            //["education", "occupation"]

            hash_value = this->cross_feature(this->hash_bucket_size ,2,
                                             attributes[this->_columns["education"]].c_str() ,
                                             attributes[this->_columns["occupation"]].c_str());
            wide_line_data[hash_value]=1.f;

            //["native_country", "occupation"]
            hash_value = this->cross_feature(this->hash_bucket_size ,2,
                                             attributes[this->_columns["native_country"]].c_str() ,
                                             attributes[this->_columns["occupation"]].c_str());

            wide_line_data[this->hash_bucket_size+hash_value]=1.f;

            // ["age_buckets", "education", "occupation"]
            int age_buckets = static_cast<int>(
                    this->op_age(
                            std::stoi(
                                    attributes[this->_columns["age"]])));

            hash_value = this->cross_feature(this->hash_bucket_size ,3,
                                             std::to_string(age_buckets).c_str() ,
                                             attributes[this->_columns["education"]].c_str(),
                                             attributes[this->_columns["occupation"]].c_str());
            wide_line_data[(this->hash_bucket_size<<1)+hash_value]=1.f;
            //wide_data->insert(wide_data->begin(),wide_line_data.begin(),wide_line_data.end());
            wide_data->push_back((wide_line_data));
            //计算 deep
            column = 0;
            for (int i = 0; i < LEN(embedding_columns); ++i) {
                dl_line_data[i] = this->embedding(attributes[this->_columns[embedding_columns[i]]] ,
                                                 this->hash_bucket_size);
                ++column;
            }

            // for  embedding
            for (int j = 0; j < LEN(indicator_columns); ++j) {
                int pos = static_cast<int>(this->op_class(attributes[this->_columns[indicator_columns[j]]] ,
                                                          indicator_columns[j], alg_map[j]));
                dl_line_data[column+pos] = 1.f;
                column+= alg_map[j].size();
            }

            // for continues
            for (const auto &continuous_column : continuous_columns) {
                dl_line_data[column++] = std::stof(attributes[this->_columns[continuous_column]]);
            }

            dl_data->push_back((dl_line_data));
            //处理label
            label->push_back(attributes[this->_columns["income_bracket"]] ==">50K");
            _N++;
        }

        inf.close();
        LOG(INFO)<<"data load completed...\n";
        // w_data_len , dl_data_len , line_num;
        std::map<std::string, size_t > resp;
        resp["w_data_len"]=w_vec_len;
        resp["dl_data_len"]=dl_vec_len;
        resp["line_num"]=_N;
        return resp;
    }

} // end wddl
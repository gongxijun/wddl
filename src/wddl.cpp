/*
----------------------------------
Version    : ??
File Name :     wddl.py
Description :
Author  :       xijun1
Date    :       2018/1/24
-----------------------------------
Change Activity  :   2018/1/24
-----------------------------------
__author__ = 'xijun1'
*/

//

#include "../include/wddl.h"
namespace  wdl {
    Wddl::Wddl():
              ctx_dev(mxnet::cpp::Context(mxnet::cpp::DeviceType::kCPU, 0)) {
        std::cout<<"wddl(): finsh.\n";
    }

    size_t Wddl::GetData(const char * train_data_path,
                         const unsigned int batch_size,
                         std::vector< vector<float> > &w_data,
                         std::vector< vector<float> > &d_data,
                         std::vector< vector<float> > &label) {


        size_t train_num = 0;
        wdl::Feature ft;
        //got data
        vector<float> label_vec;
        vector< vector<float> > w_data_vec,d_data_vec;
        std::map<std::string, size_t > data_count = ft.GetData(train_data_path,&w_data_vec,&d_data_vec, &label_vec);



        int stride = batch_size;
        for (int i = 0; i <data_count["line_num"] ; i+=batch_size) {
            vector<float> sub_w_data_vec , sub_d_data_vec,sub_label_vec;
            if (i+stride > data_count["line_num"])
                stride = static_cast<int>(data_count["line_num"] - i);
            for (int j = i; j < i + stride; ++j) {
                 sub_w_data_vec.insert(sub_w_data_vec.end(), w_data_vec[j].begin(), w_data_vec[j].end());
                 sub_d_data_vec.insert(sub_d_data_vec.end(), d_data_vec[j].begin(), d_data_vec[j].end());
                }
            sub_label_vec.insert(sub_label_vec.end(), label_vec.begin() + i, label_vec.begin() + i + stride);

            //填充数据
            if(stride < batch_size) {
                for (int k = 0; k < (batch_size - stride); ++k) {
                    sub_w_data_vec.insert(sub_w_data_vec.end(), w_data_vec[k].begin(), w_data_vec[k].end());
                    sub_d_data_vec.insert(sub_d_data_vec.end(), d_data_vec[k].begin(), d_data_vec[k].end());
                }
                sub_label_vec.insert(sub_label_vec.end(), label_vec.begin(),
                                     label_vec.begin() + (batch_size - stride));
            }

            train_num = data_count["line_num"];
            w_data.push_back( std::move(sub_w_data_vec));
            d_data.push_back( std::move(sub_d_data_vec));
            label.push_back( std::move(sub_label_vec));

        }
        return train_num;
    }

    void Wddl::print(mxnet::cpp::NDArray nd) {
        int cnt=0;
        auto data_1 =nd.GetData();
        for (int i = 0; i <nd.Size(); ++i) {
            if(data_1[i]>0){
                cnt++;
                LOG(INFO)<<i<<"  :  "<< data_1[i];
            }
            if(cnt>6){
                std::cout<<"-------\n";
                break;

            }
        }

    }
    void Wddl::Run(){
        //wide lr
        auto lr_sym_x = Symbol::Variable("w_data");
        auto mlp_sym_x = Symbol::Variable("d_data");
        auto sym_label = Symbol::Variable("label");
        Symbol weight=Symbol::Variable("l_w") ;
        Symbol bias=Symbol::Variable("l_b");
        Symbol lr_output  =  mxnet::cpp::FullyConnected(std::string("l_fc"),lr_sym_x,weight,bias,2);

        //mlp
        const int nLayers = 3;
        vector<int> layerSizes={50,100,2};
        vector<Symbol> weights(nLayers);
        vector<Symbol> biases(nLayers);
        for (int i = 0; i < nLayers; i++) {
            string istr = to_string(i);
            weights[i] = Symbol::Variable(string("d_w") + istr);
            biases[i] = Symbol::Variable(string("d_b") + istr);
        }

        //dns
        mxnet::cpp::Symbol d_fc1  = mxnet::cpp::FullyConnected(string("d_fc1"),mlp_sym_x,
                                               weights[0], biases[0], layerSizes[0]);

        mxnet::cpp::Symbol relu_1 =  mxnet::cpp::LeakyReLU("relu_1",d_fc1,mxnet::cpp::LeakyReLUActType::kLeaky);

        mxnet::cpp::Symbol d_fc2  = mxnet::cpp::FullyConnected(string("d_fc2"),relu_1,
                                                               weights[1], biases[1], layerSizes[1]);

        mxnet::cpp::Symbol relu_2 = mxnet::cpp::LeakyReLU("relu_2",d_fc2,mxnet::cpp::LeakyReLUActType::kLeaky);

        mxnet::cpp::Symbol d_fc3  = mxnet::cpp::FullyConnected(string("d_fc3"),relu_2,
                                                               weights[2], biases[2], layerSizes[2]);


        mxnet::cpp::Symbol pred = mxnet::cpp::SoftmaxOutput("SoftMaxOutput", lr_output+d_fc3 ,sym_label);

        for (const auto &s : pred.ListArguments()) {
            LOG(INFO) << s;
        }
        /*setup basic configs*/
        int batch_size = 100;
        int max_epoch = 20000;
        float learning_rate = 0.001;
        float weight_decay = 1e-4;
        /*prepare the data*/
        //for train
        size_t train_num = this->GetData("/Users/sina/github/mxnet-wddl/data/adult.data",
                                         static_cast<const unsigned int>(batch_size),
                                         w_train_data,
                                         d_train_data,
                                         train_label);

        // for val
        this->GetData("/Users/sina/github/mxnet-wddl/data/adult.test",
                      static_cast<const unsigned int>(batch_size ),
                      w_val_data,
                      d_val_data,
                      val_label);

        /*init some of the args*/
        args_map["w_data"] =  NDArray(Shape(batch_size,3000), ctx_dev);
        args_map["d_data"] =  NDArray(Shape(batch_size,40), ctx_dev);
        args_map["label"] =  NDArray(Shape(batch_size), ctx_dev);
        NDArray::WaitAll();
        auto *executor = pred.SimpleBind(this->ctx_dev, args_map);
        aux_map = executor->aux_dict();
        args_map = executor->arg_dict();

        //init param
        Xavier xavier = Xavier(Xavier::uniform, Xavier::in, 2.34);
        for (auto &arg : args_map) {
            /*be careful here, the arg's name must has some specific ends or starts for
             * initializer to call*/
            xavier(arg.first, &arg.second);
        }

        /*print out to check the shape of the net*/
        for (const auto &s : pred.ListArguments()) {
            std::stringstream in;
            const auto &k = args_map[s].GetShape();
            in <<"(";
            for (const auto &i : k) {
                in << i << ",";
            }
            in<<")";
            std::string data;
            in>>data;
            LOG(INFO) << s<<" : "<<data;
        }

        Optimizer* opt = mxnet::cpp::OptimizerRegistry::Find("ccsgd");
        opt->SetParam("momentum", 0.9)
                ->SetParam("rescale_grad", 1.0 / batch_size)
                ->SetParam("clip_gradient", 10)
                ->SetParam("lr", learning_rate)
                ->SetParam("wd", weight_decay);

        auto arg_names = pred.ListArguments();

        mxnet::cpp::NDArray::WaitAll();
        //训练
        std::cout<<" Training "<<std::endl;
        size_t max_iter = train_num / batch_size;
        Timer timer1;
        mxnet::cpp::Accuracy accuracy;
        mxnet::cpp::LogLoss logLoss;

        for (int ITER = 0; ITER < max_epoch; ++ITER) {

            timer1.tic();

            size_t _index = 0;
            logLoss.Reset();
            accuracy.Reset();

            LOG(INFO) << "Train Epoch: " << ITER;

            while (_index < max_iter) {

                args_map["w_data"].SyncCopyFromCPU(w_train_data[_index]);
                args_map["d_data"].SyncCopyFromCPU(d_train_data[_index]);
                args_map["label"].SyncCopyFromCPU(train_label[_index]);
                NDArray::WaitAll();

                ++_index;
                executor->Forward(true);

                if (ITER % 20 == 0) {
                    logLoss.Update(args_map["label"], executor->outputs[0]);
                    accuracy.Update(args_map["label"], executor->outputs[0]);
                }

                executor->Backward();
                // Update parameters
                for (size_t arg_ind = 0; arg_ind < arg_names.size(); ++arg_ind) {

                    //LOG(INFO)<<arg_names[arg_ind]<< executor->arg_arrays[arg_ind];
                    if (arg_names[arg_ind] == "d_data" ||
                        arg_names[arg_ind] == "w_data" ||
                        arg_names[arg_ind] == "label")
                        continue;

                    opt->Update(arg_ind, executor->arg_arrays[arg_ind], executor->grad_arrays[arg_ind]);
                    NDArray::WaitAll();
                    //LOG(INFO)<<"grad_"<<arg_names[arg_ind]<< executor->grad_arrays[arg_ind];
                }
                //LOG(INFO)<<"pred: "<<executor->outputs[0].ArgmaxChannel();
                //NDArray::WaitAll();
            }

            if (ITER % 20 == 0) {

                LOG(INFO) << "Train Epoch: "
                          << ITER
                          << ", train accuracy: "
                          << accuracy.Get()
                          << "          , train loss: "
                          << logLoss.Get();
                this->ValAccuracy(batch_size, ITER, executor);
            }

            if (ITER % 100 == 0  && ITER > 0 ) {
                /*save the parameters*/
                stringstream ss;
                ss << ITER;
                string iter_str;
                ss >> iter_str;
                string save_path_param = "/Users/sina/github/mxnet-wddl/model/wddl_param_" + iter_str+".ckpt";
                auto save_args = args_map;
                /*we do not want to save the data and label*/
                save_args.erase(save_args.find("w_data"));
                save_args.erase(save_args.find("d_data"));
                save_args.erase(save_args.find("label"));
                /*the alexnet does not get any aux array, so we do not need to save
                 * aux_map*/
                LG << "ITER: " << ITER << " Saving to..." << save_path_param;
                NDArray::Save(save_path_param, save_args);
                NDArray::WaitAll();
            }
        }
        delete executor;

    }


    void Wddl::ValAccuracy(int batch_size,size_t iter,  Executor *executor) {
        size_t val_num = w_val_data.size()-1;

        mxnet::cpp::Accuracy accuracy;
        mxnet::cpp::LogLoss logLoss;

        size_t _index = 0;
        map <string, NDArray> args_map;
        args_map["w_data"] =  NDArray(Shape(batch_size,3000), ctx_dev);
        args_map["d_data"] =  NDArray(Shape(batch_size,40), ctx_dev);
        args_map["label"] =  NDArray(Shape(batch_size), ctx_dev);
        logLoss.Reset();
        accuracy.Reset();

        while (_index < val_num) {
            args_map["w_data"].SyncCopyFromCPU(w_val_data[_index]);
            args_map["d_data"].SyncCopyFromCPU(d_val_data[_index]);
            args_map["label"].SyncCopyFromCPU( val_label[_index]);
            ++_index;
            NDArray::WaitAll();
            executor->Forward(false);
            logLoss.Update(args_map["label"], executor->outputs[0]);
            accuracy.Update(args_map["label"], executor->outputs[0]);
            NDArray::WaitAll();

        }

        LOG(INFO)<<"Train Epoch: "
                 << iter
                 << ", val accuracy: "
                 << accuracy.Get()
                 << "          , val loss: "
                 <<logLoss.Get() ;

    }
}
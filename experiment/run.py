from continuum.continuum import continuum
from continuum.data_utils import setup_test_loader

from utils.setup_elements import setup_opt, setup_architecture
from utils.utils import maybe_cuda
from utils.name_match import agents

from experiment.tune_hyperparam import tune_hyper
from experiment.metrics import compute_performance, single_run_avg_end_fgt
from types import SimpleNamespace
from utils.io import load_yaml, save_dataframe_csv, check_ram_usage
import time
import numpy as np

import pandas as pd
import os
import pickle
from utils.plot_SNE import plot_SNE
from utils.utils import mini_batch_deep_features


def multiple_run(params, store=False, save_path=None):
    # 数据集的构建，在这里通过调用continuum.dataset_scripts构建持续学习的数据集（分成了多个子任务）
    # Set up data stream
    print(params)

    start = time.time()
    print('开始构建数据集')
    data_continuum = continuum(params.data, params.cl_type, params)
    data_end = time.time()
    print('数据构建耗时: {}'.format(data_end - start))

    # 将模型参数存储
    if store:
        result_path = load_yaml('config/global.yml', key='path')['result']
        table_path = result_path + params.data
        print(table_path)
        os.makedirs(table_path, exist_ok=True)
        if not save_path:
            save_path = params.model_name + '_' + params.data_name + '.pkl'

    accuracy_list = []
    # 
    for run in range(params.num_runs):
        tmp_acc = []
        run_start = time.time()
        data_continuum.new_run()
        # 此处根据数据集初始化模型，初始化相关结构
        model = setup_architecture(params)
        import torch
        model = torch.nn.DataParallel(model).cuda()#,device_ids=[0,1]
        # model = maybe_cuda(model, params.cuda)
        # print("模型在",next(model.parameters()).device,"上训练")

        
        # import torch
        # print("正在如下设备训练",torch.cuda.current_device())
        
        # torch.cuda.set_device(7)
        # model = torch.nn.DataParallel(model).cuda()
        # print("模型在",next(model.parameters()).device,"上训练")

        # 初始化优化器
        opt = setup_opt(params.optimizer, model, params.learning_rate, params.weight_decay)
        # 初始化智能体，持续学习方法的主要部分，负责进行训练等,比如基于回放，基于正则化等
        agent = agents[params.agent](model, opt, params)

        # 构建测试集
        # prepare val data loader
        test_loaders = setup_test_loader(data_continuum.test_data(), params)
        # 开始进行训练，online表示正常持续学习，offline表示不分成一个个任务而一起训练
        if params.online:
            for i, (x_train, y_train, labels) in enumerate(data_continuum):


                # # ============debug专用===============
                # if i == 0 and params.agent=="ER":
                    
                #     plot_SNE(x_train,y_train,agent.buffer)
                #     print("没有bug！")
                # # ============debug专用===============
                # # 
                #    
                print("-----------第{}次 任务{}-------------".format(run, i))
                print('训练集大小: {}, {}'.format(x_train.shape, y_train.shape))
                # 在这里执行实际的训练过程

                agent.train_learner(x_train, y_train)
                # 此处将特征空间上的点绘制出来
                Plot_SNE = False
                if i == 0 and params.agent=="ER" and Plot_SNE:
                    plot_SNE(x_train,y_train,agent.buffer)
   

                acc_array = agent.evaluate(test_loaders)
                tmp_acc.append(acc_array)
            run_end = time.time()
            print(
                "-----------第{}次-----------最终平均准确率{}-----------训练用时{}".format(run, np.mean(tmp_acc[-1]),
                                                                               run_end - run_start))
            accuracy_list.append(np.array(tmp_acc))
        else:
            x_train_offline = []
            y_train_offline = []
            for i, (x_train, y_train, labels) in enumerate(data_continuum):
                x_train_offline.append(x_train)
                y_train_offline.append(y_train)
            print('Training Start')
            x_train_offline = np.concatenate(x_train_offline, axis=0)
            y_train_offline = np.concatenate(y_train_offline, axis=0)
            print("----------run {} training-------------".format(run))
            print('size: {}, {}'.format(x_train_offline.shape, y_train_offline.shape))
            agent.train_learner(x_train_offline, y_train_offline)
            acc_array = agent.evaluate(test_loaders)
            accuracy_list.append(acc_array)

    accuracy_array = np.array(accuracy_list)
    end = time.time()
    if store:
        result = {'time': end - start}
        result['acc_array'] = accuracy_array
        save_file = open(table_path + '/' + save_path, "wb")
        pickle.dump(result, save_file)
        save_file.close()
    if params.online:
        # Fwt:训练之前比随机初始化的准确率高多少（衡量这个任务之前任务对该任务的影响
        # Bwt:训练完最终任务后比刚训练完这个任务时的准确率高多少（衡量这个任务之后任务对该任务的影响
        # 最终平均准确率：最后一个任务训练结束后的准确率
        # 最终平均遗忘率：训练过程中最高准确率与最终准确率之差
        # 平均准确率：在该任务训练完以后的所有准确率的平均
        # 各种评价指标在此处进行计算
        avg_end_acc, avg_end_fgt, avg_acc, avg_bwtp, avg_fwt = compute_performance(accuracy_array)
        print('----------- Total {} run: {}s -----------'.format(params.num_runs, end - start))
        print('----------- 最终平均准确率avg_end_acc {} 最终平均遗忘率avg_end_fgt {} 平均准确率avg_acc {} 后向影响avg_bwtp {} 前向影响avg_fwt {}-----------'
              .format(avg_end_acc, avg_end_fgt, avg_acc, avg_bwtp, avg_fwt))
        print(accuracy_array)
        return (avg_end_acc, avg_end_fgt, avg_acc, avg_bwtp, avg_fwt)
    else:
        print('----------- Total {} run: {}s -----------'.format(params.num_runs, end - start))
        print("avg_end_acc {}".format(np.mean(accuracy_list)))
        return(np.mean(accuracy_list))




def multiple_run_tune(defaul_params, tune_params, save_path):
    # Set up data stream
    start = time.time()
    print('Setting up data stream')
    data_continuum = continuum(defaul_params.data, defaul_params.cl_type, defaul_params)
    data_end = time.time()
    print('data setup time: {}'.format(data_end - start))

    #store table
    # set up storing table
    table_path = load_yaml('config/global.yml', key='path')['tables']
    metric_list = ['Avg_End_Acc'] + ['Avg_End_Fgt'] + ['Time'] + ["Batch" + str(i) for i in range(defaul_params.num_val, data_continuum.task_nums)]
    param_list = list(tune_params.keys()) + metric_list
    table_columns = ['Run'] + param_list
    table_path = table_path + defaul_params.data
    os.makedirs(table_path, exist_ok=True)
    if not save_path:
        save_path = defaul_params.model_name + '_' + defaul_params.data_name + '.csv'
    df = pd.DataFrame(columns=table_columns)
    # store list
    accuracy_list = []
    params_keep = []
    for run in range(defaul_params.num_runs):
        tmp_acc = []
        tune_data = []
        run_start = time.time()
        data_continuum.new_run()
        # prepare val data loader
        test_loaders = setup_test_loader(data_continuum.test_data(), defaul_params)
        tune_test_loaders = test_loaders[:defaul_params.num_val]
        test_loaders = test_loaders[defaul_params.num_val:]
        for i, (x_train, y_train, labels) in enumerate(data_continuum):
            if i < defaul_params.num_val:
                #collection tune data
                tune_data.append((x_train, y_train, labels))
                if len(tune_data) == defaul_params.num_val:
                    # tune
                    best_params = tune_hyper(tune_data, tune_test_loaders, defaul_params, tune_params)
                    params_keep.append(best_params)
                    final_params = vars(defaul_params)
                    final_params.update(best_params)
                    final_params = SimpleNamespace(**final_params)
                    # set up
                    print('Tuning is done. Best hyper parameter set is {}'.format(best_params))
                    model = setup_architecture(final_params)
                    model = maybe_cuda(model, final_params.cuda)
                    opt = setup_opt(final_params.optimizer, model, final_params.learning_rate, final_params.weight_decay)
                    agent = agents[final_params.agent](model, opt, final_params)
                    print('Training Start')
            else:
                print("----------run {} training batch {}-------------".format(run, i))
                print('size: {}, {}'.format(x_train.shape, y_train.shape))
                agent.train_learner(x_train, y_train)
                acc_array = agent.evaluate(test_loaders)
                tmp_acc.append(acc_array)

        run_end = time.time()
        print(
            "-----------run {}-----------avg_end_acc {}-----------train time {}".format(run, np.mean(tmp_acc[-1]),
                                                                           run_end - run_start))
        accuracy_list.append(np.array(tmp_acc))

        #store result
        result_dict = {'Run': run}
        result_dict.update(best_params)
        end_task_acc = tmp_acc[-1]
        for i in range(data_continuum.task_nums - defaul_params.num_val):
            result_dict["Batch" + str(i + defaul_params.num_val)] = end_task_acc[i]
        result_dict['Avg_End_Acc'] = np.mean(tmp_acc[-1])
        result_dict['Avg_End_Fgt'] = single_run_avg_end_fgt(np.array(tmp_acc))
        result_dict['Time'] = run_end - run_start
        df = df.append(result_dict, ignore_index=True)
        save_dataframe_csv(df, table_path, save_path)
    accuracy_list = np.array(accuracy_list)
    avg_end_acc, avg_end_fgt, avg_acc, avg_bwtp, avg_fwt = compute_performance(accuracy_list)
    end = time.time()
    final_result = {'Run': 'Final Result'}
    final_result['Avg_End_Acc'] = avg_end_acc
    final_result['Avg_End_Fgt'] = avg_end_fgt
    final_result['Time'] = end - start
    df = df.append(final_result, ignore_index=True)
    save_dataframe_csv(df, table_path, save_path)
    print('----------- Total {} run: {}s -----------'.format(defaul_params.num_runs, end - start))
    print('----------- Avg_End_Acc {} Avg_End_Fgt {} Avg_Acc {} Avg_Bwtp {} Avg_Fwt {}-----------'
          .format(avg_end_acc, avg_end_fgt, avg_acc, avg_bwtp, avg_fwt))



def multiple_run_tune_separate(default_params, tune_params, save_path):
    # Set up data stream
    start = time.time()
    print('Setting up data stream')
    data_continuum = continuum(default_params.data, default_params.cl_type, default_params)
    data_end = time.time()
    print('data setup time: {}'.format(data_end - start))

    if default_params.num_val == -1:
        # offline tuning
        default_params.num_val = data_continuum.data_object.task_nums
    #store table
    # set up storing table
    result_path = load_yaml('config/global.yml', key='path')['result']
    table_path = result_path + default_params.data + '/' + default_params.cl_type
    for i in default_params.trick:
        if default_params.trick[i]:
            trick_name = i
            table_path = result_path + default_params.data + '/' + default_params.cl_type + '/' + trick_name
            break
    print(table_path)
    os.makedirs(table_path, exist_ok=True)
    if not save_path:
        save_path = default_params.model_name + '_' + default_params.data_name + '_' + str(default_params.seed) + '.pkl'
    # store list
    accuracy_list = []
    params_keep = []
    if isinstance(default_params.num_runs, int):
        run_list = range(default_params.num_runs)
    else:
        run_list = default_params.num_runs
    for run in run_list:
        tmp_acc = []
        run_start = time.time()
        data_continuum.new_run()
        if default_params.train_val:
            single_tune_train_val(data_continuum, default_params, tune_params, params_keep, tmp_acc, run)
        else:
            single_tune(data_continuum, default_params, tune_params, params_keep, tmp_acc, run)
        run_end = time.time()
        print(
            "-----------run {}-----------avg_end_acc {}-----------train time {}".format(run, np.mean(tmp_acc[-1]),
                                                                           run_end - run_start))
        accuracy_list.append(np.array(tmp_acc))

    end = time.time()
    accuracy_array = np.array(accuracy_list)
    result = {'seed': default_params.seed}
    result['time'] = end - start
    result['acc_array'] = accuracy_array
    result['ram'] = check_ram_usage()
    result['best_params'] = params_keep
    save_file = open(table_path + '/' + save_path, "wb")
    pickle.dump(result, save_file)
    save_file.close()
    print('----------- Total {} run: {}s -----------'.format(default_params.num_runs, end - start))
    print('----------- Seed {} RAM: {}s -----------'.format(default_params.seed, result['ram']))

def single_tune(data_continuum, default_params, tune_params, params_keep, tmp_acc, run):
    tune_data = []
    # prepare val data loader
    test_loaders_full = setup_test_loader(data_continuum.test_data(), default_params)
    tune_test_loaders = test_loaders_full[:default_params.num_val]
    test_loaders = test_loaders_full[default_params.num_val:]

    if default_params.online:
        for i, (x_train, y_train, labels) in enumerate(data_continuum):
            if i < default_params.num_val:
                # collection tune data
                tune_data.append((x_train, y_train, labels))
                if len(tune_data) == default_params.num_val:
                    # tune
                    best_params = tune_hyper(tune_data, tune_test_loaders, default_params, tune_params, )
                    params_keep.append(best_params)
                    final_params = vars(default_params)
                    final_params.update(best_params)
                    final_params = SimpleNamespace(**final_params)
                    # set up
                    print('Tuning is done. Best hyper parameter set is {}'.format(best_params))
                    model = setup_architecture(final_params)
                    model = maybe_cuda(model, final_params.cuda)
                    opt = setup_opt(final_params.optimizer, model, final_params.learning_rate, final_params.weight_decay)
                    agent = agents[final_params.agent](model, opt, final_params)
                    print('Training Start')
            else:
                print("----------run {} training batch {}-------------".format(run, i))
                print('size: {}, {}'.format(x_train.shape, y_train.shape))
                agent.train_learner(x_train, y_train)
                acc_array = agent.evaluate(test_loaders)
                tmp_acc.append(acc_array)
    else:
        x_train_offline = []
        y_train_offline = []
        x_tune_offline = []
        y_tune_offline = []
        labels_offline = []
        for i, (x_train, y_train, labels) in enumerate(data_continuum):
            if i < default_params.num_val:
                # collection tune data
                x_tune_offline.append(x_train)
                y_tune_offline.append(y_train)
                labels_offline.append(labels)
            else:
                x_train_offline.append(x_train)
                y_train_offline.append(y_train)
        tune_data = [(np.concatenate(x_tune_offline, axis=0), np.concatenate(y_tune_offline, axis=0),
                      np.concatenate(labels_offline, axis=0))]
        best_params = tune_hyper(tune_data, tune_test_loaders, default_params, tune_params, )
        params_keep.append(best_params)
        final_params = vars(default_params)
        final_params.update(best_params)
        final_params = SimpleNamespace(**final_params)
        # set up
        print('Tuning is done. Best hyper parameter set is {}'.format(best_params))
        model = setup_architecture(final_params)
        model = maybe_cuda(model, final_params.cuda)
        opt = setup_opt(final_params.optimizer, model, final_params.learning_rate, final_params.weight_decay)
        agent = agents[final_params.agent](model, opt, final_params)
        print('Training Start')
        x_train_offline = np.concatenate(x_train_offline, axis=0)
        y_train_offline = np.concatenate(y_train_offline, axis=0)
        print("----------run {} training-------------".format(run))
        print('size: {}, {}'.format(x_train_offline.shape, y_train_offline.shape))
        agent.train_learner(x_train_offline, y_train_offline)
        acc_array = agent.evaluate(test_loaders)
        tmp_acc.append(acc_array)



def single_tune_train_val(data_continuum, default_params, tune_params, params_keep, tmp_acc, run):
    tune_data = []
    # prepare val data loader
    test_loaders_full = setup_test_loader(data_continuum.test_data(), default_params)
    tune_test_loaders = test_loaders_full[:default_params.num_val]
    if default_params.online:
        for i, (x_train, y_train, labels) in enumerate(data_continuum):
            if i < default_params.num_val:
                # collection tune data
                tune_data.append((x_train, y_train, labels))
                if len(tune_data) == default_params.num_val:
                    # tune
                    best_params = tune_hyper(tune_data, tune_test_loaders, default_params, tune_params, )
                    params_keep.append(best_params)
                    final_params = vars(default_params)
                    final_params.update(best_params)
                    final_params = SimpleNamespace(**final_params)
                    print('Tuning is done. Best hyper parameter set is {}'.format(best_params))
                    break

        data_continuum.reset_run()
        # set up
        model = setup_architecture(final_params)
        model = maybe_cuda(model, final_params.cuda)
        opt = setup_opt(final_params.optimizer, model, final_params.learning_rate, final_params.weight_decay)
        agent = agents[final_params.agent](model, opt, final_params)
        print('Training Start')
        for i, (x_train, y_train, labels) in enumerate(data_continuum):
            print("----------run {} training batch {}-------------".format(run, i))
            print('size: {}, {}'.format(x_train.shape, y_train.shape))
            agent.train_learner(x_train, y_train)
            acc_array = agent.evaluate(test_loaders_full)
            tmp_acc.append(acc_array)
    else:
        x_train_offline = []
        y_train_offline = []
        x_tune_offline = []
        y_tune_offline = []
        labels_offline = []
        for i, (x_train, y_train, labels) in enumerate(data_continuum):
            if i < default_params.num_val:
                # collection tune data
                x_tune_offline.append(x_train)
                y_tune_offline.append(y_train)
                labels_offline.append(labels)
            x_train_offline.append(x_train)
            y_train_offline.append(y_train)
        tune_data = [(np.concatenate(x_tune_offline, axis=0), np.concatenate(y_tune_offline, axis=0), labels_offline)]
        best_params = tune_hyper(tune_data, tune_test_loaders, default_params, tune_params, )
        params_keep.append(best_params)
        final_params = vars(default_params)
        final_params.update(best_params)
        final_params = SimpleNamespace(**final_params)
        # set up
        print('Tuning is done. Best hyper parameter set is {}'.format(best_params))
        model = setup_architecture(final_params)
        model = maybe_cuda(model, final_params.cuda)
        opt = setup_opt(final_params.optimizer, model, final_params.learning_rate, final_params.weight_decay)
        agent = agents[final_params.agent](model, opt, final_params)
        print('Training Start')
        x_train_offline = np.concatenate(x_train_offline, axis=0)
        y_train_offline = np.concatenate(y_train_offline, axis=0)
        print("----------run {} training-------------".format(run))
        print('size: {}, {}'.format(x_train_offline.shape, y_train_offline.shape))
        agent.train_learner(x_train_offline, y_train_offline)
        acc_array = agent.evaluate(test_loaders_full)
        tmp_acc.append(acc_array)



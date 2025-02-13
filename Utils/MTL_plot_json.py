"""
generate the plot of the results    Script  ver： Dec 29th 15:00
"""
import os
import json
import matplotlib.pyplot as plt


def recore_decode(file_path, task_dict, phases=['Train', 'Val']):
    with open(file_path) as f:

        out_put_dict = {}
        for phase in phases:
            out_put_dict[phase] = {}
            for task_name in task_dict:
                out_put_dict[phase][task_name] = {'loss': [], 'measure': [], 'Matrix': []}

        load_dict = json.load(f)

        check_flag = 0  # check teh task dict in the experiment

        for epoch in load_dict:
            for phase in phases:
                Average_sample_loss = load_dict[epoch][phase]['Average_sample_loss']
                epoch_results = load_dict[epoch][phase]['epoch_results']

                if check_flag == 0:
                    check_flag = 1
                    task_idx_dict = {}
                    for idx, task_record in enumerate(epoch_results):
                        task_idx_dict[task_record[0]] = idx
                else:
                    pass

                for task_name in task_dict:
                    out_put_dict[phase][task_name]['loss'].append(float(Average_sample_loss[task_idx_dict[task_name]]))
                    # -1 use the last indicator as the measure indicator
                    out_put_dict[phase][task_name]['measure'].append(float(epoch_results[task_idx_dict[task_name]][-1]))
                    # 1 decode the confusion matrix
                    out_put_dict[phase][task_name]['Matrix'].append(epoch_results[task_idx_dict[task_name]][1])

    return out_put_dict


def draw_lines_fig(data_list, task_dict_names, phase=None, task_dict=None, save_path=None):
    fig = plt.figure(dpi=500)
    color_list = ['r', 'g', 'b', 'violet', 'darkorange', 'olive', 'c', 'm', 'hotpink', 'slategrey',
                  'lime', 'brown', 'gold', 'navy', 'k']

    cls_flag = 0
    reg_flag = 0
    two_type_flag = 0

    for idx, data in enumerate(data_list):
        epochs = [i + 1 for i in range(len(data))]

        if cls_flag + reg_flag == 0:
            ax1 = fig.add_subplot(111)
            ax1.plot(epochs, data, color_list[idx], label=task_dict_names[idx])
            # ax1.legend(loc='best', fontsize=10)

            if task_dict[task_dict_names[idx]] == list:
                cls_flag += 1
            else:
                reg_flag += 1
        else:
            if two_type_flag == 0:
                if task_dict[task_dict_names[idx]] == list and reg_flag == 1:
                    ax2 = ax1.twinx()
                    two_type_flag = 1
                elif task_dict[task_dict_names[idx]] == float and cls_flag == 1:
                    ax2 = ax1.twinx()
                    two_type_flag = 1
                else:
                    pass

            if reg_flag == 1 and two_type_flag == 1:  # ax1 is reg
                if task_dict[task_dict_names[idx]] == float:
                    ax1.plot(epochs, data, color_list[idx], label=task_dict_names[idx])
                    # ax1.legend(loc='best', fontsize=10)
                else:
                    ax2.plot(epochs, data, color_list[idx], label=task_dict_names[idx])
                    # ax2.legend(loc='best', fontsize=10)

            elif cls_flag == 1 and two_type_flag == 1:  # ax1 is cls
                if task_dict[task_dict_names[idx]] == list:
                    ax1.plot(epochs, data, color_list[idx], label=task_dict_names[idx])
                    # ax1.legend(loc='best', fontsize=10)
                else:
                    ax2.plot(epochs, data, color_list[idx], label=task_dict_names[idx])
                    # ax2.legend(loc='best', fontsize=10)
            else:
                ax1.plot(epochs, data, color_list[idx], label=task_dict_names[idx])
                # ax1.legend(loc='best', fontsize=10)

    if phase is not None:
        if two_type_flag == 1:
            if phase[-4:] == 'loss':
                if reg_flag == 1:
                    ax1.set_ylabel('L1 loss')
                    ax2.set_ylabel('cross entropy')
                else:
                    ax2.set_ylabel('L1 loss')
                    ax1.set_ylabel('cross entropy')
            elif phase[-7:] == 'measure':
                if reg_flag == 1:
                    ax1.set_ylabel('absolute distance (l1)')
                    ax2.set_ylabel('Accuracy')
                else:
                    ax2.set_ylabel('absolute distance (l1)')
                    ax1.set_ylabel('Accuracy')
            else:
                pass
            ax1_lines, ax1_labels = fig.axes[0].get_legend_handles_labels()
            ax2_lines, ax2_labels = fig.axes[1].get_legend_handles_labels()
            ax1_lines.extend(ax2_lines)
            ax1_labels.extend(ax2_labels)
            fig.legend(ax1_lines, ax1_labels, loc='upper right', fontsize=5)
        elif reg_flag == 1:  # ax1 is reg
            if phase[-4:] == 'loss':
                plt.ylabel('L1 loss')
            elif phase[-7:] == 'measure':
                plt.ylabel('absolute distance (l1)')
            else:
                pass
            ax1_lines, ax1_labels = fig.axes[0].get_legend_handles_labels()
            fig.legend(ax1_lines, ax1_labels, loc='upper right', fontsize=5)
        elif cls_flag == 1:  # ax1 is reg
            if phase[-4:] == 'loss':
                plt.ylabel('cross entropy')
            elif phase[-7:] == 'measure':
                plt.ylabel('Accuracy')
            else:
                pass
            ax1_lines, ax1_labels = fig.axes[0].get_legend_handles_labels()
            fig.legend(ax1_lines, ax1_labels, loc='upper right', fontsize=5)
        else:
            pass

    if phase is not None:
        plt.title(phase)
        if save_path is not None:
            if not os.path.exists(save_path):
                os.mkdir(save_path)

            plt.savefig(os.path.join(save_path, phase + '.jpg'))

    if save_path is None:
        plt.show()


def draw_confusion_matrix():
    pass


def check_json_with_plot(file_path, task_dict, phases=['Train', 'Val'], save_path=None):
    task_dict_names = [key for key in task_dict]

    out_put_dict = recore_decode(file_path, task_dict, phases=phases)

    for phase in phases:
        loss_list = [out_put_dict[phase][key]['loss'] for key in task_dict]
        draw_lines_fig(loss_list, task_dict_names, phase + ' loss', task_dict, save_path=save_path)

        measure_list = [out_put_dict[phase][key]['measure'] for key in task_dict]
        draw_lines_fig(measure_list, task_dict_names, phase + ' measure', task_dict, save_path=save_path)


def check_json_with_matrix(file_path, task_dict, phases=['Train', 'Val'], save_path=None):
    task_dict_names = [key for key in task_dict]

    out_put_dict = recore_decode(file_path, task_dict, phases=phases)

    # a_task_matrix_at_epoch_phase = out_put_dict[phase][task_name]['Matrix'][epoch_idx]

    for phase in phases:
        print(out_put_dict[phase]['EGFR']['Matrix'][0])
        # draw_confusion_matrix


if __name__ == '__main__':
    file_path = r'/4tbB/WSIT/temp_check_runs/Test_MTL_task_num_3_mil_pooling_strategy_Ave_sample_mode_local_bag_scheduler_strategy_loss_back/2022_12_08_log.json'
    # task_dict = {'EGFR': list, 'Relapse': list, 'purity': float}
    task_dict = {'EGFR': list, 'Relapse': list, 'purity_ABSOLUTE': float, 'recurrence_days': float, 'Survival': list,
                 'Follow_up_days': float, 'Purity_titanCNA': float, 'Purity_AbsCNseq': float, 'Purity_PurBayes': float,
                 'TMB_normalised_status': float}
    # check_json_with_matrix(file_path, task_dict, phases=['Test'], save_path='/4tbB/WSIT')
    check_json_with_plot(file_path, task_dict, phases=['Train', 'Val'], save_path='/4tbB/WSIT')
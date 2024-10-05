import torch
import torch.nn as nn
from torchvision import models
from torch.autograd import Variable

from collections import OrderedDict
import numpy as np

def summary(model, inputs, *args):
    if len(args) >= 1:
        x = [inputs, *args]
    else:
        x = [inputs]
    summary_str = ''

    # create properties
    net_name, module_names = get_names_dict(model)
    summary = OrderedDict()
    hooks = []

    global max_layer_length
    max_layer_length = 0
    def register_hook(module):
        def hook(module, input, output):
            global max_layer_length
            class_name = str(module.__class__).split(".")[-1].split("'")[0]
            module_idx = len(summary)

            module_name = str(module_idx)
            for name, item in module_names.items():
                if item == module:
                    module_name = name
                    break
            
            sep_module_name = module_name.split("-")
            if len(sep_module_name) > 1:
                module_name = "-".join(sep_module_name[:-1])
            length = len(module_name)
            if length > max_layer_length:
                max_layer_length = length


            # m_key = "%i-%s" % (module_idx + 1, module_name)
            m_key = "{:>}> {:<10}".format(str(module_idx+1).zfill(len(str(len(module_names)))), module_name)
            summary[m_key] = OrderedDict()
            summary[m_key]["id"] = id(module)
            summary[m_key]["output_size"] = 0
            if isinstance(output, (list, tuple)):
                summary[m_key]["output_shape"] = []
                for o in output:
                    if isinstance(o, (list, tuple)):
                        for io in o:
                            try:
                                summary[m_key]["output_shape"].append(list(io.size()))
                                summary[m_key]["output_size"] += 1
                            except AttributeError:
                                # pack_padded_seq and pad_packed_seq store feature into data attribute
                                summary[m_key]["output_shape"].append(list(io.data.size()))
                                summary[m_key]["output_size"] += 1
                    else:
                        try:
                            summary[m_key]["output_shape"].append(list(o.size()))
                            summary[m_key]["output_size"] += 1
                        except AttributeError:
                            # pack_padded_seq and pad_packed_seq store feature into data attribute
                            summary[m_key]["output_shape"].append(list(o.data.size()))
                            summary[m_key]["output_size"] += 1
            else:
                summary[m_key]["output_shape"] = list(output.size())
                summary[m_key]["output_size"] += 1

            b_params = 0
            w_params = 0
            summary[m_key]["ksize"] = "-" 
            summary[m_key]["nb_params"] = 0
            for name, param in module.named_parameters():
                if name == "weight":    
                    ksize = list(param.size())
                    if len(ksize) > 1:
                        ksize[0], ksize[1] = ksize[1], ksize[0]
                    summary[m_key]["ksize"] = ksize
            for name, param in module.named_parameters():
                if "weight" in name:
                    w_params += param.nelement()
                    summary[m_key]["trainable"] = param.requires_grad
                elif "bias" in name:
                    b_params += param.nelement()
            summary[m_key]["nb_params"] = w_params + b_params

            if list(module.named_parameters()):
                for k, v in summary.items():
                    if summary[m_key]["id"] == v["id"] and k != m_key:
                        summary[m_key]["nb_params"] = 0
        
        if (
            not isinstance(module, nn.Sequential)
            and not isinstance(module, nn.ModuleList)
            and not module._modules
        ):
            hooks.append(module.register_forward_hook(hook))

    
    # register hook
    model.apply(register_hook)
    # make a forward pass
    model(*x)

    # remove these hooks
    for h in hooks:
        h.remove()

    max_layer_length += 5
    lengths = [max_layer_length, 20, 20, 15]
    total_length = sum(lengths) + 5

    summary_str += "-" * total_length + "\n"
    line_new = "{:<{width}}||{:>20} {:>20} {:>15}".format(
        "Layer(type) ", "Kernel Shape", "Output Shape", "Param #", width=max_layer_length)
    summary_str += line_new + "\n"
    summary_str += "=" * total_length + "\n"
    total_params = 0
    total_output = 0
    trainable_params = 0

    for _ in range(len(x)):
        if _ == 0:
            line_new = "{:<{width}}||{:>20} {:>20} {:>15}".format(
                str(net_name) + " Inputs", "-", str(list(x[_].size())), "-", width=max_layer_length)
            summary_str += line_new + "\n"
        else:
            line_new = "{:<{width}}||{:>20} {:>20} {:>15}".format(
                " ", "-", str(list(x[_].size())), "-", width=max_layer_length)
            summary_str += line_new + "\n"
    line_new = "{:<{width}}||{:>20} {:>20} {:>15}".format(
            " ", " ", " " + " ", " ", width=max_layer_length)
    summary_str += line_new + "\n"

    for layer in summary:
        _output_size = summary[layer]["output_size"]
        if  _output_size > 1:
            for i in range(_output_size):
                if i == 0:
                    line_new = "{:<{width}}||{:>20} {:>20} {:>15}".format(
                        layer,
                        str(summary[layer]["ksize"]),
                        str(summary[layer]["output_shape"][0]),
                        "{0:,}".format(summary[layer]["nb_params"]),
                        width=max_layer_length
                    )
                    total_params += summary[layer]["nb_params"]
                    total_output += np.sum(np.prod(summary[layer]["output_shape"], axis=1))
                    if "trainable" in summary[layer]:
                        if summary[layer]["trainable"] == True:
                            trainable_params += summary[layer]["nb_params"]
                    summary_str += line_new + "\n"  
                else:
                    line_new = "{:<{width}}||{:>20} {:>20} {:>15}".format(" ", "-", str(summary[layer]["output_shape"][i]), "-", width=max_layer_length)
                    summary_str += line_new + "\n"  
        else:
            line_new = "{:<{width}}||{:>20} {:>20} {:>15}".format(
                layer,
                str(summary[layer]["ksize"]),
                str(summary[layer]["output_shape"]),
                "{0:,}".format(summary[layer]["nb_params"]),
                width=max_layer_length
            )
            total_params += summary[layer]["nb_params"]
            total_output += np.prod(summary[layer]["output_shape"])
            if "trainable" in summary[layer]:
                if summary[layer]["trainable"] == True:
                    trainable_params += summary[layer]["nb_params"]
            summary_str += line_new + "\n"

    # assume 4 bytes/number (float on cuda).
    total_input_size = 0
    for i in range(len(x)):
        size = np.array(x[i].size())
        total_input_size += np.prod(size) * 4. / (1024 ** 2.)

    total_output_size = abs(2. * total_output * 4. /
                            (1024 ** 2.))
    total_params_size = abs(total_params * 4. / (1024 ** 2.))
    total_size = total_params_size + total_output_size + total_input_size

    summary["Total_Params_Size"] = total_params

    summary_str += "=" * total_length + "\n"
    summary_str += "Total params: {0:,}".format(total_params) + "\n"
    summary_str += "Trainable params: {0:,}".format(trainable_params) + "\n"
    summary_str += "Non-trainable params: {0:,}".format(total_params -
                                                        trainable_params) + "\n"
                                                        
    summary_str += "-" * total_length + "\n"
    summary_str += "Input size (MB): %0.2f" % total_input_size + "\n"
    summary_str += "Forward/backward pass size (MB): %0.2f" % total_output_size + "\n"
    summary_str += "Params size (MB): %0.2f" % total_params_size + "\n"
    summary_str += "Estimated Total Size (MB): %0.2f" % total_size + "\n"
    summary_str += "-" * total_length + "\n"
    
    print(summary_str)
    return summary, (total_params, trainable_params)

def get_names_dict(model):
    """Recursive walk to get names including path."""
    names = {}
    net_name = model._get_name_()

    def _get_names(module, parent_name=None):
        for key, m in module.named_children():
            if str.isdigit(key):
                key = int(key) + 1
                key = str(key)

            key = str.capitalize(key)
            cls_name = str(m.__class__).split(".")[-1].split("'")[0]
            num_named_children = len(list(m.named_children()))

            if num_named_children > 1:
                name = "{}-{}".format(parent_name, key) if parent_name else key
            else:
                name = "{}-{}-{}".format(parent_name, cls_name, key) if parent_name else key
            names[name] = m

            if isinstance(m, torch.nn.Module):
                _get_names(m, parent_name=name)

    _get_names(model, net_name)
    return net_name, names
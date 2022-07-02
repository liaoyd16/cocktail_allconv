
import os
import sys
sys.path.append("../")
import Meta

def clean_results_in_(phase_no, test_or_train):
    ret = os.system("rm -rf " + os.path.join(Meta.PROJ_ROOT, "../../data2/results/phase{}/{}/*".format(phase_no, test_or_train)))
    print(ret)

def clean_log_dir():
    os.system("rm " + os.path.join(Meta.PROJ_ROOT, "scripts/log/event*"))

def mkdir_in(path_parent, path_child):
    assert(os.path.isdir(path_parent) and ('/' not in path_child))
    if os.path.isdir(os.path.join(path_parent, path_child)):
        os.system("rm -rf " + os.path.join(path_parent, path_child, "*"))
    elif os.path.isfile(os.path.join(path_parent, path_child)):
        os.system("rm " + os.path.join(path_parent, path_child))
    else:
        os.system("mkdir " + os.path.join(path_parent, path_child))


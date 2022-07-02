import __init__
from __init__ import *

from torch.utils.data import DataLoader

import Meta
from models.SpeechEmbedder import SpeechEmbedder
if Meta.model_meta['attend_layers'] == 3:
    from models.AttentionNet4 import AttentionNet
    from models.ResAE4 import ResAE
elif Meta.model_meta['attend_layers'] == 5:
    from models.AttentionNet import AttentionNet
    from models.ResAE import ResAE
elif Meta.model_meta['attend_layers'] == 8:
    from models.AttentionNet9 import AttentionNet
    from models.ResAE9 import ResAE

#from utils.logger import Logger
from utils.skip import *

from aux import *

train_meta = {
    'model': {
        'shortcut': {4:  [True for _ in range(3)], 
                     6:  [True for _ in range(5)],
                     9:  [True for _ in range(8)]
                    },
    },
    're_dump_embeddings': False,
    'do_sample': False,
    '2' : {
        'lr': 1e-3,
        'reuse': 2, # -1 for testing dataset IO, no reuse, no train
        'test_before_continue': False,
        'epochs': 20,
        'batch': {
            'train': 10,
            'test': 10,
        },
        'lossF': nn.MSELoss(),
        'batch_count': 0,
        'test_batch_count': 0,
        'loss_test': 0,
        'optimizer': 'adam',
    },
    '3' : {
        'lr': 1e-3,
        'reuse': 2, # -1 for testing dataset IO
        'test_before_continue': True,
        'epochs':20,
        'batch': {
            'train': 10,
            'test': 10,
        },
        'lossF': nn.MSELoss(),
        'image_batch_count': 0,
        'batch_count': 0,
        'test_batch_count': 0,
        'loss_test': 0,
        'optimizer': 'adam',
    },
}

import sys

def _iterate_and_update_(model_updater, logger, phase_no, train):
    if skip(): return

    model_updater.mode(train)
    if not train:
        loss_test = []

    batchsize = train_meta[phase_no]['batch'][{True:'train', False:'test'}[train]]
    dataloader = JsonDataLoader(batchsize, phase_no, train)
    test_batch_count = 0
    print("ready")
    len_data = len(dataloader)
    for i in range(len_data):
        if train:
            print("\ttrain batch #{}, ".format(train_meta[phase_no]['batch_count']), end='')
            sys.stdout.flush()
        else:
            print(f"test batch #{test_batch_count}, ", end='')
            sys.stdout.flush()

        batch = dataloader[i]
        model_updater.update(batch)
        if train:
            #logger.summary('loss{}_train'.format(phase_no), 
            #    model_updater.loss(mean=True), train_meta[phase_no]['batch_count'])
            print("{} = {}".format('loss{}_train'.format(phase_no), \
                                               model_updater.loss(mean=False)))
            train_meta[phase_no]['batch_count'] += 1
        else: # test
            loss_test.append(model_updater.loss(mean=True))
            if train_meta['do_sample']:
                model_updater.sample(test_batch_count)
            test_batch_count += 1
            print("finished")

        if skip(): return

    if not train:
        train_meta[phase_no]['loss_test'] = np.mean(loss_test)
        print(f"[{datetime.datetime.now()}]: test loss = {train_meta[phase_no]['loss_test']}")


if __name__ == '__main__':
    NUM_LAYERS = Meta.model_meta['attend_layers'] + 1


    """ init models """
    print(f"[{datetime.datetime.now()}]: init models")
    shortcut = [False for _ in range(NUM_LAYERS-1)]
    aeNet_2 = ResAE(Meta.model_meta['keepsize_num_en'], \
                  Meta.model_meta['keepsize_num_de'], \
                  shortcut)
    aeNet_3 = ResAE(Meta.model_meta['keepsize_num_en'], \
                  Meta.model_meta['keepsize_num_de'], \
                  train_meta['model']['shortcut'][NUM_LAYERS])
    aNet = AttentionNet()


    """ load models """
    # model structure -> directory
    print(f"[{datetime.datetime.now()}]: load models")
    hyper_param = "n={}_en={}_resblock_3x3conv_5shortcut".format(Meta.model_meta['attend_layers']+1,\
                                      Meta.model_meta['keepsize_num_en'])
    if not os.path.isdir("../pickled/{}/".format(hyper_param)): os.makedirs("../pickled/{}/".format(hyper_param))

    if Meta.model_meta['keepsize_num_de']==0: hyper_param += "/de=0"
    else: hyper_param += "/de=en"
    if not os.path.isdir("../pickled/{}/".format(hyper_param)): os.makedirs("../pickled/{}/".format(hyper_param))    


    """ logging stuff """
    #logger = Logger("./log")
    #clean_log_dir()
    logger = None


    """ phase 2: autoencode """
    ## load model
    if train_meta['2']['reuse'] in [1,2]:
        aeNet_2 = torch.load("../pickled/{}/aeNet_2.pickle".format(hyper_param),
                                map_location=torch.device(Meta.DEVICE_ID))
    aeNet_2.to(Meta.device)
    print("aeNet_2 init finish")

    ## optimizer
    if train_meta['2']['optimizer']=='adam':
        optimizer = torch.optim.Adam(aeNet_2.parameters(), lr=train_meta['2']['lr'])
    else:
        optimizer = torch.optim.SGD(aeNet_2.parameters(), lr=train_meta['2']['lr'], momentum=0.9)

    ## train or test
    if train_meta['2']['reuse'] in [0,1]:
        model_updater = ModelUpdater_autoencode(aeNet_2, train_meta['2']['lossF'], optimizer)
        clean_results_in_(2, "test")
        clean_results_in_(2, "train")
        for epo in range(train_meta['2']['epochs']):
            print(f"[{datetime.datetime.now()}]: phase 2, epoch {epo}")
            _iterate_and_update_(model_updater, logger, phase_no='2', train=True)
            torch.cuda.empty_cache()
            _iterate_and_update_(model_updater, logger, phase_no='2', train=False)
            if skip():
                reset_skip()
                break
        torch.save(aeNet_2, "../pickled/{}/aeNet_2.pickle".format(hyper_param))
    elif train_meta['2']['test_before_continue']:
        model_updater = ModelUpdater_autoencode(aeNet_2, train_meta['2']['lossF'], optimizer)
        _iterate_and_update_(model_updater, logger, phase_no='2', train=False)


    """ phase 3: attended denoise """
    if train_meta['3']['reuse'] in [1,2]:
        aeNet_3 = torch.load("../pickled/{}/aeNet_3.pickle".format(hyper_param),
                                map_location=torch.device(Meta.DEVICE_ID))
        aNet = torch.load("../pickled/{}/aNet.pickle".format(hyper_param), 
                             map_location=torch.device(Meta.DEVICE_ID))
    elif train_meta['2']['reuse'] != -1 and train_meta['3']['reuse'] == 0:
        aeNet_2 = torch.load("../pickled/{}/aeNet_2.pickle".format(hyper_param),
                                map_location=torch.device(Meta.DEVICE_ID))
        aeNet_3.load_from_2(aeNet_2)
        #aeNet_3 = torch.load("../pickled/{}/aeNet_2.pickle".format(hyper_param),
        #                        map_location=torch.device(Meta.DEVICE_ID))

    aNet.to(Meta.device)
    aeNet_3.to(Meta.device)
    print("aeNet_3 init finish")

    # optimizer
    parameters = list(aNet.parameters())
    parameters.extend(aeNet_3.parameters())
    if train_meta['3']['optimizer']=='adam':
        optimizer = torch.optim.Adam(parameters, lr=train_meta['3']['lr'])
    else:
        optimizer = torch.optim.SGD(parameters, lr=train_meta['3']['lr'], momentum=0.9)

    # train or test
    if train_meta['3']['reuse'] in [0,1]:
        model_updater = ModelUpdater_denoise(aNet, aeNet_3, train_meta['3']['lossF'], optimizer)
        clean_results_in_(3, "test")
        clean_results_in_(3, "train")
        for epo in range(train_meta['3']['epochs']):
            print(f"[{datetime.datetime.now()}]: phase 3, epoch {epo}")
            _iterate_and_update_(model_updater, logger, phase_no='3', train=True)
            torch.cuda.empty_cache()
            _iterate_and_update_(model_updater, logger, phase_no='3', train=False)
            if skip():
                reset_skip()
                break
        torch.save(aNet, "../pickled/{}/aNet.pickle".format(hyper_param))
        torch.save(aeNet_3, "../pickled/{}/aeNet_3.pickle".format(hyper_param))
    elif train_meta['3']['test_before_continue']:
        model_updater = ModelUpdater_denoise(aNet, aeNet_3, train_meta['3']['lossF'], optimizer)
        _iterate_and_update_(model_updater, logger, phase_no='3', train=False)

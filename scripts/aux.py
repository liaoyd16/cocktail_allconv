
import __init__
from __init__ import *

import sys
import Meta
import pdb
import math

sampling_meta = {
    'per_batch' : 10,
}


def _make_FloatTensor(array):
    if torch.cuda.is_available(): return torch.cuda.FloatTensor(array,device=Meta.DEVICE_ID)
    else:                         return torch.Tensor(array)

def _make_LongTensor(array):
    if torch.cuda.is_available(): return torch.cuda.LongTensor(array,device=Meta.DEVICE_ID)
    else:                         return torch.LongTensor(array)



class JsonDataset:
    def __init__(self, speaker, train, do_shuffle=True):
        self.speaker = speaker
        self.slices_per_block = Meta.data_meta['slices_per_block']
        if train:
            self.blocks = Meta.data_meta['train_blocks_per_speaker']
            self.do_shuffle = do_shuffle
        else:
            self.blocks = Meta.data_meta['test_blocks_per_speaker']
            self.do_shuffle = False
        print("cache init ...", end=''); sys.stdout.flush()
        self.cache_init()
        print("finish ... check dataset ready ...", end=''); sys.stdout.flush()
        self.check_dataset_ready()
        print("finish", end=''); sys.stdout.flush()


    def check_dataset_ready(self):
        ready = True
        for iblock in range(1+Meta.data_meta['blocks_per_speaker']):
            if os.path.isfile(Meta.audio_name_(self.speaker, iblock)):
                continue
            ready = False
            break

        if not ready: self.do_dataset_generation()
        print(f"[{datetime.datetime.now()}]: dataset for {self.speaker} is now ready")


    def do_dataset_generation(self):
        print(f"[{datetime.datetime.now()}]: dataset generation for {self.speaker} started")
        speaker_dir, mp3fname_list = Meta.mp3_names_(self.speaker)

        # [1] load, concatenate mp3 -> embed_seq / blocks_seq
        full_seq = []
        for mp3fname in mp3fname_list:
            print(f"\treading {mp3fname}")
            xs, _ = librosa.load(os.path.join(speaker_dir, mp3fname))
            full_seq.append(xs)
        
        full_seq = np.concatenate(full_seq, axis=0).tolist()

        slice_len = Meta.data_meta['slice_len']
        embed_seq = full_seq[:slice_len]
        blocks_seq = full_seq[slice_len:]

        # [2] slice -save-> (iblock).json
        iblock_embedder = Meta.data_meta['iblock_embedder']
        iblock_all = Meta.data_meta['train_blocks_per_speaker']
        iblock_all.extend(Meta.data_meta['test_blocks_per_speaker'])

        # embedder block
        dest_json = Meta.audio_name_(self.speaker, iblock_embedder)
        json.dump(embed_seq, open(dest_json, "w"))

        # later blocks
        for iblock in iblock_all:
            dest_json = Meta.audio_name_(self.speaker, iblock)
            block_content = []
            for iseq in range(self.slices_per_block):
                start = ((iblock-1) * self.slices_per_block + iseq) * slice_len
                end = start + slice_len
                seq = blocks_seq[start : end]
                assert(len(seq)==slice_len)
                seq = seq / np.max(seq)
                block_content.append(seq.tolist())

            json.dump(block_content, open(dest_json, "w"))


    def cache_init(self):
        self.cached_block = None
        self.cached_iblock = -1

        self.j_block_map = copy.copy(self.blocks)
        self.j_offset_map = np.arange(self.slices_per_block)
        if self.do_shuffle: self.shuffle()

    def cache_update(self, i_block):
        fhand = open(Meta.audio_name_(self.speaker, i_block), 'rb')
        _ = fhand.read(1)
        print('read json success', end=''); sys.stdout.flush()
        fhand = open(Meta.audio_name_(self.speaker, i_block))
        self.cached_block = json.load(fhand)
        self.cached_iblock = i_block

    def shuffle(self):
        """
            shuffle before every epoch begins (i.e. when JsonDataset initialized)
            NEVER shuffle while epoch still going
            shuffle along: block_num axis & offset axis
            hence indexing with __getitem__ in order
            is both cache-friendly & randomized in each epoch
        """
        np.random.shuffle(self.j_block_map)
        np.random.shuffle(self.j_offset_map)

    def __getitem__(self, index):
        j_block = index // self.slices_per_block
        j_offset = index % self.slices_per_block

        try:
            i_block = self.j_block_map[j_block]
            offset = self.j_offset_map[j_offset]

            if self.cached_iblock != i_block:
                print(f'{self.speaker}: {index} cache_update ...', end='')
                self.cache_update(i_block)
                print('cache update finish')
            
            return np.array(self.cached_block[offset])
        except:
            print(index, self.j_block_map)
            

    def __len__(self):
        return self.slices_per_block * len(self.blocks)


class JsonDataLoader:
    def __init__(self, batchsize, phase_no, train):
        self.using_speakers = Meta.get_using_speakers()
        self.json_datasets = dict()
        self.batchsize = batchsize
        for speaker in self.using_speakers:
            self.json_datasets[speaker] = JsonDataset(speaker, train)

        all_speakers = Meta.data_meta['speakers']
        self.pairing = [[all_speakers[i], all_speakers[j]] for (i,j) in Meta.data_meta['pairing']]
        self.phase_no = phase_no

    def __getitem__(self, index):
        ans = []
        if self.phase_no == '3':
            indices = np.arange(index * self.batchsize, (index+1) * self.batchsize)
            for sp1, sp2 in self.pairing:
                slices1 = [self.json_datasets[sp1][i] for i in indices]
                slices2 = [self.json_datasets[sp2][i] for i in indices]
                ans.append([slices1, slices2])

            return ans
        elif self.phase_no == '2':
            indices = np.arange(index * self.batchsize, (index+1) * self.batchsize)
            for sp in self.using_speakers:
                slices = [self.json_datasets[sp][i] for i in indices]
                ans.append(slices)

            return ans

    def __len__(self):
        return len(self.json_datasets[self.using_speakers[0]]) // self.batchsize


class ModelUpdater_autoencode:
    def __init__(self, model, lossF, optimizer):
        self.model = model
        self.lossF = lossF
        self.optimizer = optimizer
        self._loss = 0
        self._train = False
        self.attentions = [torch.ones(1,1,1,1).to(Meta.device)
            for l in range(Meta.model_meta['attend_layers'])]

        self.total_tops = []

    def mode(self, train):
        self._train = train
        if train:
            self.model.train()
        else:
            self.model.eval()

    def update(self, batch):
        n_speakers = len(batch)
        batchsize = len(batch[0])
        sp_i_s = list(itertools.product( range(n_speakers), range(batchsize) ))
        batch_per_sp = [batch[sp_i[0]][sp_i[1]] for sp_i in sp_i_s]
        coch = _make_FloatTensor(spectrogram(batch_per_sp))

        outputs = self.model(coch, self.attentions).view(-1, *Meta.data_meta['specgram_size'])

        loss = self.lossF(outputs, coch)
        self._loss = loss.item()
        # for sampling outputs & inputs
        self.ground_truth = coch[0 : sampling_meta['per_batch']].cpu().detach().numpy()
        self.recover = outputs[0 : sampling_meta['per_batch']].cpu().detach().numpy()

        if self._train:
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()

    def loss(self, mean):
        return self._loss

    def sample(self, batch_count):
        train_or_test = {True:"train", False:"test"}[self._train]
        spb = sampling_meta['per_batch']
        for k in range(spb):
            path_to_sample = os.path.join("../results/phase2/{}".format(train_or_test), "sample_{}".format(batch_count * spb + k))
            if not os.path.isdir("../results/phase2/{}".format(train_or_test)):
                os.makedirs("../results/phase2/{}".format(train_or_test))

            utils.dir_utils.mkdir_in( \
                "../results/phase2/{}".format(train_or_test), \
                "sample_{}".format(batch_count * spb + k), \
            )
            json.dump(self.ground_truth[k].tolist(),
                      open(os.path.join(path_to_sample, "ground_truth.json"), "w")
            )
            json.dump(self.recover[k].tolist(),
                      open(os.path.join(path_to_sample, "recover.json"), "w")
            )

import pdb
import sys
sys.path.append("..")
from models.SpeechEmbedder import SpeechEmbedder
def _load_features_of_speakers():
    speech_embedder = SpeechEmbedder()
    speech_embedder.load_state_dict(torch.load("../pickled/speech_embedder.pth"))
    speech_embedder.to(Meta.device)

    slices = []
    using_speakers = Meta.get_using_speakers()
    for sp in Meta.data_meta['speakers']:
        embedder_blockname = Meta.audio_name_(sp, Meta.data_meta['iblock_embedder'])
        if sp in using_speakers: x = np.array(json.load(open(embedder_blockname)))
        else: x = np.zeros(Meta.data_meta['slice_len'])
        slices.append(x)

    embedding = speech_embedder.embed(np.array(slices))
    ans = dict()
    for i, sp in enumerate(using_speakers):
        ans[sp] = embedding[i]
    return ans


class ModelUpdater_denoise:
    def __init__(self, anet, aenet, lossF, optimizer):
        self.anet = anet
        self.aenet = aenet

        self.lossF = lossF
        self.optimizer = optimizer

        self.speakers_features = _load_features_of_speakers()
        self.attentions_none = [torch.ones(1,1,1,1).to(Meta.device) 
            for l in range(Meta.model_meta['attend_layers'])]
        self._train = False

        self.all_speakers = Meta.data_meta['speakers']
        self.pairing = [[self.all_speakers[i], self.all_speakers[j]] for (i,j) in Meta.data_meta['pairing']]
        self.init_sample_cache()
        
    def init_sample_cache(self):
        self.sp1_specs = []
        self.sp2_specs = []
        self.mixed = [] 
        self.recover_12 = [] 
        self.recover_21 = [] 
        self.recover_none = [] 


    def mode(self, train):
        self._train = train
        if train:
            self.anet.train()
        else:
            self.anet.eval()

    def update(self, batch):
        npairs = len(batch)
        batch_size = len(batch[0][0])

        # now it's 2-sized batch
        losses = [[],[]]
        embed_size = Meta.model_meta['embedder']['proj']
        for i_pair, (sp1, sp2) in enumerate(self.pairing):
            features_1 = torch.Tensor.repeat(self.speakers_features[sp1].resize(1,embed_size), (batch_size,1))
            attentions_1 = self.anet(features_1)
            features_2 = torch.Tensor.repeat(self.speakers_features[sp2].resize(1,embed_size), (batch_size,1))
            attentions_2 = self.anet(features_2)

            sp1_specs = _make_FloatTensor(
                spectrogram([batch[i_pair][0][i] for i in range( batch_size )]))
            sp2_specs = _make_FloatTensor(
                spectrogram([batch[i_pair][1][i] for i in range( batch_size )]))
            mixed = _make_FloatTensor(
                spectrogram([batch[i_pair][0][i] + batch[i_pair][1][i] for i in range(batch_size)]))

            top_12 = self.aenet.upward(mixed, attentions_1)
            recover_12 = self.aenet.downward(top_12).view(-1, *Meta.data_meta['specgram_size'])
            top_21 = self.aenet.upward(mixed, attentions_2)
            recover_21 = self.aenet.downward(top_21).view(-1, *Meta.data_meta['specgram_size'])

            if not self._train:
                top_none = self.aenet.upward(mixed, self.attentions_none)
                recover_none = self.aenet.downward(top_none).view(-1, *Meta.data_meta['specgram_size'])

            loss_12 = self.lossF(recover_12, sp1_specs)
            loss_21 = self.lossF(recover_21, sp2_specs)
            loss = (loss_12 + loss_21) * 0.5
            if self._train:
                loss.backward(retain_graph=True)
                self.optimizer.step()
                self.optimizer.zero_grad()

            losses[0].append(loss_12.item())
            losses[1].append(loss_21.item())

            # save things for sample() method
            spb = sampling_meta['per_batch']
            self.sp1_specs.append( sp1_specs.cpu().detach().numpy()[:spb] )
            self.sp2_specs.append( sp2_specs.cpu().detach().numpy()[:spb] )
            self.mixed.append( mixed.cpu().detach().numpy()[:spb] )
            self.recover_12.append( recover_12.cpu().detach().numpy()[:spb] )
            self.recover_21.append( recover_21.cpu().detach().numpy()[:spb] )
            if not self._train:
                self.recover_none.append(recover_none.cpu().detach().numpy()[:spb])

        self._loss = [np.mean(losses[0]),np.mean(losses[1])]

    def loss(self, mean):
        if mean:
            return np.mean(self._loss)
        else:
            return self._loss

    def sample(self, batch_count):
        # json dump self.ground_truth[k], self.masker[k], self.mixed[k], self.recover[k]
        train_or_test = {True:"train", False:"test"}[self._train]

        spb = sampling_meta['per_batch']
        if not os.path.isdir(f"../results/phase3/{train_or_test}"):
            os.makedirs(f"../results/phase3/{train_or_test}")

        for k in range(spb):
            utils.dir_utils.mkdir_in(f"../results/phase3/{train_or_test}", "sample_{}".format(batch_count * spb + k))
            for i_pair, (sp1, sp2) in enumerate(self.pairing):
                path_to_pair = os.path.join("../results/phase3/{}/sample_{}".format(train_or_test, batch_count * spb + k), "pair_{}".format(i_pair))
                utils.dir_utils.mkdir_in( \
                    "../results/phase3/{}/sample_{}".format(train_or_test, batch_count * spb + k), \
                    "pair_{}".format(i_pair) \
                )
                json.dump(self.sp1_specs[i_pair][k].tolist(), \
                          open(os.path.join(path_to_pair, "sp1.json") , "w") \
                )
                json.dump(self.sp2_specs[i_pair][k].tolist(), \
                          open(os.path.join(path_to_pair, "sp2.json") , "w") \
                )
                json.dump(self.mixed[i_pair][k].tolist(), \
                          open(os.path.join(path_to_pair, "mix.json") , "w") \
                )
                json.dump(self.recover_12[i_pair][k].tolist(), \
                          open(os.path.join(path_to_pair, "re1.json") , "w") \
                )
                json.dump(self.recover_21[i_pair][k].tolist(), \
                          open(os.path.join(path_to_pair, "re2.json") , "w") \
                )
                json.dump(self.recover_none[i_pair][k].tolist(), \
                          open(os.path.join(path_to_pair, "re_none.json") , "w") \
                )

        self.init_sample_cache()

def Corr(a,b):
    a = a - np.mean(a)
    b = b - np.mean(b)
    r = (a*b).sum() / math.sqrt((a*a).sum() * (b*b).sum())
    return r

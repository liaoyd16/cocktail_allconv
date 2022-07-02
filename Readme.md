### Readme

---

#### Project description

This project is code directory for our work on cocktail party problem. The project trains and tests an auto-encoder model that is able to separate one spectrogram component from a 'cocktail party' mixture (this is called cocktail party effect). Structurally, the model contains a ResNet-based autoencoder, with multi-layered attention signal projecting on it.

For our project, we are using data from pairs of speakers in LibriVox dataset. See links below: 
- female speaker 1 (fem1): https://librivox.org/the-adventures-of-a-nature-guide-by-enos-a-mills/
- female speaker 2 (fem2): https://librivox.org/discourses-biological-geological-by-thomas-henry-huxley/
- female speaker 3 (fem3): https://librivox.org/cloud-studies-by-arthur-william-clayden/
- male speaker 1 (male1): https://librivox.org/100-the-story-of-a-patriot-by-upton-sinclair/
- male speaker 2 (male2): https://librivox.org/anti-imperialist-writings-by-mark-twain/

In our study, we trained our model using pairs: fem1-male2, fem2-male1. You could also train and test using other speaker pairs.

To prepare dataset, please download from the links. You do not need to download all mp3 files, but only selected sections:
- fem1: section01 section02 section03
- fem2: section00 section01 section02 section03
- fem3: section00 section01 section02 section03 section04
- male1: section01 section02 section03 section04 section05 section06 section07 section09
- male2: section02 section06

Make sure downloaded mp3s are in the corresponding directories with correct renaming:
- fem1: dataset/fem1/{ch1, ch2, ch3}.mp3
- fem2: dataset/fem2/{ch0, ch1, ch2, ch3}.mp3
- fem3: dataset/fem3/{ch0, ch1, ch2, ch3, ch4}.mp3
- male1: dataset/male1/{ch1, ch2, ch3, ch4, ch5, ch6, ch7, ch9}.mp3
- male2: dataset/male2/{ch2, ch6}.mp3

#### Running tips

##### 1. SpeechEmbedder

For SpeechEmbedder, we trained the model described in https://github.com/HarryVolek/PyTorch_Speaker_Verification. If you would like to re-train this speaker embedder model, clone the aforementioned directory to your computer and follow their instructions.

Or, you could directly use .pth file provided: https://drive.google.com/file/d/1Zi86RoIz0cWa-sLx3eFu6bN6IGwyKX2r/view?usp=sharing.

Create folder pickled/ and place acquired speech_embedder.pth in it.

##### 2. Autoencoder

Training code are all integrated in `scripts/supervised.py`. To train everything from scratch, just `cd scripts/` and `python supervised.py`.

Configuration of training plan is hard-coded in `scripts/supervised.py`. To configure an autoencoder, you would like to open  `scripts/supervised.py`, then focus on section `2` in configuration variable `train_meta (type 'dict')`.

Key `'reuse'` has three legitimate values:` 0`,`1` and `2`. `0` tells the program to train everything from scratch, `1` tells the program to load checkpoint and continue training, and `2` skips this training session.

Key `'test_before_continue'` tells the program whether test checkpoint model before training begins or not.

##### 3. Cocktail party!

Similarly, training plan associated with `AttentionNet` is specified in section `3`. if you would like to skip this section, change key `'reuse'` to `2`. To do everything from scratch, change `'reuse'` to `0`. To train from previous checkpoint, change it to `1`.

#### Changing model meta-parameters

All model meta-parameter information are configured in `Meta.py`. Two parts of configuration are coded here: `data_meta` and `model_meta`.

`data_meta`: Choose your interested pair in `'using_speakers'`, see key `'pairing'`.

`model_meta`: Here, you have the opportunity to disable attentions at layers. This is useful when plotting figures for the model under different attending scenarios. Specifically, edit `attend_layers` for different model depths (`4`, `6`, `9`) and `layer_attentions` (a sequence of 0 and 1, specifying whether or not the program projects attention signal onto each level in encoder).

##### Figure 2

Obtain figure 2 by running `scripts/supervised.py`. Find the "speaker1, speaker2, mixture, mixture-recovered, speaker1-recovered, speaker2-recovered" hexa-tuple in directory `results/phase3/test/sample_?/pair_?/` (for cocktail party task). Check up `results/phase2/test/sample_?/` for spectrogram task performance.

To plot spectrograms in Figure 2, copy interested hexa-tuple pair (e.g. results/phase3/test/sample_9/pair_0/) to results/figures/figure2/ (e.g. `cp -r results/phase3/test/sample_9/pair_0/ results/figures/figure2/`). Go to results/figures/figure2/ then and execute ` python savefig.py`.

##### Figure 3

###### AMI scatter and histogram

To run this part, first run `scripts/multifunction_dump.py`. Be sure to first set variables `cumu` to `True`, `topdown_scan` to `True` and `Type` to `'spec'`.

Go to file `scripts/scatter.py` and uncomment `figure3` part. Then switch to `scripts/` and run `python scatter.py cumu`. This will give you `results/figures/figure3/ami_hist_6.png` and `results/figures/figure3/ami_scatter_6_mark_legend.png`.

###### AMI barplot

To run this part, first run `scripts/multifunction_dump.py`. Be sure to first set variables `cumu` to `True`, `topdown_scan` to `True` and `Type` to `'spec'`.

Go to file `scripts/barplot.py` and uncomment `figure3` part. Then run `python barplot.py`. This will give you `results/figures/figure3/bar.png`.

##### Figure 4

###### AMI scatters

To run this part, first run `scripts/multifunction_dump.py`. Be sure to first set variables `cumu` to `Fal`, `topdown_scan` to `True` and `Type` to `'spec'`.

Go to file `scripts/scatter.py` and uncomment `figure4/5` part. Then switch to `scripts/` and run `python scatter.py cumu`. This will give you `ami_scatter_*.png` under directory `results/figures/[model metaparameters]/` and `ami_scatter_*_legend.png` under directory `results/figures/[model metaparamters]`.

###### AMI barplots

To run this part, first run `scripts/multifunction_dump.py`. Be sure to first set variables `cumu` to `True`, `topdown_scan` to `True` and `Type` to `'spec'`.

Go to file `scripts/barplot.py` and uncomment `figure4,5` part. Edit `cumu_single` to `'cumu'`. Then switch to `scripts/` and run `python barplot.py`. This will give you `results/barplot_1x_cumu.png` and `results/barplot_2x_cumu.png`.

##### Figure 5

###### AMI scatters

To run this part, first run `scripts/multifunction_dump.py`. Be sure to first set variables `cumu` to `False`, `topdown_scan` to `True` and `Type` to `'spec'`.

Go to file `scripts/scatter.py` and uncomment `figure4/5` part. Then switch to `scripts/` and run `python scatter.py single`. This will give you `ami_scatter_*.png` under directory `results/figures/figure5/` and `ami_scatter_*_legend.png` under directory `results/figures/figure5/`.

###### AMI barplots

To run this part, first run `scripts/multifunction_dump.py`. Be sure to first set variables `cumu` to `False`, `topdown_scan` to `True` and `Type` to `'spec'`.

Go to file `scripts/barplot.py` and uncomment `figure4,5` part. Edit `cumu_single` to `'single'`. Then switch to `scripts/` and run `python barplot.py`. This will give you `results/figures/figure5/barplot_1x_single.png` and `results/figures/figure5/barplot_2x_single.png`.

###### Figure 6
- convert `.json` files to `.mat` files


- cumulative projection

- single layer projection


#### Required libraries

- python 3.7.1
- librosa 0.7.1
- numpy 1.21.4
- matplotlib 3.1.1
- pytorch 1.0.1.post2
- scipy 1.7.3

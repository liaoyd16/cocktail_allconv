import matplotlib.pyplot as plt
import numpy as np
import json
import math
import os

## Matplotlib setting 
plt.rcParams['figure.dpi'] = 300
plt.rcParams.update({'font.style': 'normal'})
plt.rcParams.update({'font.size': 20})
arfont = {'fontname':'Arial'}
plt.style.use('default')

## 計算兩張圖之間 correlation 
def mean2(x):
    y = np.sum(x) / np.size(x);
    return y

def corr2(a,b):
    a = a - mean2(a)
    b = b - mean2(b)

    r = (a*b).sum() / math.sqrt((a*a).sum() * (b*b).sum());
    return r

cumu_single = None
n = 6
k = 3
m = 3
de = 'en' # or: 0
MARK = 9

## AMI 計算函數
def ami_cal(con):
    '''
    輸入參數 con 為 condition, 使用時僅需輸入 json 檔名，例如 AMI_condition_0

    '''
    ami_path = f'../results/dumped/n={n}_en={k}_resblock_3x3conv_5shortcut/de={de}/'

    with open("{}{}.json".format(ami_path, con)) as fil:
        spec = np.array(json.load(fil))

    sp1_sp1 = [] # attend SP1, original SP1
    sp1_sp2 = [] # attend SP1, original SP2
    sp2_sp2 = [] # attend SP2, original SP2
    sp2_sp1 = [] # attend SP2, original SP1
    
    ami_list = []
    n_samples = len(spec[0])
    for i in range(n_samples):
        sp1_sp1.append(corr2(spec[0][i], spec[2][i]))
        sp1_sp2.append(corr2(spec[0][i], spec[3][i]))

        sp2_sp2.append(corr2(spec[1][i], spec[3][i]))
        sp2_sp1.append(corr2(spec[1][i], spec[2][i]))

    return np.array(sp1_sp1), np.array(sp1_sp2), np.array(sp2_sp2), np.array(sp2_sp1)
    

## 散點圖函數
def scat(con, attend, do_legend=False, mark_sample=-1):
    '''
        當產生帶 attention 的散點圖時設置 attend = True, 會將 sp1 和 sp2 分開繪製.
        當 attend = False 時不會分開繪製 sp1 和 sp2 
    '''
    sp1_sp1, sp1_sp2, sp2_sp2, sp2_sp1 = ami_cal(con)
    
    if attend == False:
        plt.scatter(sp1_sp1, sp1_sp2, label = 'Non-attend', s = 10, color = '#747474', alpha = 0.7)
        if not mark_sample == -1:
            plt.scatter([sp1_sp1[mark_sample]], [sp1_sp2[mark_sample]], \
            label=f'#{MARK} no attend', marker='o', color='y')
    else:
        plt.scatter(sp1_sp1, sp1_sp2, label = 'Target SP1', s = 10, color = 'red')
        plt.scatter(sp2_sp1, sp2_sp2, label = 'Target SP2', s = 10, color = 'blue')
        if not mark_sample == -1:
            plt.scatter([sp1_sp1[mark_sample]], [sp1_sp2[mark_sample]], marker='o', color='g', label=f'#{MARK} attend')
            plt.scatter([sp2_sp1[mark_sample]], [sp2_sp2[mark_sample]], marker='o', color='g')

    plt.plot([0,1], [0,1], color= "black", ls = "--", alpha = 0.55)

    plt.gca().set_aspect('equal', adjustable='box')
    axes = plt.gca()
    axes.set_xlim([-0.1,1.01])
    axes.set_ylim([-0.1,1.01])

    plt.tick_params(
        axis= 'both',          
        bottom= True,      # ticks along the bottom edge are off
        top= False,         # ticks along the top edge are off
        labelbottom= True,
        labelleft = True,
        left = True,
        right = False) # labels along the bottom edge are off

    plt.xticks([0, 0.2, 0.4, 0.6, 0.8, 1.0], fontsize=15)
    plt.yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0], fontsize=15)
    if do_legend:
        plt.legend(loc = 3, fontsize=15)

## AMI histogram
def ami_hist(con):
    sp1_sp1, sp1_sp2, sp2_sp2, sp2_sp1 = ami_cal(con)
    amis = sp1_sp1 + sp2_sp2 - sp1_sp2 - sp2_sp1
    plt.xlim(0, 2)
    plt.xticks(ticks=[0,.5,1.,1.5, 2.], fontsize=15)
    plt.yticks(ticks=[0,.5,1.,1.5, 2.], fontsize=15)
    plt.hist(amis, density=True, label="AMI", bins=20, color='green')
    plt.legend(loc='best', fontsize=15)

if __name__ == '__main__':
    import tqdm
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("cumu_single", nargs=1, type=str)
    args = parser.parse_args()
    cumu_single = args.cumu_single[0]

    which_figure = {'single': 5, 'cumu': 4}[cumu_single]

    """
    # figure 4/5 0~4
    for i in tqdm.tqdm(range(0, n-1)):    ## 散點圖執行範例
        plt.figure(figsize = (3.5, 3.5)) # 正方的圖統一將大小設置為 3.5
        
        scat("condition_{}-tpdwn_0_spec".format(cumu_single), attend=False) # 使用 scat 函數輸入 AMI_condition_0.json 產生散點圖
        scat("condition_{}-tpdwn_{}_spec".format(cumu_single,i), attend=True)
        plt.savefig(f"../results/figures/figure{which_figure}/corr_scatter_{i}.png", dpi = 300, bbox_inches='tight') # 設置 bbox_inches = 'tight' 確保在存圖時最外圍的字不會被切掉

    # figure 4/5 5th
    print("do legend scatter")
    plt.figure(figsize = (3.5, 3.5))
    scat("condition_{}-tpdwn_0_spec".format(cumu_single), attend=False, do_legend=True)
    scat("condition_{}-tpdwn_{}_spec".format(cumu_single,n-1), attend=True, do_legend=True)
    plt.savefig(f"../results/figures/figure{which_figure}/corr_scatter_{n-1}_legend.png", dpi = 300, bbox_inches='tight')
    #"""

    # figure 3 - mark, ami_hist.png
    if cumu_single == 'cumu':
        which_figure = 3
        print(f"do legend & mark=#{MARK} scatter")
        plt.figure(figsize = (3.5, 3.5))
        scat("condition_cumu-tpdwn_0_spec", attend=False, do_legend=True, mark_sample=MARK)
        scat(f"condition_cumu-tpdwn_{n-1}_spec", attend=True, do_legend=True, mark_sample=MARK)
        plt.legend(bbox_to_anchor=(1.1, .5), loc='center left', borderaxespad=0.)
        plt.savefig(f"../results/figures/figure{which_figure}/corr_scatter_{n-1}_mark_legend.png", dpi = 300, bbox_inches='tight')

        plt.figure(figsize = (3.5, 3.5))
        ami_hist(f"condition_cumu-tpdwn_{n-1}_spec")
        plt.savefig(f"../results/figures/figure{which_figure}/ami_hist_{n-1}.png", dpi = 300, bbox_inches='tight')

    #"""

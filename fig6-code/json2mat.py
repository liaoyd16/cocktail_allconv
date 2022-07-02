
import argparse
import json
import numpy as np
import scipy
import scipy.io as io
import pdb

if __name__ == '__main__':
	# argparser
	parser = argparse.ArgumentParser(description='cumu/single=?, n=?, k=?, m=0/de?')
	parser.add_argument('cumu_single', type=str, nargs=1)
	parser.add_argument('n', type=int, nargs=1)
	parser.add_argument('k', type=int, nargs=1)
	parser.add_argument('m', type=str, nargs=1)
	args = parser.parse_args()

	# 图位置
	cumu_single = args.cumu_single[0]
	n = args.n[0]

	# 一条图线
	k = args.k[0]
	m = args.m[0] # 0

	# 每个图点: template for attention projection range
	json_name_template = 'C:\\Users\\Lenovo\\Desktop\\dumped_{}\\n={}_en={}_resblock_3x3conv_5shortcut\\de={}\\condition_{}-tpdwn_{}_spec_ami.json';
	
	AMI_test_array = []
	# pdb.set_trace()
	for icond in range(n):
		AMI_test = json.load(open(json_name_template.format(cumu_single, n, k, m, cumu_single, icond)));
		AMI_test_array.append(AMI_test)
	AMI_test_array = np.array(AMI_test_array)
	
	# 图线数据
	mat_place = 'C:\\Users\\Lenovo\\Desktop\\cocktail_windows\\mat\\mat\\condition_{}-tpdwn\\n={}\\en={}_de={}.mat'
	io.savemat(mat_place.format(cumu_single, n,k,m), {'ami_array':AMI_test_array})
	print("dump to mat succeed: shape={}".format(AMI_test_array.shape))
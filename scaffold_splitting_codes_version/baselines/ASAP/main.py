import math, random, argparse, time, uuid
import os, os.path as osp
from helper import makeDirectory, set_gpu

import numpy as np
import torch
from torch.nn import functional as F
from sklearn.model_selection import KFold
from torch_geometric.data import DataLoader, DenseDataLoader as DenseLoader
from torch_geometric.datasets import TUDataset
from asap_pool_model import ASAP_Pool

torch.backends.cudnn.benchmark = False

from torch_geometric.utils import degree
import torch_geometric.transforms as T


from dataset_utils.chem_datasets import MoleculeDataset
from dataset_utils.chem_pre import pre_filter, pre_transform
import numpy as np
from dataset_utils.chem_splitters import random_scaffold_split
from dataset_utils.utils import auc_metric
import pandas as pd
import os
from torch_geometric.data import DataLoader




class Trainer(object):

	def __init__(self, params, graph_data):
		self.p = params

		# set GPU
		if self.p.gpu != '-1' and torch.cuda.is_available():
			self.device = torch.device('cuda:%d' % int(self.p.gpu))
			torch.cuda.set_rng_state(torch.cuda.get_rng_state())
			torch.backends.cudnn.deterministic = True
		else:
			self.device = torch.device('cpu')

		# build the data
		self.p.use_node_attr = (self.p.dataset == 'FRANKENSTEIN')
		self.data = graph_data

		# build the model
		self.model	= None
		self.optimizer	= None


	# load model
	def addModel(self):
		if self.p.model == 'ASAP_Pool':
			model = ASAP_Pool(
				dataset=self.data,
				num_layers=self.p.num_layers,
				hidden=self.p.hid_dim,
				ratio = self.p.ratio,
				dropout_att = self.p.dropout_att,
				)
		
		else:
			raise NotImplementedError
		model.to(self.device).reset_parameters()
		return model

	def addOptimizer(self):
		return torch.optim.Adam(self.model.parameters(), lr=self.p.lr, weight_decay=self.p.l2)
		
	# train model for an epoch
	def run_epoch(self, loader, criterions, args):
		self.model.train()

		total_loss = 0
		for data in loader:
			self.optimizer.zero_grad()
			data	= data.to(self.device)
			out	= self.model(data)
			y = data.y.view(-1, args.num_tasks)

			loss = 0.0
			for i in range(args.num_tasks):
				is_valid = y[:, i] ** 2 > 0
				loss += criterions[i](out[is_valid, :, i], ((y[is_valid, i] + 1) / 2).long())

			loss.backward()
			total_loss += loss.item() * self.num_graphs(data)
			self.optimizer.step()
		return total_loss / len(loader.dataset)

	# validate or test model
	def predict(self, loader):
		self.model.eval()

		y_true = []
		y_scores = []
		for data in loader:
			data = data.to(self.device)
			with torch.no_grad():
				out = self.model(data)
				y = data.y.view(-1, args.num_tasks)

				y_true.append(y)
				y_scores.append(out)
		auc = auc_metric(y_true, y_scores)
		return auc

	# save model locally
	def save_model(self, save_path):
		state = {
			'state_dict': 	self.model.state_dict(),
			'optimizer' : 	self.optimizer.state_dict(),
			'args'      : 	vars(self.p)
		}
		torch.save(state, save_path)

	# load model from path
	def load_model(self, load_path):
		state		= torch.load(load_path)
		self.model.load_state_dict(state['state_dict'])
		self.optimizer.load_state_dict(state['optimizer'])



	def num_graphs(self, data):
		if data.batch is not None: 	return data.num_graphs
		else: 				return data.x.size(0)


	# main function for running the experiments
	def run(self):
		smiles_list = pd.read_csv(os.path.join(args.data_root, args.data_name, 'processed/smiles.csv'), header=None)[0].tolist()
		train_dataset, valid_dataset, test_dataset = random_scaffold_split(self.data, smiles_list, null_value=0,
																		   frac_train=0.8, frac_valid=0.1, frac_test=0.1)

		train_loader = DataLoader(train_dataset, args.batch_size, shuffle=True)
		valid_loader = DataLoader(valid_dataset, args.batch_size, shuffle=False)
		test_loader = DataLoader(test_dataset, args.batch_size, shuffle=False)


		best_valid = 0.0
		best_valid_test = 0.0

		self.model = self.addModel()
		self.optimizer = self.addOptimizer()
		criterions = [torch.nn.CrossEntropyLoss() for i in range(args.num_tasks)]

		if torch.cuda.is_available():
			torch.cuda.synchronize()


		for epoch in range(1, self.p.max_epochs + 1):
			train_loss = self.run_epoch(train_loader, criterions, self.p)
			valid_auc = self.predict(valid_loader)
			test_auc = self.predict(test_loader)

			# lr_decay
			if epoch % self.p.lr_decay_step == 0:
				for param_group in self.optimizer.param_groups:
					param_group['lr'] = self.p.lr_decay_factor * param_group['lr']

			if valid_auc >= best_valid:
				best_valid = valid_auc
				best_valid_test = test_auc
			if epoch % 10 == 0:
				print('epoch:{}, valid_auc:{:.4f}, test_auc:{:.4f}'.format(epoch, valid_auc, test_auc))
		return best_valid, best_valid_test

if __name__== '__main__':

	## the example of run command: python main.py -data clintox -gpu 0

	parser = argparse.ArgumentParser(description='Neural Network Trainer Template')

	parser.add_argument('-model', 	  	dest='model', 				default='ASAP_Pool',					help='Model to use')
	parser.add_argument('-data', 	  	dest='dataset', 			default='clintox',	type=str,			help='Dataset to use')
	parser.add_argument('-epoch', 	  	dest='max_epochs', 			default=100,   		type=int, 			help='Max epochs')
	parser.add_argument('-l2', 	  		dest='l2', 					default=5e-4, 		type=float, 		help='L2 regularization')
	parser.add_argument('-num_layers', 	dest='num_layers', 			default=2, 			type=int, 			help='Number of GCN Layers')
	parser.add_argument('-lr_decay_step', 	dest='lr_decay_step', 	default=50, 		type=int, 			help='lr decay step')
	parser.add_argument('-lr_decay_factor', dest='lr_decay_factor', default=0.5, 		type=float, 		help='lr decay factor')

	parser.add_argument('-batch', 	  	dest='batch_size', 			default=128,   		type=int, 			help='Batch size')
	parser.add_argument('-hid_dim', 	dest='hid_dim', 			default=64, 		type=int, 			help='hidden dims')
	parser.add_argument('-dropout_att',	dest='dropout_att', 		default=0.1, 		type=float,			help='dropout on attention scores')
	parser.add_argument('-lr',			dest='lr', 					default=0.01,  		type=float,			help='Learning rate')
	parser.add_argument('-ratio', 		dest='ratio', 				default=0.5, 		type=float, 		help='ratio')
	
	
	parser.add_argument('-folds', 	  	dest='folds', 				default=20, 		type=int, 			help='Cross validation folds')

	parser.add_argument('-name', 	  	dest='name', 				default='test_'+str(uuid.uuid4())[:8], 	help='Name of the run')
	parser.add_argument('-gpu', 	  	dest='gpu', 				default='0',							help='GPU to use')
	parser.add_argument('-restore',	  	dest='restore', 			action='store_true', 					help='Model restoring')
	
	args = parser.parse_args()
	if not args.restore:
		args.name = args.name + '_' + time.strftime('%d_%m_%Y') + '_' + time.strftime('%H:%M:%S')

	
	print('Starting runs...')
	print(args)

	args.data_root = '../../chem_data'
	args.data_name = args.dataset
	graph_data = MoleculeDataset(os.path.join(args.data_root, args.data_name), dataset=args.data_name,
								 pre_filter=pre_filter, pre_transform=pre_transform)

	args.num_features = graph_data.num_features
	args.num_classes = graph_data.num_classes
	args.num_tasks = int(graph_data.data.num_tasks[0])

	# get 20 run average
	seeds = [8971, 85688, 9467, 32830, 28689, 94845, 69840, 50883, 74177, 79585, 1055, 75631, 6825, 93188, 95426, 54514, 31467, 70597, 71149, 81994]

	counter = 0
	args.log_db = args.name
	print("log_db:", args.log_db)
	tests = []
	for seed in seeds:

		# set seed
		args.seed = seed
		random.seed(seed)
		np.random.seed(seed)
		torch.manual_seed(seed)
		set_gpu(args.gpu)

		args.name = '{}_run_{}'.format(args.log_db, counter)

		# start training the model
		model		= Trainer(args, graph_data)
		val_auc, test_auc	= model.run()
		print("{}-th run in all {} runs || Test_auc:{:.4f}".format(counter + 1, len(seeds), test_auc))
		tests.append(test_auc)
		counter += 1
	tests = np.array(tests)
	print("Final results --- Test_auc:{:.4f}±{:.4f}".format(tests.mean(), tests.std()))


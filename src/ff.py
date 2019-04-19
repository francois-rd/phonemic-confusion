from src.data_loader import DataLoader
import pickle
import torch
from torch import nn, optim, utils
import torch.nn.functional as F
import torch.utils.data
import numpy as np
import more_itertools
from sklearn.metrics import confusion_matrix

class FFNN(nn.Module):
	"""
	Architecture definition for fully-connected neural network
	"""
	def __init__(self, forward_context, backward_context, embed_dim, phoneme_count, error_weight):
		super().__init__()

		# Parameters
		self.forward_context = forward_context
		self.backward_context = backward_context
		self.embed_dim = embed_dim
		self.phoneme_count = phoneme_count
		self.error_weight = error_weight

		# Layers
		self.embed = nn.Embedding(self.phoneme_count, self.embed_dim)
		self.linear1 = nn.Linear(self.embed_dim * (self.forward_context + self.backward_context + 1), 512)
		self.linear2 = nn.Linear(512, 256)
		self.linear3 = nn.Linear(256, 256)
		self.linear4 = nn.Linear(256, 128)
		self.linear5 = nn.Linear(128, 128)
		self.linear6 = nn.Linear(128, 64)
		self.linear7 = nn.Linear(64, 64)
		self.linear8 = nn.Linear(64, 1)

	def forward(self, inputs):
		out = self.embed(inputs)
		out = torch.reshape(out, (-1, self.embed_dim * (self.forward_context + self.backward_context + 1)))
		out = F.relu(self.linear1(out))
		out = F.relu(self.linear2(out))
		out = F.relu(self.linear3(out))
		out = F.relu(self.linear4(out))
		out = F.relu(self.linear5(out))
		out = F.relu(self.linear6(out))
		out = F.relu(self.linear7(out))
		out = torch.sigmoid(self.linear8(out))

		return out

	def save(self, path):
		"""
		Saves model to a file.
		
		:param path: path to file where to save model
		"""
		settings = {'model': FFNN(self.forward_context, self.backward_context, self.embed_dim, 
			self.phoneme_count, self.error_weight), 'state_dict': self.state_dict()}
		torch.save(settings, path)

	@classmethod
	def load(cls, path):
		"""
		Loads model (oppposite of 'save' function).

		:param path: path to file where model is saved
		"""
		file = torch.load(path)
		model = file['model']
		model.load_state_dict(file['state_dict'])
		return model

def data_transformer(data, forward_context, backward_context, vocab_size):
	"""
	Given list of list of phonemes representing phrases, as well as binary labels specifying 
	whether the phoneme was transcribed correctly, splits into smaller lists with appropriate 
	padding to form uniformly-sized tensor
	
	:param data: the data which padding should be applied to (tensor with input and output)
	:param forward_context: the amount of padding to add after the last phoneme
	:param backward_context: the amount of padding to add before the first phoneme
	"""
	START_TOKEN = vocab_size - 2 # TO DO: double check these subtractions
	END_TOKEN = vocab_size - 1
	start_padding = forward_context
	end_padding = backward_context

	x, y = data
	
	inputs = []
	for phrase in x:
		padded_phrase = [START_TOKEN] * start_padding + phrase + [END_TOKEN] * end_padding
		inputs.extend(list(more_itertools.windowed(padded_phrase, n = 1 + start_padding + end_padding)))
	
	inputs = torch.tensor(inputs, dtype = torch.int64)
	targets = torch.tensor(np.concatenate(y), dtype = torch.float).reshape(-1, 1)

	return inputs, targets

def evaluate_model(model, validation_data, forward_context, backward_context, vocab_size, error_weight):
	"""
	Helper function to evaluate trained model on validation (or test) data

	:param model: trained model to use for predictions
	:param validation_data: dataloader with input and ground truth pairs
	:param forward_context: Number of phonemes after (context)
	:param backward_context: Number of phonemes before (context)
	:param vocab_size: Number of token types
	:param error_weight: Multiplier for the penalty for not identifying errors correctly vs. identifying non-errors correctly
	"""

	val_targets = torch.tensor([])
	val_predictions = torch.tensor([])
	
	for minibatch in validation_data:
		new_inputs, new_targets = data_transformer(minibatch, forward_context, backward_context, vocab_size)
		new_predictions = model(new_inputs)
		val_targets = torch.cat((val_targets, new_targets))
		val_predictions = torch.cat((val_predictions, new_predictions))

	val_weights = torch.where(val_targets == 1, torch.tensor(error_weight, dtype = torch.float), torch.tensor(1, dtype = torch.float))
	validation_loss = float(F.binary_cross_entropy(val_predictions, val_targets, weight = val_weights, reduction = 'sum').detach()) / len(val_predictions)
	tn, fp, fn, tp = confusion_matrix(y_true = val_targets.detach().numpy(), y_pred = np.round(val_predictions.detach().numpy())).ravel()

	return validation_loss, tn, fp, fn, tp

def train(model_parameters):
	"""
	Trains neural network given a dictionary of parameters with the following values.

	:param data_dir: Where transcripts are stored
	:param phoneme_out: Phoneme translation file
	:param forward_context: Number of phonemes after (context)
	:param backward_context: Number of phonemes before (context)
	:param embed_dim: Number of dimensions in phoneme embedding
	:param num_epochs: Number of epochs to train neural network for
	:param batch_size: Number of phrases to process at once (split into phonemes)
	:param error_weight: Multiplier for the penalty for not identifying errors correctly vs. identifying non-errors correctly
	:param weights_path: Location where to save models during training
	:param print_batch: Frequency with which to print updates
	:param loss_multiplier: Multiplying constant to make reading loss values easier
	:param save_epoch: Model is saved at any multiple of this number of epochs
	"""
	#Unpacking variables
	DATA_DIR = model_parameters['data_dir']
	PHONEME_OUT = model_parameters['phoneme_out']
	forward_context = model_parameters['forward_context']
	backward_context = model_parameters['backward_context']
	embed_dim = model_parameters['embed_dim']
	num_epochs = model_parameters['num_epochs']
	batch_size = model_parameters['batch_size']
	error_weight = model_parameters['error_weight']
	weights_path = model_parameters['weights_path']
	print_batch = model_parameters['print_batch']
	loss_multiplier = model_parameters['loss_multiplier']
	save_epoch = model_parameters['save_epoch']

	# Instantiation
	print('Setting everything up...')
	vocab_size = DataLoader(DATA_DIR, PHONEME_OUT, 'hypothesis', 'full', 'train', batch_size = batch_size).vocab_size + 2 # Adding 2 for start/end padding
	training_data = DataLoader(DATA_DIR, PHONEME_OUT, 'hypothesis', 'binary', 'train', batch_size = batch_size)
	validation_data = DataLoader(DATA_DIR, PHONEME_OUT, 'hypothesis', 'binary', 'val', batch_size = batch_size)

	model = FFNN(forward_context = forward_context, backward_context = backward_context, embed_dim = embed_dim, phoneme_count = vocab_size, error_weight = error_weight)
	opt = optim.Adam(model.parameters())

	# Model training
	print('Starting training!')
	i = 1
	best_validation_loss = np.inf

	while i <= num_epochs:

		training_loss = 0
		mb = 1
		predictions_made = 0

		print('######################################################################', '\nEpoch', 
			i, '\n######################################################################')

		for minibatch in training_data:
			opt.zero_grad()

			inputs, targets = data_transformer(minibatch, forward_context, backward_context, vocab_size)
			predictions = model(inputs)
			weights = torch.where(targets == 1, torch.tensor(error_weight, dtype = torch.float), torch.tensor(1, dtype = torch.float))
			loss = F.binary_cross_entropy(predictions, targets, weight = weights, reduction = 'sum')

			loss.backward()
			opt.step()

			training_loss += float(loss.detach())
			
			mb += 1
			predictions_made += len(predictions)

			if mb % print_batch == 0:
				print('Minibatch', mb, 'Cumulative Training Loss:', training_loss / predictions_made * loss_multiplier)
				tn, fp, fn, tp = confusion_matrix(y_true = targets.detach().numpy(), y_pred = np.round(predictions.detach().numpy())).ravel()
				print('TN:', tn, '| FP:', fp, '| FN:', fn, '| TP:', tp, '| Accuracy:', (tn + fn) / (tn + fp + fn + tp))

		validation_loss, tn, fp, fn, tp = evaluate_model(model, validation_data, forward_context, backward_context, vocab_size, error_weight)

		if i % save_epoch == 0:
			location = weights_path + 'epoch_' + str(i) + '_' + str(round(validation_loss * loss_multiplier, 3)) + '.weights'
			model.save(location)

		print('######################################################################', '\nEpoch', i, 
			'Training Loss:', training_loss / predictions_made * loss_multiplier, 
			'| Validation Loss:', validation_loss * loss_multiplier)
		print('TN:', tn, '| FP:', fp, '| FN:', fn, '| TP:', tp, '| Accuracy:', (tn + fn) / (tn + fp + fn + tp))

		i += 1
	
def investigate_model(model_parameters, file):
	"""
	Given a trained model name, outputs its performance on validation (test) data

	:param model_parameters: dictionary with model parameters:
		Uses data directory, phoneme file, weights_path, batch size, and loss_multiplier
	:param file: file path where model is saved
	"""
	DATA_DIR = model_parameters['data_dir']
	PHONEME_OUT = model_parameters['phoneme_out']
	batch_size = model_parameters['batch_size']
	loss_multiplier = model_parameters['loss_multiplier']
	weights_path = model_parameters['weights_path']

	print('Hold on. Fetching results!')

	validation_data = DataLoader(DATA_DIR, PHONEME_OUT, 'hypothesis', 'binary', 'val', batch_size = batch_size)

	model = FFNN.load(weights_path + file)
	validation_loss, tn, fp, fn, tp = evaluate_model(model, validation_data, model.forward_context, 
		model.backward_context, model.phoneme_count, model.error_weight)

	print('######################################################################', '\nValidation Loss:', validation_loss * loss_multiplier)
	print('TN:', tn, '| FP:', fp, '| FN:', fn, '| TP:', tp, '| Accuracy:', (tn + fn) / (tn + fp + fn + tp))


torch.manual_seed = 1
model_parameters = {
	'forward_context':	4,
	'backward_context':	4,
	'embed_dim':		15,
	'num_epochs':		100,
	'batch_size':		32,
	'error_weight':		10,
	'weights_path':		'C:/Users/Ilan/Desktop/MSc Statistics/Neural Networks/Final Project Code/models/',
	'data_dir':			'C:/Users/Ilan/Desktop/MSc Statistics/Neural Networks/Final Project Code/data/transcripts/',
	'phoneme_out':		'C:/Users/Ilan/Desktop/MSc Statistics/Neural Networks/Final Project Code/data/phonemes.txt',
	'print_batch':		100,
	'loss_multiplier':	10**3,
	'save_epoch':		1
}

train(model_parameters)
#investigate_model(model_parameters, file = 'epoch_2_1206.231.weights')
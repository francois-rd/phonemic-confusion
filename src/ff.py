from src.data_loader import DataLoader
import pickle
import torch
from torch import nn, optim, utils
import torch.nn.functional as F
import torch.utils.data
import numpy as np
import more_itertools
from sklearn.metrics import confusion_matrix, roc_curve
from matplotlib import pyplot

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
	full_phrases = []
	for phrase in x:
		padded_phrase = [START_TOKEN] * start_padding + phrase + [END_TOKEN] * end_padding
		new_sub_phrases = list(more_itertools.windowed(padded_phrase, n = 1 + start_padding + end_padding))
		inputs.extend(new_sub_phrases)
		full_phrases.extend([phrase] * len(new_sub_phrases))
	
	inputs = torch.tensor(inputs, dtype = torch.int64)
	targets = torch.tensor(np.concatenate(y), dtype = torch.float).reshape(-1, 1)

	return inputs, targets, full_phrases

def phonemes_to_phrase(show_transcription = True):
	"""
	Returns a dictionary that converts a list of phonemes into the English phrase that created it 
	
	:param show_transcription: Whether ground truth phrases or both ground truth phrases and transcription should be shown
	"""

	from align_phonemes import load_data, load_phonemes, text2phonemes
	from src.phonemes import DATA_DIR, FILES, PHONEME_OUT

	true_phrases, transcribed_phrases = load_data(DATA_DIR, FILES)
	phoneme_dictionary = load_phonemes(PHONEME_OUT)
	
	if show_transcription:
		phoneme2phrase = {text2phonemes(true_phrases[i], phoneme_dictionary) : (true_phrases[i], transcribed_phrases[i]) for i in range(len(true_phrases))}
	else:
		phoneme2phrase = {text2phonemes(phrase, phoneme_dictionary) : phrase for phrase in true_phrases}

	return phoneme2phrase

def ROC(y_target, y_probability, path):
	"""
	Returns data to plot ROC curve

	:params y_target: The true label
	:params y_probability: The predicted probability
	:params path: Where to save the text file with ROC curve data
	"""
	fpr, tpr, thresholds = roc_curve(y_true = y_target, y_score = y_probability)

	with open(path, 'w') as f:
		f.write('FPR,TPR,THRESHOLDS')
		for i in range(1, len(fpr), 500): # Ignoring first row due to arbitrary calculation, skipping rows for small file
			f.write('\n' + str(fpr[i]) + ',' + str(tpr[i]) + ',' + str(thresholds[i]))

	return fpr, tpr, thresholds

def evaluate_model(model, validation_data, forward_context, backward_context, vocab_size, 
	error_weight, show_transcription, roc = None, decoder = None, phrase_count = 0):
	"""
	Helper function to evaluate trained model on validation (or test) data

	:param model: trained model to use for predictions
	:param validation_data: dataloader with input and ground truth pairs
	:param forward_context: Number of phonemes after (context)
	:param backward_context: Number of phonemes before (context)
	:param vocab_size: Number of token types
	:param error_weight: Multiplier for the penalty for not identifying errors correctly vs. identifying non-errors correctly
	:param show_transcription: Whether ground truth phrases or both ground truth phrases and transcription should be shown
	:param roc: Whether to write ROC curve information to a file (provide path if so)
	:param decoder: Dictionary with integers as keys and phonemes as values; must be provided if phrases = True
	:param phrases: Number of maximum and minimum predicted error phrases to print
	"""

	val_targets = torch.tensor([])
	val_predictions = torch.tensor([])
	val_inputs = torch.tensor([], dtype = torch.int64)
	val_full_phrases = []

	for minibatch in validation_data:
		new_inputs, new_targets, new_full_phrases = data_transformer(minibatch, forward_context, backward_context, vocab_size)
		new_predictions = model(new_inputs)
		val_targets = torch.cat((val_targets, new_targets))
		val_predictions = torch.cat((val_predictions, new_predictions))
		val_inputs = torch.cat((val_inputs, new_inputs))
		val_full_phrases.extend(new_full_phrases)

	if phrase_count > 0 and decoder:
		sort_ids = np.argsort(val_predictions.detach().numpy().ravel())
		phoneme2phrase = phonemes_to_phrase(show_transcription)

		print('######################################################################')
		print('Top', str(phrase_count), 'Predicted Least Confusing Phrases')
		print('Predicted Confusion:')
		print(val_predictions.detach().numpy()[sort_ids[:phrase_count]])
		
		print('Phrases:')
		phrases = val_inputs.detach().numpy()[sort_ids[:phrase_count]]
		print(phrases)
		
		print('Translated Phrases:')
		print(np.vectorize(decoder.get)(phrases))

		print('English Phrases (True, Transcribed):')
		phoneme_list = [[decoder[integer] for integer in val_full_phrases[phrase]] for phrase in sort_ids[:phrase_count]]
		print([phoneme2phrase[' '.join(phonemes)] for phonemes in phoneme_list])
		
		print('Actual Confusion?')
		print(val_targets.detach().numpy()[sort_ids[:phrase_count]])

		print('######################################################################')
		print('Top', str(phrase_count), 'Predicted Most Confusing Phrases')
		print('Predicted Confusion:')
		print(np.flip(val_predictions.detach().numpy()[sort_ids[-phrase_count:]], axis = 0))
		
		print('Phrases:')
		phrases = np.flip(val_inputs.detach().numpy()[sort_ids[-phrase_count:]], axis = 0)
		print(phrases)
		
		print('Translated Phrases:')
		print(np.vectorize(decoder.get)(phrases))		
		
		print('English Phrases (True, Transcribed):')
		phoneme_list = np.flip([[decoder[integer] for integer in val_full_phrases[phrase]] for phrase in sort_ids[-phrase_count:]], axis = 0)
		print([phoneme2phrase[' '.join(phonemes)] for phonemes in phoneme_list])

		print('Actual Confusion?')
		print(np.flip(val_targets.detach().numpy()[sort_ids[-phrase_count:]], axis = 0))

	if roc:
		ROC(val_targets.detach().numpy(), val_predictions.detach().numpy(), roc)

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

			inputs, targets, _ = data_transformer(minibatch, forward_context, backward_context, vocab_size)
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
		Uses data directory, phoneme file, weights_path, batch size, loss_multiplier, and ROC location (where to write file)
	:param file: file path where model is saved
	"""
	DATA_DIR = model_parameters['data_dir']
	PHONEME_OUT = model_parameters['phoneme_out']
	batch_size = model_parameters['batch_size']
	loss_multiplier = model_parameters['loss_multiplier']
	weights_path = model_parameters['weights_path']
	roc = model_parameters['roc_location']

	print('Hold on. Fetching results!')

	validation_data = DataLoader(DATA_DIR, PHONEME_OUT, 'hypothesis', 'binary', 'test', batch_size = batch_size)
	int_to_phoneme = dict((v,k) for k,v in validation_data.phoneme_to_int.items())

	model = FFNN.load(weights_path + file)
	validation_loss, tn, fp, fn, tp = evaluate_model(model, validation_data, model.forward_context, 
		model.backward_context, model.phoneme_count, model.error_weight, roc = roc, show_transcription = True, decoder = int_to_phoneme, phrase_count = 5)

	print('######################################################################', '\nValidation Loss:', validation_loss * loss_multiplier)
	print('TN:', tn, '| FP:', fp, '| FN:', fn, '| TP:', tp, '| Accuracy:', (tn + fn) / (tn + fp + fn + tp))

def investigate_data(model_parameters, path):
	"""
	Returns number of phonemes correctly and incorrectly transcribed in dataset, by phoneme

	:param model_parameters: dictionary with model parameters:
		Uses data directory, phoneme file, batch size
	:param path: file where phoneme data saved
	"""
	from collections import Counter

	DATA_DIR = model_parameters['data_dir']
	PHONEME_OUT = model_parameters['phoneme_out']
	batch_size = model_parameters['batch_size']

	train_data = DataLoader(DATA_DIR, PHONEME_OUT, 'hypothesis', 'binary', 'train', batch_size = batch_size)
	#int_to_phoneme = dict((v,k) for k,v in train_data.phoneme_to_int.items())

	cnt_correct = Counter()
	cnt_incorrect = Counter()

	for x, y in train_data:
		x = [phoneme for phrase in x for phoneme in phrase]
		y = [label for phrase in y for label in phrase]
		for i in range(len(x)):
			if y[i]:
				cnt_incorrect[int_to_phoneme[x[i]]] += 1
			else:
				cnt_correct[int_to_phoneme[x[i]]] += 1
	
	with open(path, 'w') as f:
		f.write('PHONEME,CORRECT,INCORRECT')
		for key, value in cnt_correct.items():
			f.write('\n' + str(key) + ',' + str(value) + ',' + str(cnt_incorrect[key]))

torch.manual_seed = 1
model_parameters = {
	'forward_context':	4,
	'backward_context':	4,
	'embed_dim':		15,
	'num_epochs':		100,
	'batch_size':		32,
	'error_weight':		10,
	'weights_path':		'../models/',
	'data_dir':			'../data/transcripts/',
	'phoneme_out':		'../data/phonemes.txt',
	'roc_location':		'../roc.csv',
	'print_batch':		100,
	'loss_multiplier':	10**3,
	'save_epoch':		1
}


train(model_parameters)
investigate_model(model_parameters, file = 'epoch_7_1135.176.weights')
#investigate_data(model_parameters, '../diagrams/phoneme_frequency.csv')
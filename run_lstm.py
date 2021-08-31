import glob
import argparse
from tqdm import tqdm
import pickle

import numpy as np
import torch

from music21 import converter, note, chord, stream
from torch import nn, tensor, LongTensor
from torch.nn.functional import softmax


def preprocess_midi(data_dir):
    """
    Preprocess music file in midi format using music21
    @param data_dir: path to data
    @return: a list of compositions with note/chords and durations, and a mapping between note/chord: idx
    """
    training_data = []
    note_chord_combinations = set()
    starting_index = []


    for composition in glob.glob(data_dir):
        all_notes = []
        music_score = converter.parse(composition)
        starting_index.append(len(training_data))
        for element in music_score.flat.notes:
            next_element = []
            if type(element) == note.Note:
                note_pitch = str(element.pitch)
                next_element.append(note_pitch)
                note_chord_combinations.add(note_pitch)
            elif type(element) == chord.Chord:
                chord_pitch = tuple([str(e) for e in element.pitches])
                next_element.append(chord_pitch)
                note_chord_combinations.add(chord_pitch)
            next_element.append(float(element.quarterLength))
            next_element.append(float(element.volume.velocity))
            all_notes.append(next_element)
        training_data.append(all_notes)
    note_chord_mapping = {combo:idx for idx, combo in enumerate(note_chord_combinations)}
    return training_data, note_chord_mapping, note_chord_combinations, starting_index

def generate_music_sequence(training_data, melody_length, note_chord_mapping):
    """
    Generate input and target sequences based on given melody (sequence) length
    @param melody_length: length of melody sequence
    @param training_data: input training data
    @param note_chord_mapping: mapping between nodes/chords and an index
    @return: input and target sequence datasets

    *** Convert nodes/chords to integer (categorical) representations
    """

    input_dataset = []
    target_dataset = []
    duration_dataset = []
    velocity_dataset = []
    for composition in training_data:
        for idx in range(0, len(composition) - melody_length):
            melody_input = [note_chord_mapping[note] for note, duration, velocity in composition[idx: idx + melody_length]]
            melody_target = [note_chord_mapping[note] for note, duration, velocity in composition[idx + 1: idx + melody_length + 1]]
            input_dataset.append(melody_input)
            target_dataset.append(melody_target)
        duration_dataset.extend(duration for note, duration, velocity in composition)
        velocity_dataset.extend(velocity for note, duration, velocity in composition)
    input_dataset = LongTensor(input_dataset)
    target_dataset = LongTensor(target_dataset)
    return input_dataset, target_dataset, duration_dataset, velocity_dataset


class Encoder(nn.Module):
    """
    Customized Encoder that uses embedding and a feed foward linear layer
    """
    def __init__(self, source_melody_size, embedding_dim, num_layers, hidden_size, dropout):
        """
        @param source_melody_size: int, dimension of target output
        @param embedding_dim: int, dimension to embed the input data
        @param num_layers: int, number of hidden layers
        @param hidden_size: int, size of hidden state
        @param dropout: float, probability of dropping connection in the last layer
        """

        super().__init__()

        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(input_size=embedding_dim,
                            hidden_size=hidden_size,
                            num_layers=num_layers,
                            dropout=dropout)
        self.embeddings = nn.Embedding(num_embeddings=source_melody_size,
                                       embedding_dim=embedding_dim)

    def forward(self, melody_seq):
        """
        Pass the melody sequence to output format
        @param melody_seq: tensor, sequence_length * batch_size
        """
        embedded_seq = self.embeddings(melody_seq)
        output, _ = self.lstm(embedded_seq)
        return output


class Decoder(nn.Module):
    """
    Customized Decoder that uses teacher forcing and attention mechanism
    """
    def __init__(self, target_melody_size, embedding_dim, hidden_size):
        """
        @param target_melody_size: int, number of possible notes/chords in the target melody
        @param embedding_dim: int, dimension to embed the input data
        @param hidden_size: int, size of hidden state
        """

        super().__init__()

        self.hidden_size = hidden_size
        self.lstm_cell = nn.LSTMCell(input_size=embedding_dim + hidden_size, hidden_size=hidden_size)
        self.embeddings = nn.Embedding(num_embeddings=target_melody_size, embedding_dim=embedding_dim)
        self.feed_forward = nn.Linear(in_features=hidden_size, out_features=target_melody_size)

    def get_init_hidden_state(self, prev_target):
        """
        Initialize a hidden state
        @param prev_target: tensor, previous target in the sequence
        @return: tuple of LSTM hidden, cell state
        """
        return (torch.zeros((prev_target.shape[0], self.hidden_size)).to(prev_target.device),
                torch.zeros((prev_target.shape[0], self.hidden_size)).to(prev_target.device))

    def forward(self, prev_target, prev_hidden, enc_hidden):
        """
        Forward pass the data
        @param prev_target: tensor, previous target melody in the sequence
        @param prev_hidden: tensor, previous hidden state in the decoder
        @param enc_hidden: tensor, all encoder hidden states
        @return:
        """

        if prev_hidden is None:
            prev_hidden = self.get_init_hidden_state(prev_target)

        # Update decoder input
        cur_input = self.get_decoder_input(prev_target, prev_hidden, enc_hidden)

        # Update decoder hidden state
        cur_hidden = self.get_decoder_hidden(prev_hidden, cur_input)

        # Find unnormalized probability through feed_forward network
        prob_unnormed = self.feed_forward(cur_hidden[0])

        return prob_unnormed, cur_hidden

    def get_decoder_input(self, prev_target, prev_hidden, enc_hidden):
        """
        Get the current input for the decoder network
        @param prev_target: tensor, previous target melody in the sequence
        @param prev_hidden: tensor, previous hidden state in the decoder
        @param enc_hidden: tensor, all encoder hidden states
        @return:
        """

        # Embedded input
        cur_input = self.embeddings(prev_target)

        # Context vector with attention mechanism
        context = self.find_context(prev_hidden, enc_hidden)

        # Concatenate together and use as input
        cur_input = torch.cat((cur_input, context), dim=1).to(prev_target.device)
        return cur_input

    def get_decoder_hidden(self, prev_hidden, cur_input):
        """
        Get the current hidden state for decoder
        @param prev_hidden: tensor, previous hidden state in the decoder
        @param cur_input: tensor, current input data
        @return: tuple of (hidden, cell) state
        """
        return self.lstm_cell(cur_input, prev_hidden)

    def find_context(self, prev_hidden, enc_hidden):
        """
        Find the weights of a linear combination of the encoder states

        @param prev_hidden: tensor, previous hidden state in the decoder
        @param enc_hidden: tensor, all encoder hidden states
        @return: context vector
        """
        weights = self.get_attention_weights(prev_hidden, enc_hidden).T.unsqueeze(dim=1)
        return torch.matmul(weights, enc_hidden.transpose(0, 1)).squeeze(dim=1)

    def get_attention_weights(self, prev_hidden, enc_hidden):
        """
        Use cosine similarities to find the weight for each encoder hidden state
        @return: weights that sum to 1
        """
        similarity = nn.CosineSimilarity(dim=-1)(prev_hidden[0], enc_hidden)
        return softmax(similarity, dim=0)

class EncoderDecoder(nn.Module):
    """
    Main structure of the Encoder-Decoder LSTM network
    """

    def __init__(self, source_melody_size, embedding_dim, num_layers, hidden_size, dropout):
        """
        Initialize encoder and decoder
        """
        super().__init__()
        self.encoder = Encoder(source_melody_size,
                               embedding_dim, num_layers, hidden_size, dropout)
        self.decoder = Decoder(source_melody_size, embedding_dim, hidden_size)
        self.source_melody_size = source_melody_size

    def switch_to_eval(self):
        """
        Switch to evaluation mode (deactivate dropout layer)
        """
        self.encoder.eval()
        self.decoder.eval()

    def teacher_forcing(self, enc_hidden, target_melody):
        """
        Input each element in the target melody to the decoder to avoid exposure bias from decoder output
        @return: unnormalized probability matrix
        """

        cur_hidden = None
        target_prob = torch.zeros(target_melody.shape[0] - 1, enc_hidden.shape[1], self.source_melody_size).to(enc_hidden.device)
        for t in range(1, target_melody.shape[0]):
            cur_prob, cur_hidden = self.decoder(target_melody[t - 1, :], cur_hidden, enc_hidden)
            target_prob[t - 1, :, :] = cur_prob
        return target_prob.to(enc_hidden.device)

    def forward(self, source_melody, target_melody):
        """
        Forward pass the source and target melody
        """
        enc_hidden = self.encoder(source_melody)
        return self.teacher_forcing(enc_hidden, target_melody)

    def predict(self, source_melody, seq_length, note_chord_combinations, arg_max=False):
        """
        Given a source melody and a desired sequence length, generate a music sequence

        @param arg_max: selects the next "note/chord" with the highest probability if set to True,
                        otherwise select according to the probability matrix
        @return: generated music sequence
        """
        enc_hidden = self.encoder(source_melody)
        cur_hidden = None
        predict_seq = torch.zeros(len(source_melody) + seq_length, 1).to(torch.long).to(enc_hidden.device)
        predict_seq[:len(source_melody), :] = source_melody

        for i in range(len(source_melody), len(source_melody) + seq_length - 1):
            cur_logit, cur_hidden = self.decoder(predict_seq[i, :], cur_hidden, enc_hidden)

            prob_norm = softmax(cur_logit, -1)
            if arg_max:
                predict_seq[i + 1, :] = torch.argmax(prob_norm)
            else:
                prob_norm = prob_norm.detach().cpu().numpy().flatten()
                next_note = np.random.choice(np.arange(0, len(list(note_chord_combinations))),
                                             p=prob_norm)
                predict_seq[i + 1, :] = next_note

            # Update encoder states every X iterations
            if i > 0 and i % len(source_melody) == 0:
                source_melody = predict_seq[i - len(source_melody):i + 1, :]
                enc_hidden = self.encoder(source_melody)

        return predict_seq.flatten()


def train_for_epoch(model, input_data, target_data, optimizer, device, batch_size=256):
    """
    Training for each epoch
    @param model: LSTM model
    @param optimizer: default Adam
    @param device: default Cuda
    @param batch_size: default=256
    @return: mean loss per batch
    """

    # Define loss function
    loss_fn = nn.CrossEntropyLoss(reduction="mean").to(device)

    # Compute average loss over epoch
    tracking_loss = []

    # Randomize the batch
    shuffle = torch.randperm(input_data.size()[0])

    for idx in tqdm(range(0, input_data.shape[0], batch_size), position=0, leave=True):
        optimizer.zero_grad()

        # Find random batch
        rand_indices = shuffle[idx:idx + batch_size]
        input_batch, output_batch = input_data[rand_indices], target_data[rand_indices]

        input_batch = input_batch.T.to(device)
        output_batch = output_batch.T.to(device)

        logits = model(input_batch, output_batch)

        # Optimization
        loss = loss_fn(logits.permute(1, 2, 0), output_batch[1:, :].T)
        tracking_loss.append(loss)

        loss.backward()
        optimizer.step()

    return torch.mean(tensor(tracking_loss))


def train_network(input_data, target_data, cuda, source_melody_size, embedding_dim=256, num_layers=2, hidden_size=512, dropout=0.1, num_epochs=20, batch_size=256):
    """
    Training the network from scratch
    @param input_data: input data in Tensor
    @param target_data: target data in Tensor
    @param cuda: device, default is cuda
    @param source_melody_size: int, size of the number of distinct notes/chords
    @param num_epochs: int, default=20
    @param batch_size: int, default=256
    @return:
    """
    model = EncoderDecoder(
        source_melody_size=source_melody_size,
        embedding_dim=embedding_dim,
        hidden_size=hidden_size,
        num_layers=num_layers,
        dropout=dropout)
    optimizer = torch.optim.Adam(model.parameters())

    input_data = input_data.to(cuda)
    target_data = target_data.to(cuda)
    model.to(cuda)

    for epoch in range(num_epochs):
        epoch_loss = train_for_epoch(model, input_data, target_data, optimizer, cuda, batch_size)
        print(f"Epoch: {epoch}, Loss: {epoch_loss}")
    torch.save(model.state_dict(), 'lstm_music.pkl')
    return model


def music_seq_production(input_dataset, input_idx, duration_dataset, velocity_dataset, music_seq_model, cuda, note_chord_combinations,
                         output_path="generated_music.mid",  composition_length=256, argmax=False):
    """
    Output a midi formatted music file with the first part of the melody composed by Chopin and 2nd part by algorithm
    """
    start = np.random.choice(input_idx)
    melody_seq = input_dataset[start].view(-1, 1).to(cuda)
    duration_seq = duration_dataset[start: start + len(melody_seq) + composition_length]
    velocity_seq = velocity_dataset[start: start + len(melody_seq) + composition_length]
    predicted_seq = music_seq_model.predict(melody_seq, composition_length, note_chord_combinations, arg_max=argmax)
    original_seq = torch.cat([input_dataset[k] for k in range(start, start + composition_length, len(melody_seq))])
    convert_to_midi(predicted_seq, duration_seq, velocity_seq, note_chord_combinations, output_path)
    convert_to_midi(original_seq, duration_seq, velocity_seq, note_chord_combinations, "original_" + output_path)


def random_music(note_chord_combinations, output_path="random_music.mid", composition_length=256):
    """
    Generate random music for comparison purpose
    """
    random_piece = np.random.randint(0, len(list(note_chord_combinations)), composition_length)
    convert_to_midi(random_piece, [0.5] * len(random_piece), note_chord_combinations, output_path)


def convert_to_midi(melody_seq, duration_seq, velocity_seq, note_chord_combinations, output_path):
    """
    Convert a list of notes/chords to a midi file
    """
    note_chord_mapping_rev = {idx: combo for idx, combo in enumerate(note_chord_combinations)}
    music_piece = [note_chord_mapping_rev[int(symbol)] for symbol in melody_seq]
    # Convert to midi file
    music_stream = stream.Part()
    for idx, symbol in enumerate(music_piece):
        if type(symbol) != tuple:
            next_note = note.Note(symbol)
        else:
            next_note = chord.Chord(symbol)
        next_note.quarterLength = duration_seq[idx]
        next_note.volume.velocity = velocity_seq[idx]
        music_stream.append(next_note)
    final_output = stream.Score()
    final_output.insert(0, music_stream)
    final_output.write('midi', fp=output_path)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input_dir", help="input folder", default="chopin/*.mid")
    parser.add_argument("-l", "--length", help="melody length", default=64, type=int)
    parser.add_argument("-device", "--device", help="melody length", default='cuda')
    parser.add_argument("-embedding", "--embedding_size", help="embedding size", default=256, type=int)
    parser.add_argument("-num_layers", "--num_layers", help="number of layers in LSTM", default=2, type=int)
    parser.add_argument("-dropout", "--dropout", help="dropout probability", default=0.1, type=float)
    parser.add_argument("-hidden", "--hidden_size", help="size of hidden state", default=1024, type=int)
    parser.add_argument("-epoch", "--num_epochs", help="number of epochs", default=20, type=int)
    parser.add_argument("-batch", "--batch_size", help="batch size", default=256, type=int)
    parser.add_argument("-sample", "--sample_size", help="number of samples", default=5, type=int)

    args = parser.parse_args()
    device = torch.device(args.device)

    print("Preprocessing Data")
    train_data, note_mapping, note_chord_combinations, starting_index = preprocess_midi(args.input_dir)

    # with open('train_data.pkl', 'wb') as f:
    #     pickle.dump(train_data, f)
    #
    # with open('note_mapping.pkl', 'wb') as f:
    #     pickle.dump(note_mapping, f)
    #
    # with open('note_chord_combinations.pkl', 'wb') as f:
    #     pickle.dump(note_chord_combinations, f)

    print("Generating Music Sequence")
    input_data, target_data, duration_data, velocity_data = generate_music_sequence(train_data, args.length, note_mapping)
    train_idx = list(np.random.choice(np.arange(0, len(input_data)), int(len(input_data) * 0.8), replace=False))
    test_idx = list(set(np.arange(0, len(input_data))) - set(train_idx))
    train_data, train_target = input_data[train_idx, :], target_data[train_idx, :]

    print("Training Model")
    model = train_network(train_data, train_target, device, source_melody_size=len(list(note_mapping)), embedding_dim=args.embedding_size,
                          num_layers=args.num_layers, hidden_size=args.hidden_size, dropout=args.dropout,
                          num_epochs=args.num_epochs, batch_size=args.batch_size)
    model.switch_to_eval()

    print("Producing Samples")
    for i in range(args.sample_size):
        music_seq_production(input_data, starting_index[1:-1], duration_data, velocity_data, model, device, note_chord_combinations, output_path=f"beeth_sample_{i}.mid")
        # music_seq_production(input_data, test_idx, duration_data, velocity_data, model, device, note_chord_combinations, output_path=f"prob_transition_{i}.mid")

    # random_music(note_chord_combinations)


import os
from argparse import ArgumentParser, Namespace
from glob import glob

import librosa
import matplotlib.pyplot as plt
import numpy as np
import pretty_midi
import torch
from transformers import Wav2Vec2FeatureExtractor
from tqdm import tqdm

from src.constants import CKPT_FILE, CATEGORIES, SAMPLE_RATE, RESULT_DIR
from src.model import MERTClassifier
from src.utils import read_json


class_idx2MIDIClass = read_json('hw1/class_idx2MIDIClass.json')
idx2instrument_class = read_json('hw1/idx2instrument_class.json')
MIDIClassName2class_idx = read_json('hw1/MIDIClassName2class_idx.json')


def parse_arguments() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument(
        '--test_track_dir',
        type=str,
        default='hw1/test_track',
        help='source(test) midi track folder path',
    )
    parser.add_argument(
        '--ckpt_dir',
        type=str,
        default='checkpoints/10-02-05-21-25',
    )
    parser.add_argument(
        '--thresholds',
        type=float,
        nargs='+',
        default=[0.5],
    )
    return parser.parse_args()


def extract_pianoroll_from_midi(midi_file_path, time_step=5.0):
    midi_data = pretty_midi.PrettyMIDI(midi_file_path)
    # print(midi_data)

    # Determine total duration in seconds
    total_time = midi_data.get_end_time()
    print("total time:", total_time)

    # Create an empty pianoroll matrix without the "Empty" class
    num_classes = len(class_idx2MIDIClass)
    num_time_steps = int(np.ceil(total_time / time_step))
    pianoroll = np.zeros((num_classes, num_time_steps))

    # Process each instrument in the MIDI file
    for instrument in midi_data.instruments:
        program_num = instrument.program

        if instrument.is_drum:
            instrument_class = 128
        else:
            # Determine the class for this instrument
            instrument_class = idx2instrument_class.get(str(program_num), None)
        if instrument_class and instrument_class in MIDIClassName2class_idx:
            class_idx = MIDIClassName2class_idx[instrument_class]

            # Fill the pianoroll for each note
            for note in instrument.notes:
                start_time = note.start
                end_time = note.end
                start_idx = int(np.floor(start_time / time_step))
                end_idx = int(np.ceil(end_time / time_step))
                pianoroll[class_idx, start_idx:end_idx] = 1  # Mark the note as present

    return pianoroll


def pianoroll_comparison(true_pianoroll, pred_pianoroll, save_path):

    _, axes = plt.subplots(2, 1, figsize=(15, 8))

    # Plotting the true pianoroll
    axes[0].imshow(true_pianoroll, aspect='auto', cmap='Oranges', interpolation='nearest')
    axes[0].set_title('True Labels')
    axes[0].set_yticks(range(len(CATEGORIES)))
    axes[0].set_yticklabels(CATEGORIES)
    axes[0].set_xlabel('Time Steps')

    # Plotting the predicted pianoroll
    axes[1].imshow(pred_pianoroll, aspect='auto', cmap='Oranges', interpolation='nearest')
    axes[1].set_title('Predicted Labels')
    axes[1].set_yticks(range(len(CATEGORIES)))
    axes[1].set_yticklabels(CATEGORIES)
    axes[1].set_xlabel('Time Steps')

    plt.tight_layout()
    plt.savefig(save_path)


def predict_pianoroll(audio_path, length, processor, model, device, chunk_duration=5, sample_rate=SAMPLE_RATE):
    audio_data, sr = librosa.load(audio_path, sr=sample_rate)
    total_samples = len(audio_data)
    chunk_samples = chunk_duration * sr
    num_chunks = min(int(total_samples / chunk_samples), length)
    print("total samples:", total_samples)

    preds = []
    for chunk_idx in tqdm(range(num_chunks)):
        audio_chunk = audio_data[
            chunk_idx * chunk_samples: int(min((chunk_idx + 1) * chunk_samples, total_samples))
        ]

        if len(audio_chunk) < total_samples:
            audio_chunk = np.pad(audio_chunk, (0, chunk_samples - len(audio_chunk)), 'constant')

        inputs = processor(audio_chunk, return_tensors="pt", sampling_rate=sr, padding=True).input_values.to(device)
        pred = model.predict(inputs).cpu().numpy()
        preds.append(pred)

    return np.vstack(preds).T


def main(args):
    os.makedirs(RESULT_DIR, exist_ok=True)
    audio_path_list = glob(os.path.join(args.test_track_dir, '*.flac'))
    config = read_json(os.path.join(args.ckpt_dir, 'config.json'))

    processor = Wav2Vec2FeatureExtractor.from_pretrained(
        config['model_name'], trust_remote_code=True,
    )

    checkpoint = torch.load(os.path.join(args.ckpt_dir, CKPT_FILE), weights_only=True)
    device = torch.device(f'cuda:0'if torch.cuda.is_available() else 'cpu')
    model = MERTClassifier(
        model_name=config['model_name'],
        hidden_states=config['hidden_states'],
        fine_tune_method=config['fine_tune_method'],
        thresholds=args.thresholds,
    )
    model.load_state_dict(checkpoint['model'])
    model = model.to(device)
    model.eval()

    for audio_path in audio_path_list:
        src_path = audio_path.replace('.flac', '.mid')
        filename = os.path.splitext(os.path.basename(audio_path))[0]
        true_pianoroll = extract_pianoroll_from_midi(src_path)
        print(f'source path: {src_path}')
        print(f'audio path: {audio_path}')

        # pred_pianoroll is your model predict result please load your results here
        # pred_pianoroll.shape should be [9, L] and the L should be equal to true_pianoroll
        length = true_pianoroll.shape[1]
        pred_pianoroll = predict_pianoroll(audio_path, length, processor, model, device)

        print(f'true pianoroll shape: {true_pianoroll.shape}')
        print(f'pred pianoroll shape: {pred_pianoroll.shape}')

        pianoroll_comparison(true_pianoroll, pred_pianoroll, os.path.join(RESULT_DIR, f'{filename}.png'))


if __name__ == '__main__':
    args = parse_arguments()
    main(args)

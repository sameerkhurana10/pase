import os
import argparse
import json
from collections import OrderedDict
import io
import kaldi_io as kio
from random import shuffle
import soundfile as sf


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--wsj-base-dir", help="Path to the wav_as_feats.scp Kaldi file")
    parser.add_argument("--train-split", default="train_si284")
    parser.add_argument("--dev-split", default="test_dev93")
    parser.add_argument("--test-split", default="test_eval92")
    parser.add_argument("--cfg-file", help="")
    return parser


def get_data_cfg():
    data_cfg = {'train': {'data': [],
                          'speakers': []},
                'valid': {'data': [],
                          'speakers': []},
                'test': {'data': [],
                         'speakers': []},
                'speakers': []}
    return data_cfg


def read_scp_file(text_file):
    utterances = []
    with open(text_file, "r") as text_f:
        for line in text_f:
            line = line.strip()
            if not line:
                continue
            utt_id, text = line.split(None, 1)
            utterances.append((utt_id, text))
    return utterances


def read_wav(file_or_fd, wav_sr=16000):
    """ [wav] = read_mat(file_or_fd)
     Reads a single wavefile, supports binary only
     file_or_fd : file, gzipped file, pipe or opened file descriptor.
    """
    fd = kio.open_or_fd(file_or_fd)
    try:
        wav, sr = sf.read(io.BytesIO(fd.read()))
        assert sr == wav_sr
        wav = wav.astype('float32')
    finally:
        if fd is not file_or_fd:
            fd.close()
    return wav, sr


def get_file_dur(fname):
    wav, sr = read_wav(fname)
    return len(wav)


if __name__ == '__main__':
    parser = get_parser()
    args = parser.parse_args()

    train_dir = args.wsj_base_dir + '/' + args.train_split
    dev_dir = args.wsj_base_dir + '/' + args.dev_split
    test_dir = args.wsj_base_dir + '/' + args.test_split
    # get all the speakers
    utt2spk_tr = dict(read_scp_file(train_dir + '/utt2spk'))
    utt2spk_val = dict(read_scp_file(dev_dir + '/utt2spk'))
    utt2spk_test = dict(read_scp_file(test_dir + '/utt2spk'))
    speakers = [s for u, s in utt2spk_tr.items()] + [s for u, s in utt2spk_val.items()] + \
               [s for u, s in utt2spk_test.items()]
    speakers = list(set(speakers))
    speakers_to_idx = {s: i for i, s in enumerate(speakers)}

    data_cfg = get_data_cfg()

    with open(train_dir + '/wav.scp', 'r') as f:
        train_files = [l.rstrip() for l in f]
        shuffle(train_files)
        train_dur = 0
        for ti, train_file in enumerate(train_files, start=1):
            print('Processing train file {:7d}/{:7d}'.format(ti,
                                                             len(train_files)),
                  end='\r')
            splits = train_file.split()
            spk_str = utt2spk_tr[splits[0]]
            spk = speakers_to_idx[spk_str]
            if spk not in data_cfg['speakers']:
                data_cfg['speakers'].append(spk)
                data_cfg['train']['speakers'].append(spk)

            data_cfg['train']['data'].append({'filename': ' '.join(splits[1:]),
                                              'spk': spk})
            train_dur += get_file_dur(' '.join(splits[1:]))

        data_cfg['train']['total_wav_dur'] = train_dur
        print()

    with open(dev_dir + '/wav.scp', 'r') as f:
        valid_files = [l.rstrip() for l in f]
        valid_dur = 0
        for ti, valid_file in enumerate(valid_files, start=1):
            print('Processing valid file {:7d}/{:7d}'.format(ti,
                                                             len(valid_files)),
                  end='\r')
            splits = valid_file.split()
            spk_str = utt2spk_val[splits[0]]
            spk = speakers_to_idx[spk_str]
            if spk not in data_cfg['speakers']:
                data_cfg['speakers'].append(spk)
                data_cfg['valid']['speakers'].append(spk)

            data_cfg['valid']['data'].append({'filename': ' '.join(splits[1:]),
                                              'spk': spk})
            valid_dur += get_file_dur(' '.join(splits[1:]))

        data_cfg['valid']['total_wav_dur'] = valid_dur
        print()

    with open(test_dir + '/wav.scp', 'r') as f:
        test_files = [l.rstrip() for l in f]
        test_dur = 0
        for ti, test_file in enumerate(test_files, start=1):
            print('Processing test file {:7d}/{:7d}'.format(ti,
                                                             len(test_files)),
                  end='\r')
            splits = test_file.split()
            spk_str = utt2spk_test[splits[0]]
            spk = speakers_to_idx[spk_str]
            if spk not in data_cfg['speakers']:
                data_cfg['speakers'].append(spk)
                data_cfg['test']['speakers'].append(spk)

            data_cfg['test']['data'].append({'filename': ' '.join(splits[1:]),
                                             'spk': spk})
            test_dur += get_file_dur(' '.join(splits[1:]))

        data_cfg['test']['total_wav_dur'] = test_dur
        print()

    with open(args.cfg_file, 'w') as cfg_f:
        cfg_f.write(json.dumps(data_cfg))





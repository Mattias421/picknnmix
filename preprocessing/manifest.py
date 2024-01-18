# create a tsv for each stem (vocals, drums, bass, other, mixture)
"""
<root-dir>
<audio-path-1>
<audio-path-2>
...
"""

import argparse
import os
import soundfile

parser = argparse.ArgumentParser()
parser.add_argument('--split', type=str, default='test')
parser.add_argument('--root', type=str)
parser.add_argument('--root_manifest', type=str)
args = parser.parse_args()

SPLIT = args.split
ROOT = args.root
ROOT_MANIFEST = args.root_manifest

os.makedirs(os.path.dirname(ROOT_MANIFEST), exist_ok=True)
vocals_f = open(os.path.join(ROOT_MANIFEST, f'vocals.tsv'), 'w')
drums_f = open(os.path.join(ROOT_MANIFEST, f'drums.tsv'), 'w')
bass_f = open(os.path.join(ROOT_MANIFEST, f'bass.tsv'), 'w')
other_f = open(os.path.join(ROOT_MANIFEST, f'other.tsv'), 'w')
mixture_f = open(os.path.join(ROOT_MANIFEST, f'mixture.tsv'), 'w')

stem_dict = {'vocals': vocals_f, 'drums': drums_f, 'bass': bass_f, 'other': other_f, 'mixture': mixture_f}

for f in stem_dict.values():
    print(ROOT, file=f)

for root, dirs, files in os.walk(ROOT):
    for dir in dirs:
        for stem in stem_dict.keys():
            fname = f'/{dir}/{stem}.wav'
            frames = soundfile.info(os.path.join(root, dir, f'{stem}.wav')).frames
            print(f'/{dir}/{stem}.wav\t{frames}', file=stem_dict[stem])
            
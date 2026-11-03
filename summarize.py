"""Produce manuscript Table 1 and Table 2 BCFSN results from seeds 1, 2, and 3.

Table 1: per dataset HM(mean_seed(Base), mean_seed(Novel)); overall HM is
HM(mean_dataset(mean_seed(Base)), mean_dataset(mean_seed(Novel))).
Table 2: per dataset mean_seed(accuracy), then mean_dataset(accuracy).
Missing runs never silently contribute to a purported three-seed average.
"""
import argparse
import csv
import json
from pathlib import Path
from statistics import mean

from description import DEFAULT_DESCRIPTION_SET, DESCRIPTION_SETS

DATASETS = (
    'imagenet', 'sun397', 'fgvc', 'eurosat', 'stanford_cars', 'food101',
    'oxford_pets', 'oxford_flowers', 'caltech101', 'dtd', 'ucf101',
)
DATASET_LABELS = {
    'imagenet': 'ImageNet', 'sun397': 'SUN397', 'fgvc': 'FGVC Aircraft',
    'eurosat': 'EuroSAT', 'stanford_cars': 'Stanford Cars', 'food101': 'Food101',
    'oxford_pets': 'Oxford Pets', 'oxford_flowers': 'Oxford Flowers',
    'caltech101': 'Caltech101', 'dtd': 'DTD', 'ucf101': 'UCF101',
}
SHOTS = (0, 1, 4, 16)


def hm(base, novel):
    return 2.0 * base * novel / (base + novel) if base + novel > 0 else 0.0


def _load_fewshot(root, group, setting, dataset, shots, seed, exp_name):
    path = (root / group / setting / 'ViT-B-16' / dataset / f'shots_{shots}' /
            f'seed_{seed}' / 'bcfsn' / f'{exp_name}.csv')
    if not path.is_file():
        return None
    with path.open(encoding='utf-8-sig', newline='') as stream:
        rows = list(csv.DictReader(stream))
    if len(rows) != 1:
        raise ValueError(f'Expected exactly one experiment result in {path}')
    row = rows[0]
    for key, value in (('dataset', dataset), ('setting', setting),
                       ('shots', str(shots)), ('seed', str(seed)),
                       ('description_set', group), ('method', 'bcfsn')):
        if row.get(key) != value:
            raise ValueError(f'Metadata mismatch in {path}: {key}={row.get(key)!r}, expected {value!r}')
    required = ('acc_test_base', 'acc_test_new') if setting == 'base2new' else ('acc_test',)
    return tuple(float(row[key]) for key in required)


def _load_zero(root, group, dataset, seed, exp_name):
    path = (root / group / 'standard' / 'ViT-B-16' / dataset / 'shots_0' /
            f'seed_{seed}' / 'bcfsn_zero_shot' / f'{exp_name}.json')
    if not path.is_file():
        return None
    row = json.loads(path.read_text(encoding='utf-8'))
    for key, value in (('dataset', dataset), ('setting', 'standard'),
                       ('shots', 0), ('seed', seed),
                       ('description_set', group), ('method', 'bcfsn_zero_shot'),
                       ('output_format_version', 2)):
        if row.get(key) != value:
            raise ValueError(f'Metadata mismatch in {path}: {key}={row.get(key)!r}, expected {value!r}')
    if 'accuracy_by_branch' in row:
        raise ValueError(f'Unsupported branch-level results in {path}. Provide final classification results.')
    return (float(row['acc_test']),)


def _write(path, rows, columns):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    print(f'Saved {path}', flush=True)


def table1(root, outdir, group, seeds, exp_name, strict=False):
    rows, complete_means, missing = [], [], []
    for dataset in DATASETS:
        samples = [_load_fewshot(root, group, 'base2new', dataset, 16, seed, exp_name)
                   for seed in seeds]
        if any(item is None for item in samples):
            missing.extend(f'base2new/{dataset}/16-shot/seed_{seed}'
                           for seed, item in zip(seeds, samples) if item is None)
            rows.append({'Dataset': DATASET_LABELS[dataset], 'Base': '', 'Novel': '', 'HM': ''})
            continue
        base, novel = (mean(sample[i] for sample in samples) for i in (0, 1))
        complete_means.append((base, novel))
        rows.append({'Dataset': DATASET_LABELS[dataset], 'Base': f'{base:.2f}',
                     'Novel': f'{novel:.2f}', 'HM': f'{hm(base, novel):.2f}'})
    if len(complete_means) == len(DATASETS):
        base, novel = mean(x[0] for x in complete_means), mean(x[1] for x in complete_means)
        rows.append({'Dataset': 'Average', 'Base': f'{base:.2f}',
                     'Novel': f'{novel:.2f}', 'HM': f'{hm(base, novel):.2f}'})
    if strict and missing:
        raise ValueError('Table 1 requires every requested seed of all 11 datasets:\n' + '\n'.join(missing))
    _write(outdir / 'Table1_base_to_novel.csv', rows, ('Dataset', 'Base', 'Novel', 'HM'))
    return missing


def table2(root, outdir, group, seeds, exp_name, strict=False):
    rows, missing = [], []
    for dataset in DATASETS:
        row = {'Dataset': DATASET_LABELS[dataset]}
        for shots in SHOTS:
            samples = [(_load_zero(root, group, dataset, seed, exp_name) if shots == 0
                        else _load_fewshot(root, group, 'standard', dataset, shots, seed, exp_name))
                       for seed in seeds]
            if any(sample is None for sample in samples):
                missing.extend(f'standard/{dataset}/{shots}-shot/seed_{seed}'
                               for seed, sample in zip(seeds, samples) if sample is None)
                row[f'{shots}-shot'] = ''
            else:
                row[f'{shots}-shot'] = f'{mean(sample[0] for sample in samples):.2f}'
        rows.append(row)
    if not missing:
        aggregate = {'Dataset': 'Average'}
        for shots in SHOTS:
            # Average UNROUNDED per-dataset results, not displayed 2-decimal cells.
            values = []
            for dataset in DATASETS:
                samples = [(_load_zero(root, group, dataset, seed, exp_name) if shots == 0
                            else _load_fewshot(root, group, 'standard', dataset, shots, seed, exp_name))
                           for seed in seeds]
                values.append(mean(sample[0] for sample in samples))
            aggregate[f'{shots}-shot'] = f'{mean(values):.2f}'
        rows.append(aggregate)
    if strict and missing:
        raise ValueError('Table 2 requires every requested seed of all 11 datasets:\n' + '\n'.join(missing))
    _write(outdir / 'Table2_all_to_all.csv', rows,
           ('Dataset', '0-shot', '1-shot', '4-shot', '16-shot'))
    return missing


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', default='results', help='Root of prediction results, not image data')
    parser.add_argument('--outdir', default='summaries')
    parser.add_argument('--description-set', default=DEFAULT_DESCRIPTION_SET, choices=DESCRIPTION_SETS)
    parser.add_argument('--exp-name', default='bcfsn')
    parser.add_argument('--seeds', nargs='+', type=int, default=[1, 2, 3])
    parser.add_argument('--table', choices=['1', '2', 'both'], default='both')
    parser.add_argument('--strict', action='store_true',
                        help='Require all 11 datasets with every requested seed before writing')
    args = parser.parse_args()
    if not args.seeds or len(args.seeds) != len(set(args.seeds)):
        parser.error('--seeds must contain distinct integers')
    root, outdir = Path(args.root), Path(args.outdir)
    for name, action in (('1', table1), ('2', table2)):
        if args.table in (name, 'both'):
            missing = action(root, outdir, args.description_set, args.seeds, args.exp_name, args.strict)
            if missing:
                print(f'Table {name}: {len(missing)} missing runs; Average row omitted. '
                      f'Example: {missing[0]}', flush=True)


if __name__ == '__main__':
    main()

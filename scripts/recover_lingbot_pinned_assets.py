"""Recover public pinned assets; verify archived bytes without claiming model QA."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path

from huggingface_hub import HfApi, hf_hub_download


def digest(path):
    result = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            result.update(block)
    return result.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive', type=Path, required=True)
    parser.add_argument('--destination', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    lingbot = json.loads((args.archive / 'assets/lingbot_vla2/acquisition.json').read_text())
    robotwin = json.loads((args.archive / 'assets/robotwin/acquisition.json').read_text())
    jobs = []
    for entry in lingbot['files']:
        jobs.append((lingbot['repository'], 'model', lingbot['revision'],
                     'lingbot_vla2', entry['path'], entry['sha256'], entry['bytes']))
    for entry in robotwin['files']:
        jobs.append((robotwin['repo'], 'dataset', robotwin['revision'],
                     'robotwin', entry['file'], entry['sha256'], entry['bytes']))
    qwen_repo = 'Qwen/Qwen3-VL-4B-Instruct'
    qwen_revision = 'ebb281ec70b05090aa6165b016eac8ec08e71b17'
    # Metadata only: the released LingBot checkpoint contains all learned weights.
    names = HfApi().list_repo_files(qwen_repo, revision=qwen_revision)
    metadata = [name for name in names if '/' not in name and
                (name.endswith(('.json', '.jinja')) or name == 'merges.txt')]
    if not {'config.json', 'tokenizer.json', 'tokenizer_config.json'}.issubset(metadata):
        raise ValueError('Pinned Qwen metadata is incomplete')
    for name in metadata:
        jobs.append((qwen_repo, 'model', qwen_revision, 'lingbot_qwen_metadata', name, None, None))

    def recover(job):
        repo, kind, revision, category, name, expected, size = job
        relative = Path(name)
        if relative.is_absolute() or '..' in relative.parts:
            raise ValueError('Asset path escapes destination')
        path = Path(hf_hub_download(repo_id=repo, repo_type=kind, revision=revision,
                                   filename=name, local_dir=args.destination / category))
        actual = digest(path)
        if expected is not None and (actual != expected or path.stat().st_size != size):
            raise ValueError('Recovered asset differs from archived bytes: ' + name)
        print('VERIFIED ' + category + '/' + name, flush=True)
        return dict(repository=repo, repo_type=kind, revision=revision, path=str(path),
                    bytes=path.stat().st_size, sha256=actual,
                    archived_digest_verified=expected is not None)

    with ThreadPoolExecutor(max_workers=3) as pool:
        files = list(pool.map(recover, jobs))
    report = dict(status='pinned_public_assets_recovered', files=files,
                  scope='Acquisition and byte integrity only; runtime, reset, numerical and paired QA remain required.',
                  inference_qualified=False, closed_loop_qualified=False,
                  runner_sha256=digest(Path(__file__)))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + '\n')


if __name__ == '__main__':
    main()

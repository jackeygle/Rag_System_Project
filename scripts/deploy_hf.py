"""Publish an explicit source allowlist to the existing Hugging Face Space.

Use `hf auth login` locally first. --dry-run does not authenticate or make network calls.
"""
import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SPACE = 'jackeygleee/jackeygleeeeee'


def deployment_files():
    paths = ['Dockerfile', 'requirements.txt', 'requirements_streamlit.txt', 'config.py', 'streamlit_app.py']
    paths += [str(path.relative_to(ROOT)) for folder, pattern in [('src', '*.py'), ('assets', '*.css'), ('.streamlit', '*.toml')]
              for path in sorted((ROOT / folder).glob(pattern))]
    # Only tracked sample docs; no user uploads, indexes, cache, tests, or secrets.
    import subprocess
    tracked = subprocess.check_output(['git', '-C', str(ROOT), 'ls-files', 'data/documents'], text=True).splitlines()
    paths += [path for path in tracked if Path(path).suffix in {'.md', '.txt', '.pdf'}]
    files = [(path, ROOT / path) for path in paths]
    files.append(('README.md', ROOT / 'README_HF.md'))
    for destination, source in files:
        if not source.is_file():
            raise FileNotFoundError(source)
    return files


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--space', default=SPACE)
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    files = deployment_files()
    if args.dry_run:
        print('Destination:', args.space)
        print('\n'.join(destination for destination, _ in files))
        return
    from huggingface_hub import HfApi, CommitOperationAdd, get_token
    token = get_token()
    if not token:
        raise SystemExit('Hugging Face write access is missing. Run hf auth login locally first.')
    api = HfApi(token=token)
    info = api.repo_info(args.space, repo_type='space')
    result = api.create_commit(repo_id=args.space, repo_type='space', parent_commit=info.sha,
        operations=[CommitOperationAdd(path_in_repo=destination, path_or_fileobj=str(source)) for destination, source in files],
        commit_message='Deploy Streamlit document workspace with source citations and incremental indexing')
    print('Uploaded:', result.commit_url)
    print('The Space build must finish before the new UI is live.')


if __name__ == '__main__':
    main()

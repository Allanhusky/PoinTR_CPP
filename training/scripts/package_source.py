"""Package current Git-visible sources without .git history or ignored artifacts."""
import argparse
from pathlib import Path
import subprocess
import tarfile
ROOT = Path(__file__).resolve().parents[2]

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT/'artifacts/PoinTr_source.tar.gz')
    args = parser.parse_args()
    output = args.output.resolve()
    paths = subprocess.check_output(['git', 'ls-files', '-z', '--cached', '--others', '--exclude-standard'], cwd=ROOT).decode().split('\0')
    paths = sorted({name for name in paths if name and (ROOT/name).is_file() and (ROOT/name).resolve() != output})
    # Tracked files are listed even if newly ignored; exclude them as well.
    ignored = subprocess.run(['git', 'check-ignore', '--no-index', '-z', '--stdin'], input=('\0'.join(paths)+'\0').encode(), cwd=ROOT, capture_output=True)
    if ignored.returncode not in (0, 1):
        raise RuntimeError(ignored.stderr.decode())
    excluded = set(ignored.stdout.decode().split('\0'))
    paths = [name for name in paths if name not in excluded]
    oversized = [name for name in paths if (ROOT/name).stat().st_size > 50*1024*1024]
    if oversized:
        raise RuntimeError('源码包包含超过 50 MiB 的文件，请先配置忽略规则：'+', '.join(oversized))
    output.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(output, 'w:gz') as archive:
        for name in paths:
            archive.add(ROOT/name, arcname='PoinTr/'+name, recursive=False)
    print(f'{output}: {len(paths)} files, {output.stat().st_size/1024**2:.2f} MiB')

if __name__ == '__main__':
    main()

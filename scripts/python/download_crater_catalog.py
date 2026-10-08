"""Download the public USGS Robbins lunar crater catalog for the labeler.

Run from the repo root: python scripts/python/download_crater_catalog.py
"""
import argparse
from pathlib import Path
import shutil
import tempfile
import urllib.request
import zipfile

URL = ('https://astrogeology.usgs.gov/ckan/dataset/f89f5478-b69a-486c-b9b5-30d7b0c5ad2b/'
       'resource/c4f25cc2-4f8a-4207-a845-5e176da3ac5a/download/lunar_crater_database_robbins_2018')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, default=Path('data/catalogs'))
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    names = ['lunar_crater_database_robbins_2018.csv', 'lunar_crater_database_robbins_2018.xml']
    if any((args.output_dir/name).exists() for name in names):
        parser.error('Catalog files already exist. Choose a different output directory.')
    print('Downloading the USGS bundle (~92 MB; extracted CSV ~238 MB)...')
    with tempfile.TemporaryDirectory() as tmp:
        bundle = Path(tmp)/'catalog.zip'
        urllib.request.urlretrieve(URL, bundle)
        with zipfile.ZipFile(bundle) as archive:
            for name in names:
                member='lunar_crater_database_robbins_2018_bundle/data/'+name
                with archive.open(member) as src, (args.output_dir/name).open('xb') as dst:
                    shutil.copyfileobj(src,dst)
                print(args.output_dir/name)


if __name__ == '__main__':
    main()

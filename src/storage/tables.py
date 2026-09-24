import os
import pickle
import urllib.request
import urllib.parse
from io import StringIO, BytesIO
from datetime import datetime

import pandas as pd
from ..utils.config import Config as cfg


class BlobTable:
    def __init__(self, table_name="", ftype="csv"):
        self.table_name = table_name
        self.ftype = ftype
        self.dfs = {}

    @property
    def df(self):
        return pd.concat(self.dfs.values())

    def get_path(self, partition):
        path = "/".join([el for el in [self.table_name, partition] if el])
        return f"{path}.{self.ftype}"

    def download(self, partitions=[], concat=False, **kwargs):
        for p in partitions:
            try:
                self.dfs[p] = self.read(p, **kwargs)
                print(datetime.now(), f"read: {self.get_path(p)}", self)
            except FileNotFoundError:
                print(f"Warning: File not found for partition {p}, skipping.")
            except Exception as e:
                print(f"Error reading {p}: {e}")
        return self.dfs if not concat else self.df

    def upload(self, dfs):
        for p, df in dfs.items():
            self.write(p, df)
            print(datetime.now(), f"write: {self.get_path(p)}", self)
        return self


class AzureBlobTable(BlobTable):
    def __init__(self, table_name="", ftype="csv"):
        super().__init__(table_name, ftype)
        self.base_dir = os.path.join(os.getcwd(), "local_data")
        if not os.path.exists(self.base_dir):
            os.makedirs(self.base_dir)

    def _get_local_path(self, partition):
        full_path = os.path.join(self.base_dir, self.table_name, f"{partition}.{self.ftype}")
        directory = os.path.dirname(full_path)
        if not os.path.exists(directory):
            os.makedirs(directory)
        return full_path

    @property
    def read(self):
        return {"csv": self.read_csv, "parquet": self.read_parquet, "pkl": self.read_pkl}[self.ftype]

    @property
    def write(self):
        return {"csv": self.write_csv, "parquet": self.write_parquet, "pkl": self.write_pkl}[self.ftype]

    def write_csv(self, partition, data):
        data.to_csv(self._get_local_path(partition), index=False)

    def read_csv(self, partition):
        return pd.read_csv(self._get_local_path(partition))

    def write_parquet(self, partition, data):
        data.to_parquet(self._get_local_path(partition), index=False)

    def read_parquet(self, partition):
        return pd.read_parquet(self._get_local_path(partition))

    def write_pkl(self, partition, data):
        with open(self._get_local_path(partition), 'wb') as f:
            pickle.dump(data, f)

    def read_pkl(self, partition):
        with open(self._get_local_path(partition), 'rb') as f:
            return pickle.load(f)


class ExternalBlobTable(BlobTable):
    def __init__(self, table_name=""):
        super().__init__(table_name, "csv")
        self.url = cfg.FOOTBALL_DATA_URL

    def get_link(self, partition):
        return f"{self.url}/{self.get_path(partition)}"

    def read(self, partition):
        link = self.get_link(partition)
        print(f"Downloading: {link}")
        req = urllib.request.Request(link, headers={'User-Agent': 'Mozilla/5.0'})
        try:
            with urllib.request.urlopen(req) as response:
                return pd.read_csv(StringIO(response.read().decode("utf-8")))
        except Exception as e:
            print(f"Skipping {link}: {e}")
            return pd.DataFrame()
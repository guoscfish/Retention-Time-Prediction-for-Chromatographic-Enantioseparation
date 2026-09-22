"""Retrieve only ODH members of the author's README-linked 600 MB archive."""

import argparse
import hashlib
from html.parser import HTMLParser
import io
import json
from pathlib import Path
import urllib.parse
import urllib.request
import zipfile

ROOT = Path(__file__).resolve().parents[1]
URL = "https://drive.usercontent.google.com/download?id=1kzM2_pG-Ob_hNQ7ga2n6Ds2FXDeAppWx&export=download"


class DownloadForm(HTMLParser):
    def __init__(self):
        super().__init__()
        self.fields = {}

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if tag == "input" and attrs.get("type") == "hidden":
            self.fields[attrs["name"]] = attrs["value"]


class RemoteZip(io.RawIOBase):
    def __init__(self, url):
        self.url = url
        self.position = 0
        with urllib.request.urlopen(urllib.request.Request(url, headers={"Range": "bytes=0-0"}), timeout=90) as r:
            if r.status != 206:
                raise RuntimeError("Server does not support byte ranges")
            self.size = int(r.headers["Content-Range"].rsplit("/", 1)[1])

    def seekable(self):
        return True

    def seek(self, offset, whence=0):
        self.position = offset + (self.position if whence == 1 else self.size if whence == 2 else 0)
        return self.position

    def tell(self):
        return self.position

    def read(self, size=-1):
        if size < 0:
            size = self.size - self.position
        end = min(self.size, self.position + size) - 1
        if end < self.position:
            return b""
        requested = f"bytes={self.position}-{end}"
        with urllib.request.urlopen(urllib.request.Request(self.url, headers={"Range": requested}), timeout=180) as r:
            if r.status != 206 or not r.headers["Content-Range"].startswith(f"bytes {self.position}-{end}/"):
                raise RuntimeError("Unexpected byte range response")
            data = r.read()
        if len(data) != end - self.position + 1:
            raise RuntimeError("Truncated archive response")
        self.position += len(data)
        return data


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--list-only", action="store_true")
    args = parser.parse_args()
    with urllib.request.urlopen(URL, timeout=90) as response:
        form = DownloadForm()
        form.feed(response.read().decode())
    url = URL.split("?", 1)[0] + "?" + urllib.parse.urlencode(form.fields)
    remote = RemoteZip(url)
    with zipfile.ZipFile(remote) as archive:
        names = [n for n in archive.namelist() if "ODH" in n or n.endswith(("Single_column_prediction.py", "data_process.py"))]
        print("\n".join(names), flush=True)
        if args.list_only:
            return
        selected = [n for n in names if Path(n).name in {
            "dataset_ODH.npy", "dataset_ODH_morder.npy", "ODH_charity_0616.csv",
            "Single_column_prediction.py", "data_process.py",
        } or ("model_ODH_388/" in n and n.endswith("model_save_1500.pth"))]
        out = ROOT / "artifacts/reproduction/odh_cache"
        out.mkdir(parents=True, exist_ok=True)
        metadata = {"source": URL, "archive_bytes": remote.size, "members": []}
        for name in selected:
            data = archive.read(name)  # zipfile verifies the published member CRC.
            target = out / Path(name).name
            if target.exists() and target.read_bytes() != data:
                raise RuntimeError(f"Refusing to replace different cached member: {target}")
            target.write_bytes(data)
            info = {"member": name, "file": target.name, "bytes": len(data),
                    "sha256": hashlib.sha256(data).hexdigest(), "crc32": archive.getinfo(name).CRC}
            metadata["members"].append(info)
            print(info, flush=True)
        (out / "provenance.json").write_text(json.dumps(metadata, indent=2) + "\n")
        audit = ROOT / "artifacts/reproduction/audit"
        audit.mkdir(parents=True, exist_ok=True)
        (audit / "official_cache_provenance.json").write_text(json.dumps(metadata, indent=2) + "\n")


if __name__ == "__main__":
    main()

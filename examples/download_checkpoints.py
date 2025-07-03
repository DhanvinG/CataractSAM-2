import pathlib, requests

URL  = "https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_large.pt"
dest = pathlib.Path(__file__).resolve().parents[1] / "checkpoints" / "sam2_hiera_large.pt"
dest.parent.mkdir(parents=True, exist_ok=True)

print("⇣", URL)
with requests.get(URL, stream=True) as r, open(dest, "wb") as f:
    for chunk in r.iter_content(8192):
        f.write(chunk)
print("✔ downloaded →", dest)

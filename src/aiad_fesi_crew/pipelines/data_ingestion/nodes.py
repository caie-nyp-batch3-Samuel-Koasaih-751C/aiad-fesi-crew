from pathlib import Path
import gdown
from zipfile import ZipFile

def download_from_gdrive(file_id: str, output_dir: str) -> str:
    """
    Downloads and extracts a zip file from a public Google Drive file ID.
    """
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    zip_path = out_path / "dataset.zip"
    url = f"https://drive.google.com/uc?id={file_id}"
    gdown.download(url, str(zip_path), quiet=False)

    with ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(out_path)

    zip_path.unlink()  # remove the zip after extraction
    return str(out_path)

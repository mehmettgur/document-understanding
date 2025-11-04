#!/usr/bin/env python3
"""
Tek dosyalı İK RAG yardımcı aracı.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import warnings
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

import requests

# Temel bağımlılık kontrolleri (zorunlu olanlar burada tutuluyor)
REQUIRED_IMPORTS = {
    "numpy": "pip install numpy",
    "pandas": "pip install pandas",
    "slugify": "pip install python-slugify",
    "tqdm": "pip install tqdm",
    "sentence_transformers": "pip install sentence-transformers",
    "qdrant_client": "pip install qdrant-client",
    "pypdfium2": "pip install pypdfium2",
}

warnings.filterwarnings(
    "ignore",
    message=r"Qdrant client version .* is incompatible with server version .*",
    category=UserWarning,
)

for module_name, install_hint in REQUIRED_IMPORTS.items():
    try:
        globals()[module_name] = __import__(module_name)
    except ImportError as exc:  # pragma: no cover - başlangıç hatası
        raise SystemExit(
            f"{module_name} modülü bulunamadı. Lütfen şu komutla yükleyin: {install_hint}"
        ) from exc

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from slugify import slugify  # noqa: E402
from tqdm import tqdm  # noqa: E402

from qdrant_client import QdrantClient  # noqa: E402
from qdrant_client.models import Distance, PointStruct, VectorParams  # noqa: E402

try:
    from huggingface_hub.constants import HF_HOME
except ImportError:  # pragma: no cover - hatalı kurulum durumunda varsayılan
    HF_HOME = os.environ.get(
        "HF_HOME", str(Path.home() / ".cache" / "huggingface")
    )

_EMBED_MODEL_CACHE: Dict[str, "sentence_transformers.SentenceTransformer"] = {}


def _parse_env_bool(value: Optional[str], default: bool = False) -> bool:
    """Basit bir bool env değişkeni ayrıştırıcısı."""
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "evet", "on", "yes"}


def _clean_env_str(value: Optional[str]) -> Optional[str]:
    """Boş env değerlerini None'a çevirir."""
    if value is None:
        return None
    stripped = value.strip()
    return stripped or None


@dataclass
class RagServiceConfig:
    """Web arayüzü ve API için yeniden kullanılabilir yapılandırma."""

    collection: str = "hr_rag"
    model_dir: Optional[str] = None
    qdrant_url: str = "http://localhost:6333"
    qdrant_api_key: Optional[str] = None
    threads: int = 4
    docling_batch: int = 2
    lang: str = "tr"
    preprocess: bool = False
    preprocess_profile: str = "bw"  # auto | color | bw (default: bw)
    target_chars_min: int = 1200
    target_chars_max: int = 1600
    overlap_chars: int = 220
    embedding_batch_size: int = 256
    top_k: int = 3
    persist_outputs: bool = False
    outputs_root: Path = field(default_factory=lambda: Path("outputs").resolve())
    use_ollama: bool = True
    ollama_url: str = "http://localhost:11434"
    ollama_model: str = "aya-expanse:32b"
    ollama_temperature: float = 0.1
    ollama_max_context_chunks: int = 3
    chat_use_collection: bool = False

    @classmethod
    def from_env(cls) -> "RagServiceConfig":
        """Varsayılanları ortam değişkenlerinden okur."""
        model_env = _clean_env_str(
            os.environ.get("TRENDYOL_MODEL_DIR") or os.environ.get("MODEL_DIR")
        )
        return cls(
            collection=os.environ.get("HR_RAG_COLLECTION")
            or os.environ.get("QDRANT_COLLECTION")
            or "hr_rag",
            model_dir=model_env,
            qdrant_url=os.environ.get("QDRANT_URL", "http://localhost:6333"), # 192.168.84.157
            qdrant_api_key=_clean_env_str(os.environ.get("QDRANT_API_KEY")),
            threads=int(os.environ.get("HR_RAG_THREADS", "4")),
            docling_batch=int(os.environ.get("HR_RAG_DOCLING_BATCH", "2")),
            lang=os.environ.get("HR_RAG_LANG", "tr"),
            preprocess=_parse_env_bool(os.environ.get("HR_RAG_PREPROCESS")),
            preprocess_profile=(os.environ.get("HR_RAG_PREPROCESS_PROFILE", "bw").strip().lower()),
            target_chars_min=int(os.environ.get("HR_RAG_CHARS_MIN", "1200")),
            target_chars_max=int(os.environ.get("HR_RAG_CHARS_MAX", "1600")),
            overlap_chars=int(os.environ.get("HR_RAG_OVERLAP", "220")),
            embedding_batch_size=int(os.environ.get("HR_RAG_EMBED_BATCH", "256")),
            top_k=int(os.environ.get("HR_RAG_TOP_K", "3")),
            persist_outputs=_parse_env_bool(os.environ.get("HR_RAG_PERSIST_OUTPUTS")),
            outputs_root=Path(
                os.environ.get("HR_RAG_OUTPUTS_ROOT", "outputs")
            ).resolve(),
            use_ollama=_parse_env_bool(os.environ.get("HR_RAG_USE_OLLAMA"), True),
            ollama_url=os.environ.get("HR_RAG_OLLAMA_URL", "http://localhost:11434"),
            ollama_model=os.environ.get("HR_RAG_OLLAMA_MODEL", "aya-expanse:32b"),
            ollama_temperature=float(os.environ.get("HR_RAG_OLLAMA_TEMPERATURE", "0.1")),
            ollama_max_context_chunks=int(os.environ.get("HR_RAG_OLLAMA_MAX_CONTEXT", "3")),
            chat_use_collection=_parse_env_bool(os.environ.get("HR_RAG_CHAT_USE_COLLECTION")),
        )

    def ensure_outputs_root(self) -> Path:
        """Çıktı klasörünü oluşturur ve döndürür."""
        if not self.outputs_root.exists():
            self.outputs_root.mkdir(parents=True, exist_ok=True)
        return self.outputs_root


def log_info(message: str) -> None:
    """Kısa bilgi mesajlarını yazdırır."""
    print(message)


def log_error(message: str) -> None:
    """Hata mesajlarını stderr üzerine aktarır."""
    print(message, file=sys.stderr)


def insert_spaces_from_camel_or_pascal(name: str) -> str:
    """Bitişik yazılmış büyük harfli isimlerde (Türkçe karakterler dahil) aralara boşluk ekler."""
    if not name:
        return ""
    pattern = re.compile(r"(?<=.)(?=[A-ZÇĞİÖŞÜ])")
    spaced = pattern.sub(" ", name)
    return spaced.strip()


def slugify_tr(text: str) -> str:
    """Türkçe'ye uygun slug üretir."""
    replacements = {
        "ç": "c",
        "ğ": "g",
        "ı": "i",
        "i": "i",
        "ö": "o",
        "ş": "s",
        "ü": "u",
        "Ç": "c",
        "Ğ": "g",
        "İ": "i",
        "I": "i",
        "Ö": "o",
        "Ş": "s",
        "Ü": "u",
    }
    normalized = text or ""
    for src, dest in replacements.items():
        normalized = normalized.replace(src, dest)
    return slugify(normalized, lowercase=True, separator="-")


_COLLECTION_COMPONENT_RE = re.compile(r"[^a-z0-9_-]")


def _normalize_collection_component(value: str) -> str:
    """Qdrant koleksiyon adında kullanılacak parçayı normalize eder."""
    if not value:
        return ""
    slug = slugify_tr(value)
    slug = slug.replace("-", "_")
    slug = _COLLECTION_COMPONENT_RE.sub("_", slug).strip("_")
    return slug


def _collection_name_for_document(
    base_name: Optional[str],
    doc_stem: str,
) -> str:
    """Belgeye özgü Qdrant koleksiyon adı üretir."""
    base_component = _normalize_collection_component(base_name or "")
    doc_component = _normalize_collection_component(doc_stem)
    if doc_component:
        name = doc_component
    elif base_component:
        name = base_component
    else:
        name = "hr_rag"
    if not name or not name[0].isalnum():
        name = f"c_{name or 'hr_rag'}"
    return name[:255]


def parse_filename_metadata(file_path: str) -> Dict[str, str]:
    """PDF dosya adından çalışan ve departman bilgilerini çıkarır."""
    stem = Path(file_path).stem
    if "-" not in stem:
        raise ValueError(
            "Dosya adı 'AdSoyad-Bölüm.pdf' biçiminde olmalıdır."
        )
    raw_name, raw_department = stem.split("-", 1)
    employee_name = insert_spaces_from_camel_or_pascal(raw_name)
    department = raw_department.replace("_", " ").strip()
    return {
        "employee_name": employee_name,
        "department": department,
        "doc_title": stem,
    }


def normalize_text_safe_markdown_tables(text: str) -> str:
    """
    Markdown tablo bloklarını (satırı '|' ile başlayan ardışık satırlar) koruyarak normalizasyon yapar.
    Tablo satırlarının içine asla regex ile newline eklemez ya da bölmez.
    """
    if not text:
        return ""

    # Tablo bloklarını yakala: '|' ile başlayan en az iki ardışık satır
    lines = text.replace("\r\n", "\n").replace("\r", "\n").split("\n")
    out = []
    i = 0
    n = len(lines)

    def is_table_line(s: str) -> bool:
        ss = s.lstrip()
        return ss.startswith("|") and ss.count("|") >= 2  # tek hücre değil, gerçek satır

    while i < n:
        if is_table_line(lines[i]):
            # tablo bloğu
            block = []
            while i < n and is_table_line(lines[i]):
                # sadece sağ-sol boşlukları kırp; satırın içindeki '|' yapısına dokunma
                block.append(lines[i].rstrip())
                i += 1
            out.extend(block)
        else:
            # tablo dışı metin: daha agresif sadeleştirilebilir
            chunk = lines[i]
            # sekme/çoklu boşluk → tek boşluk
            chunk = re.sub(r"[ \t]+", " ", chunk)
            out.append(chunk.rstrip())
            i += 1

    # 3+ boş satırı 2’ye indir
    joined = "\n".join(out)
    joined = re.sub(r"\n{3,}", "\n\n", joined)
    return joined.strip()



def robust_docling_call(
    input_pdf: Path,
    output_dir: Path,
    threads: int,
    batch: int,
    lang: str,
    timeout: int = 1800,
) -> Tuple[Path, Optional[Path], Optional[Path]]:
    """
    Docling CLI çağrısını güvenli şekilde yapar ve çıktı yollarını döndürür.
    Zorunlu rapidocr/tr parametrelerini burada sabitliyoruz.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = input_pdf.stem

    cmd = [
        "docling",
        "--pipeline",
        "standard",
        "--to",
        "md",
        "--to",
        "json",
        "--pdf-backend",
        "pypdfium2",
        "--num-threads",
        str(threads),
        "--page-batch-size",
        str(batch),
        "--force-ocr",
        "--ocr-engine",
        "rapidocr",
        "--ocr-lang",
        lang,
        "--image-export-mode",
        "placeholder",
        "--output",
        str(output_dir),
        str(input_pdf),
    ]

    log_info("Docling çağrısı başlatılıyor...")
    log_info(" ".join(cmd))
    try:
        completed = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=True,
            timeout=timeout,
            text=True,
        )
    except FileNotFoundError as exc:
        raise SystemExit(
            "Docling CLI bulunamadı. Lütfen `pip install docling[cli]` komutunu çalıştırın."
        ) from exc
    except subprocess.TimeoutExpired as exc:  # pragma: no cover - uzun süreli çalışma
        raise SystemExit("Docling işlemi zaman aşımına uğradı.") from exc
    except subprocess.CalledProcessError as exc:
        log_error("Docling hata çıktılarını iletti:")
        log_error(exc.stderr.strip())
        raise SystemExit("Docling çalıştırması başarısız oldu.") from exc

    if completed.stdout.strip():
        log_info("Docling çıktı:")
        log_info(completed.stdout.strip())
    if completed.stderr.strip():
        log_info("Docling uyarıları:")
        log_info(completed.stderr.strip())

    md_path = output_dir / f"{stem}.md"
    json_path = output_dir / f"{stem}.json"
    artifacts_dir = output_dir / f"{stem}_artifacts"

    if not md_path.exists():
        raise SystemExit(f"Markdown çıktısı bulunamadı: {md_path}")

    log_info(f"Markdown çıktı yolu: {md_path}")
    if json_path.exists():
        log_info(f"JSON çıktı yolu: {json_path}")
    else:
        json_path = None
        log_info("JSON çıktısı bulunamadı (Docling bazı durumlarda üretmeyebilir).")
    if artifacts_dir.exists():
        log_info(f"Artifakt klasörü: {artifacts_dir}")
    else:
        artifacts_dir = None
        log_info("Artifakt klasörü bulunamadı.")

    return md_path, json_path, artifacts_dir


def _load_docling_json(json_path: Path) -> Optional[List[str]]:
    """Docling JSON yapısından sayfa metinlerini ayıklar."""
    try:
        with json_path.open("r", encoding="utf-8") as fh:
            data = json.load(fh)
    except Exception as exc:  # pragma: no cover - bozuk dosya
        log_error(f"Docling JSON okunamadı: {exc}")
        return None

    pages: List[str] = []
    # Docling JSON formatları farklılık gösterebildiği için geniş kapsamlı kontroller yapıyoruz.
    if isinstance(data, dict):
        if isinstance(data.get("pages"), list):
            for page in data["pages"]:
                fragments: List[str] = []
                if isinstance(page, dict):
                    # Önce doğrudan metin alanlarını topluyoruz.
                    for key in ("text", "markdown", "md", "plainText"):
                        raw = page.get(key)
                        if isinstance(raw, str) and raw.strip():
                            fragments.append(raw)
                    # Blok/öğe listelerini tarıyoruz.
                    for block_key in ("blocks", "items", "elements", "content"):
                        block_value = page.get(block_key)
                        if isinstance(block_value, list):
                            for block in block_value:
                                if isinstance(block, dict):
                                    block_text = block.get("text") or block.get("markdown")
                                    if isinstance(block_text, str) and block_text.strip():
                                        fragments.append(block_text)
                pages.append("\n".join(fragments).strip())
        elif isinstance(data.get("document"), dict):
            # Bazı Docling şemalarında "document" içinde sayfalar yer alıyor.
            document = data["document"]
            if isinstance(document.get("pages"), list):
                for page in document["pages"]:
                    text = ""
                    if isinstance(page, dict):
                        text = page.get("text") or ""
                    pages.append(str(text).strip())
    return pages or None


def _get_pdf_page_count(pdf_path: Path) -> int:
    """PDF sayfa sayısını olabildiğince güvenilir şekilde tespit eder."""
    errors: List[str] = []

    try:
        pdf = pypdfium2.PdfDocument(str(pdf_path))
        try:
            count = len(pdf)
            if count > 0:
                return count
            errors.append("pypdfium2 sıfır sayfa döndürdü")
        finally:
            pdf.close()
    except Exception as exc:  # pragma: no cover - bozuk dosya olasılığı
        errors.append(f"pypdfium2 hatası: {exc}")

    try:
        import fitz  # type: ignore

        with fitz.open(str(pdf_path)) as doc:
            count = doc.page_count
            if count > 0:
                return count
            errors.append("PyMuPDF sıfır sayfa döndürdü")
    except ImportError:
        pass
    except Exception as exc:
        errors.append(f"PyMuPDF hatası: {exc}")

    try:
        from PyPDF2 import PdfReader  # type: ignore

        reader = PdfReader(str(pdf_path))
        count = len(reader.pages)
        if count > 0:
            return count
        errors.append("PyPDF2 sıfır sayfa döndürdü")
    except ImportError:
        pass
    except Exception as exc:
        errors.append(f"PyPDF2 hatası: {exc}")

    try:
        from pypdf import PdfReader as PyPdfReader  # type: ignore

        reader = PyPdfReader(str(pdf_path))
        count = len(reader.pages)
        if count > 0:
            return count
        errors.append("pypdf sıfır sayfa döndürdü")
    except ImportError:
        pass
    except Exception as exc:
        errors.append(f"pypdf hatası: {exc}")

    if errors:
        log_error("PDF sayfa sayısı tespitinde sorunlar oluştu: " + " | ".join(errors))
    return 1


def _guess_pdf_color_space(pdf_path: Path) -> str:
    """Basit bir sezgisel: PDF içeriğinde DeviceGray/DeviceRGB anahtarlarına bak.
    'bw' veya 'color' döndürür."""
    try:
        with pdf_path.open('rb') as fh:
            data = fh.read(2 * 1024 * 1024)  # ilk 2MB yeterli olur çoğu dosya için
        text = data.decode('latin-1', errors='ignore')
        gray_hits = len(re.findall(r"/DeviceGray|/CalGray", text))
        color_hits = len(re.findall(r"/DeviceRGB|/CalRGB|/ICCBased|/DeviceCMYK|/Separation|/Indexed", text))
        if gray_hits > 0 and color_hits == 0:
            return 'bw'
        return 'color'
    except Exception:
        return 'color'


def preprocess_pdf_with_imagemagick(input_pdf: Path, dpi: int = 400, profile: str = "auto") -> Tuple[Path, Path]:
    """
    ImageMagick kullanarak PDF'i 400 DPI gri PNG'lere dönüştürür, iyileştirme zincirini uygular
    ve sayfaları tekrar ZIP sıkıştırmalı PDF haline getirir.
    """
    if shutil.which("magick") is None:
        raise SystemExit(
            "ImageMagick 'magick' komutu bulunamadı. Lütfen ImageMagick kurulumunu tamamlayın."
        )

    temp_dir = Path(tempfile.mkdtemp(prefix="hr_rag_pre_"))
    png_dir = temp_dir / "png"
    png_dir.mkdir(parents=True, exist_ok=True)

    output_pattern = png_dir / f"{input_pdf.stem}_p%04d.png"
    active_profile = profile if profile in {"auto", "color", "bw"} else "auto"
    if active_profile == "auto":
        active_profile = _guess_pdf_color_space(input_pdf)
    if active_profile == "bw":
        # B/W için daha yumuşak kontrast, CLAHE ile lokal iyileştirme
        cmd_raster = [
            "magick",
            "-density",
            str(dpi),
            str(input_pdf),
            "-colorspace",
            "Gray",
            "-deskew",
            "40%",
            "-trim",
            "+repage",
            "-clahe",  # tilesXxY+clipLimit+bins
            "8x8+128+3",
            "-sigmoidal-contrast",
            "7x10%",
            "-sharpen",
            "0x1",
            "-background",
            "white",
            "-alpha",
            "remove",
            "-strip",
            str(output_pattern),
        ]
        log_info("Ön işleme profili: siyah-beyaz (CLAHE + sigmoidal-contrast)")
    else:
        # Renkli belgeler için önceki zinciri koruyoruz
        cmd_raster = [
            "magick",
            "-density",
            str(dpi),
            str(input_pdf),
            "-colorspace",
            "Gray",
            "-auto-level",
            "-contrast-stretch",
            "1%x1%",
            "-deskew",
            "40%",
            "-trim",
            "+repage",
            "-sharpen",
            "0x1",
            "-background",
            "white",
            "-alpha",
            "remove",
            "-strip",
            str(output_pattern),
        ]
        log_info("Ön işleme profili: renkli (auto-level + contrast-stretch)")
    log_info("ImageMagick rasterizasyonu başlatılıyor...")
    log_info(" ".join(cmd_raster))
    try:
        subprocess.run(cmd_raster, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    except subprocess.CalledProcessError as exc:
        stderr = exc.stderr.decode("utf-8", errors="ignore") if exc.stderr else ""
        log_error(f"ImageMagick rasterizasyonu başarısız oldu: {stderr}")
        raise SystemExit("ImageMagick rasterizasyon adımı başarısız.") from exc

    png_files = sorted(png_dir.glob("*.png"))
    if not png_files:
        raise SystemExit("Ön işleme sonrası PNG üretilemedi, lütfen PDF'i kontrol edin.")

    temp_pdf = temp_dir / f"{input_pdf.stem}_preprocessed.pdf"
    cmd_merge = ["magick"] + [str(p) for p in png_files] + [
        "-units",
        "PixelsPerInch",
        "-density",
        str(dpi),
        "-compress",
        "zip",
        str(temp_pdf),
    ]
    log_info("ImageMagick PDF birleştirme adımı başlatılıyor...")
    log_info(" ".join(cmd_merge[:5]) + " ...")
    try:
        subprocess.run(cmd_merge, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    except subprocess.CalledProcessError as exc:
        stderr = exc.stderr.decode("utf-8", errors="ignore") if exc.stderr else ""
        log_error(f"ImageMagick PDF birleştirmesi başarısız oldu: {stderr}")
        raise SystemExit("ImageMagick PDF birleştirme adımı başarısız.") from exc

    log_info(f"Ön işleme PNG klasörü: {png_dir}")
    log_info(f"Ön işlenmiş PDF yolu: {temp_pdf}")
    return temp_pdf, png_dir


def _docling_artifact_page_count(artifacts_dir: Optional[Path]) -> Optional[int]:
    """Docling artifact klasörlerindeki sayfa görsellerini sayar."""
    if not artifacts_dir or not artifacts_dir.exists():
        return None
    try:
        images_dir = artifacts_dir / "images"
        if images_dir.exists():
            count = sum(1 for path in images_dir.glob("*") if path.is_file())
            return count if count > 0 else None
        # Bazı sürümlerde artifacts kökünde direkt sayfa görselleri olabilir.
        count = sum(1 for path in artifacts_dir.glob("page-*") if path.is_file())
        return count if count > 0 else None
    except Exception:
        return None


def _resolve_docling_ref(doc: Dict[str, object], ref: str) -> Optional[Dict[str, object]]:
    """Docling JSON içindeki $ref referanslarını çözer."""
    if not ref.startswith("#/"):
        return None
    parts = ref[2:].split("/")
    node: object = doc
    for part in parts:
        if isinstance(node, list):
            try:
                node = node[int(part)]
            except (ValueError, IndexError):
                return None
        elif isinstance(node, dict):
            node = node.get(part)
        else:
            return None
    return node if isinstance(node, dict) else None


def _extract_first_page_no(obj: Dict[str, object]) -> Optional[int]:
    """Docling nesnesinden ilk sayfa numarasını alır."""
    prov = obj.get("prov")
    if isinstance(prov, list):
        for item in prov:
            if isinstance(item, dict):
                page = item.get("page_no")
                if isinstance(page, int) and page > 0:
                    return page
    return None


def _table_to_markdown(table: Dict[str, object]) -> str:
    """Docling tablo yapısını Markdown'a dönüştürür."""
    data = table.get("data")
    if not isinstance(data, dict):
        return ""
    cells = data.get("table_cells")
    if not isinstance(cells, list) or not cells:
        return ""

    row_count = 0
    col_count = 0
    header_rows: set[int] = set()
    for cell in cells:
        if not isinstance(cell, dict):
            continue
        row_count = max(row_count, int(cell.get("end_row_offset_idx", 0)))
        col_count = max(col_count, int(cell.get("end_col_offset_idx", 0)))
        if cell.get("column_header"):
            header_rows.add(int(cell.get("start_row_offset_idx", 0)))

    if row_count == 0 or col_count == 0:
        return ""

    grid: List[List[str]] = [["" for _ in range(col_count)] for _ in range(row_count)]
    for cell in cells:
        if not isinstance(cell, dict):
            continue
        text = str(cell.get("text") or "").strip()
        row_start = int(cell.get("start_row_offset_idx", 0))
        row_end = int(cell.get("end_row_offset_idx", row_start + 1))
        col_start = int(cell.get("start_col_offset_idx", 0))
        col_end = int(cell.get("end_col_offset_idx", col_start + 1))
        for r in range(row_start, row_end):
            for c in range(col_start, col_end):
                if 0 <= r < row_count and 0 <= c < col_count:
                    grid[r][c] = text

    header_index = min(header_rows) if header_rows else 0
    header = grid[header_index]
    if not any(header):
        header = [""] * col_count
    separator = ["---" for _ in header]

    rows_markdown: List[str] = []
    rows_markdown.append("| " + " | ".join(header) + " |")
    rows_markdown.append("| " + " | ".join(separator) + " |")

    for idx, row in enumerate(grid):
        if idx == header_index:
            continue
        rows_markdown.append("| " + " | ".join(row) + " |")

    return "\n".join(rows_markdown)


def _docling_text_to_markdown(obj: Dict[str, object]) -> str:
    """Docling metin nesnesini basit Markdown biçimine dönüştürür."""
    text = str(obj.get("text") or obj.get("orig") or "").strip()
    if not text:
        return ""
    label = obj.get("label")
    if label == "section_header":
        level = int(obj.get("level") or 1)
        level = max(1, min(level, 6))
        return f"{'#' * level} {text}"
    if label in {"list_item", "bullet_list_item", "unordered_list_item"}:
        return f"- {text}"
    if label in {"ordered_list_item"}:
        return f"1. {text}"
    return text


def best_page_mapping(
    md_text: str,
    json_path: Optional[Path],
    pdf_path: Path,
    artifacts_dir: Optional[Path] = None,
    debug: bool = False,
) -> List[Tuple[int, str]]:
    """
    Sayfa metinlerini tespit eder.
    - Öncelikle Docling JSON sayfa içeriği varsa onu kullanır (bu en güvenilir kaynaktır).
    - JSON yoksa veya boşsa Markdown içindeki "Sayfa" başlıklarını arar.
    - Hiçbiri başarısız olursa PDF sayfa sayısına göre Markdown karakterlerini orantılı olarak dağıtır.
    Bu tercih sırasını Türkçe yorumlarla belgeleriz.
    """
    # 1) JSON içeriğini kullanmaya çalışıyoruz; çünkü Docling JSON her bloğu sayfa ile ilişkilendiriyor.
    artifact_count = _docling_artifact_page_count(artifacts_dir)
    pdf_page_count = 0
    if pdf_path.exists():
        pdf_page_count = max(pdf_page_count, _get_pdf_page_count(pdf_path))
    if artifact_count:
        pdf_page_count = max(pdf_page_count, artifact_count)
    if debug:
        log_info(
            f"PDF sayfa tespiti -> pypdfium2/{pdf_path.name}: {pdf_page_count} | Docling artifakt sayfa sayısı: {artifact_count or 'bilinmiyor'}"
        )

    if json_path and json_path.exists():
        try:
            doc_data = json.loads(json_path.read_text(encoding="utf-8"))
        except Exception as exc:
            log_error(f"Docling JSON okunamadı: {exc}")
            doc_data = None
        if isinstance(doc_data, dict) and isinstance(doc_data.get("body"), dict):
            page_buffers: Dict[int, List[str]] = {}
            for ref in doc_data["body"].get("children", []):
                if not isinstance(ref, dict):
                    continue
                target = _resolve_docling_ref(doc_data, ref.get("$ref", ""))
                if not isinstance(target, dict):
                    continue
                page_no = _extract_first_page_no(target)
                if page_no is None:
                    continue
                label = target.get("label")
                if label == "picture":
                    continue
                if label == "table":
                    snippet = _table_to_markdown(target)
                else:
                    snippet = _docling_text_to_markdown(target)
                snippet = normalize_text_safe_markdown_tables(snippet)
                if not snippet:
                    continue
                page_buffers.setdefault(page_no, []).append(snippet)
            if page_buffers:
                total_pages = max(page_buffers.keys())
                if pdf_page_count:
                    total_pages = max(total_pages, pdf_page_count)
                mapped: List[Tuple[int, str]] = []
                for page_no in range(1, total_pages + 1):
                    snippets = page_buffers.get(page_no, [])
                    mapped.append((page_no, normalize_text_safe_markdown_tables("\n\n".join(snippets))))
                log_info("Sayfa bölümlendirmesi Docling JSON prov bilgilerinden üretildi.")
                return mapped

    # 2) Docling artifact/placeholder referanslarını kullanarak sayfaları tespit etmeye çalışıyoruz.
    #    Aşağıdaki desenler hem "referenced" hem de "placeholder" modları için yedek tespit sağlar.
    image_pattern_referenced = re.compile(r"!\[[^\]]*\]\(([^)]+image_(\d{6})_[^)]+)\)")
    # örn: .../page-0001.png veya .../page_12.jpg
    image_pattern_page_dash = re.compile(r"!\[[^\]]*\]\(([^)]*page[-_](\d{1,6})[^)]*)\)")
    # Alt metinde 'page' veya 'sayfa' ile sayıyı yakala: ![sayfa 3](...)
    alt_page_pattern = re.compile(r"!\[[^\]]*?(?:page|sayfa)\s*(\d+)\s*[^\]]*\]\([^)]+\)", re.IGNORECASE)
    # Genel görsel satırı: sayfa numarası yoksa artımsal sayım ile tahmin edeceğiz
    generic_image_line = re.compile(r"^!\[[^\]]*\]\([^)]+\)")
    lines = md_text.splitlines()
    page_buffers: Dict[int, List[str]] = {}
    current_page = 1
    seen_pages: set[int] = set()

    placeholder_seq_counter = 0
    for line in lines:
        # 2.a) Referenced modundaki "image_000001" kalıbı
        m_ref = image_pattern_referenced.search(line)
        if m_ref:
            page_candidate = int(m_ref.group(2))
            if pdf_page_count:
                page_candidate = min(page_candidate, pdf_page_count)
            current_page = page_candidate or current_page
            seen_pages.add(current_page)
            continue  # görsel satırı içeriğe dahil etmiyoruz

        # 2.b) page-0001, page_12 gibi isimler
        m_page_dash = image_pattern_page_dash.search(line)
        if m_page_dash:
            try:
                page_candidate = int(m_page_dash.group(2))
            except Exception:
                page_candidate = current_page
            if pdf_page_count:
                page_candidate = min(page_candidate, pdf_page_count)
            current_page = page_candidate or current_page
            seen_pages.add(current_page)
            continue

        # 2.c) Alt metinde sayfa numarası
        m_alt = alt_page_pattern.search(line)
        if m_alt:
            try:
                page_candidate = int(m_alt.group(1))
            except Exception:
                page_candidate = current_page
            if pdf_page_count:
                page_candidate = min(page_candidate, pdf_page_count)
            current_page = page_candidate or current_page
            seen_pages.add(current_page)
            continue

        # 2.d) Genel görsel satırı: her görsel yeni sayfa varsayılır (placeholder için yedek)
        if generic_image_line.match(line.strip()):
            placeholder_seq_counter += 1
            page_candidate = placeholder_seq_counter
            if pdf_page_count:
                page_candidate = min(page_candidate, pdf_page_count)
            current_page = page_candidate or current_page
            seen_pages.add(current_page)
            continue

        # Görsel olmayan satırlar sayfa içeriğine eklenir
        page_buffers.setdefault(current_page, []).append(line)

    if seen_pages:
        total_pages = pdf_page_count or max(seen_pages)
        mapped: List[Tuple[int, str]] = []
        for page_no in range(1, total_pages + 1):
            content_lines = page_buffers.get(page_no, [])
            mapped.append((page_no, normalize_text_safe_markdown_tables("\n".join(content_lines))))
        log_info("Sayfa bölümlendirmesi Docling görsel referanslarına göre yapıldı.")
        return mapped

    # 3) Markdown içinde muhtemel başlıkları tarıyoruz.
    pattern = re.compile(r"(?im)^\s*(?:#+\s*)?Sayfa\s+(\d+)\s*$")
    pages_with_markers: List[Tuple[int, List[str]]] = []
    current_page_num = 1
    current_buffer: List[str] = []
    found_marker = False

    for line in lines:
        marker = pattern.match(line)
        if marker:
            found_marker = True
            if current_buffer:
                pages_with_markers.append((current_page_num, current_buffer))
            current_page_num = int(marker.group(1))
            current_buffer = []
        else:
            current_buffer.append(line)

    if current_buffer:
        pages_with_markers.append((current_page_num, current_buffer))

    if found_marker and pages_with_markers:
        mapped = []
        for page_num, buffer in pages_with_markers:
            mapped.append((page_num, normalize_text_safe_markdown_tables("\n".join(buffer))))
        log_info("Sayfa bölümlendirmesi Markdown içindeki başlık işaretçilerine göre yapıldı.")
        if pdf_page_count and pdf_page_count > len(mapped):
            if debug:
                log_info(
                    f"Markdown işaretçileri {len(mapped)} sayfa verdi, ancak fiziksel sayfa sayısı {pdf_page_count}. Eksik sayfalar boş olarak eklenecek."
                )
            for idx in range(len(mapped) + 1, pdf_page_count + 1):
                mapped.append((idx, ""))
        return mapped

    # 4) Son çare olarak tüm metni tek bir sayfa altında döndürüyoruz.
    page_count = max(1, pdf_page_count or 1)
    if debug:
        log_info(f"Karakter dağılımı yaklaşımı kullanılacak. Hedef sayfa sayısı: {page_count}")
    if page_count <= 1:
        log_info("Sayfa işaretçisi bulunamadı, PDF tek sayfa varsayıldı.")
        single_page = [(1, normalize_text_safe_markdown_tables(md_text))]
        return single_page 

    md_text_no_images = re.sub(r"!\[[^\]]*\]\([^)]+\)", "", md_text)
    md_text = md_text_no_images

    total_chars = len(md_text)
    if total_chars == 0:
        log_info("Markdown boş görünüyor, sayfa bölümlendirmesi boş liste döndürdü.")
        return []

    # PDF sayfa sayısına göre orantılı dağıtım yapıyoruz. Oranlar, spesifikasyon gereği son çare yaklaşımı.
    mapped_pages: List[Tuple[int, str]] = []
    pointer = 0
    for page_no in range(1, page_count + 1):
        if page_no == page_count:
            segment = md_text[pointer:]
        else:
            approx_end = int(round(total_chars * page_no / page_count))
            approx_end = max(pointer + 1, min(approx_end, total_chars))
            # Bölme noktasını olası boşluk veya satır sonuna kaydırıyoruz ki cümleler bölünmesin.
            window = md_text[pointer:approx_end]
            cut = window.rfind("\n\n")
            if cut == -1 or cut < max(0, len(window) - 200):
                cut = window.rfind("\n")
            if cut == -1 or cut < max(0, len(window) - 200):
                cut = window.rfind(" ")
            if cut == -1:
                cut = len(window)
            segment = window[:cut]
            pointer += cut
        mapped_pages.append((page_no, normalize_text_safe_markdown_tables(segment)))

    log_info(
        f"Sayfa işaretçisi bulunamadı, Markdown karakterleri PDF sayfa sayısına göre yaklaşık dağıtıldı (toplam {page_count} sayfa)."
    )
    return mapped_pages


def chunk_text_by_chars(
    text: str,
    target_min: int,
    target_max: int,
    overlap: int,
) -> List[str]:
    """Verilen metni hedef uzunluklara göre parçalar."""
    text = text.strip()
    if not text:
        return []
    chunks: List[str] = []
    start = 0
    length = len(text)
    while start < length:
        end = min(start + target_max, length)
        if end < length:
            window = text[start:end]
            last_break = window.rfind("\n\n")
            if last_break == -1:
                last_break = window.rfind(". ")
            if last_break == -1:
                last_break = window.rfind(" ")
            if last_break != -1 and last_break + start >= start + target_min:
                end = start + last_break
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end >= length:
            break
        start = max(end - overlap, 0)
    return chunks


def embed_texts_local_first(
    model_dir: Optional[str],
    texts: Sequence[str],
    batch_size: int = 32,
) -> np.ndarray:
    """
    Trendyol embedding modelini yerelde arar; yoksa Hugging Face üzerinden indirip önbelleğe alır.
    """
    from sentence_transformers import SentenceTransformer

    key = str(Path(model_dir).resolve()) if model_dir else "__hf_trendyol_default__"
    if key not in _EMBED_MODEL_CACHE:
        model_name = "Trendyol/TY-ecomm-embed-multilingual-base-v1.2.0"
        if model_dir:
            model_path = Path(model_dir)
            model_path.mkdir(parents=True, exist_ok=True)

            def _detect_snapshot(path: Path) -> Optional[Path]:
                if (path / "config_sentence_transformers.json").exists():
                    return path
                snapshots_root = path.glob("models--Trendyol--TY-ecomm-embed-multilingual-base-v1.2.0/snapshots/*")
                snapshots = sorted([p for p in snapshots_root if p.is_dir()], key=lambda x: x.stat().st_mtime, reverse=True)
                return snapshots[0] if snapshots else None

            snapshot_path = _detect_snapshot(model_path)
            model = None
            if snapshot_path is not None:
                log_info(f"Yerel model klasörü kullanılıyor: {snapshot_path}")
                try:
                    model = SentenceTransformer(
                        str(snapshot_path),
                        device="cpu",
                        trust_remote_code=True,
                        local_files_only=True,
                    )
                except (ValueError, OSError) as exc:
                    log_info(f"Yerel klasörden yükleme başarısız oldu ({exc}). Hugging Face üzerinden güncel sürüm indirilecek.")
                    model = None
            if model is None:
                log_info(f"Model klasörü indirilecek veya güncellenecek: {model_path}")
                model = SentenceTransformer(
                    model_name,
                    device="cpu",
                    cache_folder=str(model_path),
                    local_files_only=False,
                    trust_remote_code=True,
                )
                log_info(f"Model indirildi/güncellendi: {model_path}")
        else:
            log_info(f"Model yerelde bulunamadı, Hugging Face önbelleğine indirilecek: {model_name}")
            log_info(f"Varsayılan Hugging Face önbellek konumu: {HF_HOME}")
            try:
                model = SentenceTransformer(
                    model_name,
                    device="cpu",
                    trust_remote_code=False,
                    local_files_only=False,
                )
            except ValueError as exc:
                if "trust_remote_code" in str(exc):
                    log_info("Model özel kod gerektiriyor, trust_remote_code=True ile yeniden deneniyor.")
                    model = SentenceTransformer(
                        model_name,
                        device="cpu",
                        trust_remote_code=True,
                        local_files_only=False,
                    )
                else:
                    raise
            cache_folder = getattr(model, "cache_folder", None)
            if cache_folder:
                log_info(f"Model önbellek klasörü: {cache_folder}")
        _EMBED_MODEL_CACHE[key] = model
    else:
        model = _EMBED_MODEL_CACHE[key]

    embeddings = model.encode(
        list(texts),
        batch_size=batch_size,
        show_progress_bar=False,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    if not isinstance(embeddings, np.ndarray):
        embeddings = np.array(embeddings)
    return embeddings.astype(np.float32)


def _iterate_batches(data: Sequence, batch_size: int) -> Iterable[Sequence]:
    """Genel amaçlı batch üreticisi."""
    total = len(data)
    if total == 0:
        return
    for start in range(0, total, batch_size):
        yield data[start : start + batch_size]


def parse_document_with_docling(
    pdf_path: Path,
    output_dir: Path,
    threads: int,
    batch: int,
    lang: str,
    preprocess: bool,
    preprocess_profile: str,
) -> Tuple[Path, Optional[Path], Optional[Path], Optional[Path]]:
    """PDF dosyasını (isteğe bağlı ImageMagick ön işleme ile) Docling’e aktarır."""
    if not pdf_path.exists():
        raise SystemExit(f"Girdi PDF bulunamadı: {pdf_path}")

    output_dir.mkdir(parents=True, exist_ok=True)

    effective_pdf = pdf_path
    png_dir: Optional[Path] = None
    if preprocess:
        log_info("ImageMagick ön işleme etkin.")
        effective_pdf, png_dir = preprocess_pdf_with_imagemagick(pdf_path, profile=preprocess_profile)

    md_path, json_path, artifacts_dir = robust_docling_call(
        effective_pdf,
        output_dir,
        threads=threads,
        batch=batch,
        lang=lang,
    )

    log_info("Sayfa ayrıştırması tamamlandı.")
    log_info(f"Markdown: {md_path}")
    if json_path:
        log_info(f"JSON: {json_path}")
    else:
        log_info("JSON çıktısı elde edilemedi.")
    if artifacts_dir:
        log_info(f"Artifakt klasörü: {artifacts_dir}")
    else:
        log_info("Artifakt klasörü bulunamadı.")

    return md_path, json_path, artifacts_dir, png_dir


def build_page_chunks(
    md_path: Path,
    pdf_path: Path,
    output_dir: Path,
    json_path: Optional[Path],
    artifacts_dir: Optional[Path],
    target_chars_min: int,
    target_chars_max: int,
    overlap_chars: int,
    debug_pages: bool = False,
) -> Tuple[List[Dict[str, object]], Path, Path]:
    """
    Docling Markdown çıktısından sayfa bazlı parça kayıtları üretir ve dosyalara yazar.
    """
    if not md_path.exists():
        raise SystemExit(f"Markdown dosyası bulunamadı: {md_path}")

    with md_path.open("r", encoding="utf-8") as fh:
        md_text = fh.read()

    md_text = normalize_text_safe_markdown_tables(md_text)
    pages = best_page_mapping(
        md_text,
        json_path,
        pdf_path,
        artifacts_dir=artifacts_dir,
        debug=debug_pages,
    )
    if debug_pages:
        pdf_phys_count = _get_pdf_page_count(pdf_path) if pdf_path.exists() else "?"
        log_info(f"pypdfium2 fiziksel sayfa sayısı: {pdf_phys_count}")
        log_info(f"Toplam sayfa sayısı: {len(pages)}")

    try:
        metadata = parse_filename_metadata(str(pdf_path))
    except ValueError as exc:
        raise SystemExit(f"Dosya adından metaveri çıkarılamadı: {exc}") from exc

    output_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = output_dir / "chunks.jsonl"
    csv_path = output_dir / "chunks.csv"

    all_records: List[Dict[str, object]] = []
    seen_chunk_fingerprints: Dict[str, Tuple[int, int]] = {}
    skipped_duplicates = 0

    for page_no, page_text in pages:
        page_with_markers = (
            f"[SAYFA {page_no} BAŞLANGIÇ]\n{page_text}\n[SAYFA {page_no} BİTİŞ]"
            if page_text
            else f"[SAYFA {page_no} BAŞLANGIÇ]\n[SAYFA {page_no} BİTİŞ]"
        )
        chunks = chunk_text_by_chars(
            page_with_markers,
            target_min=target_chars_min,
            target_max=target_chars_max,
            overlap=overlap_chars,
        )
        unique_chunks: List[str] = []
        for chunk in chunks:
            normalized_chunk = chunk.strip()
            if not normalized_chunk:
                continue
            fingerprint = _chunk_fingerprint(normalized_chunk)
            if fingerprint in seen_chunk_fingerprints:
                skipped_duplicates += 1
                if debug_pages:
                    prev_page, prev_idx = seen_chunk_fingerprints[fingerprint]
                    log_info(
                        f"  - Tekrar eden parça atlandı (önceden sayfa {prev_page}, parça {prev_idx})."
                    )
                continue
            seen_chunk_fingerprints[fingerprint] = (page_no, len(unique_chunks))
            unique_chunks.append(chunk)

        if debug_pages:
            log_info(
                f"Sayfa {page_no}: {len(page_text)} karakter, {len(unique_chunks)} benzersiz parça üretildi."
            )
            if len(unique_chunks) != len(chunks):
                log_info(
                    f"  - {len(chunks) - len(unique_chunks)} adet tekrar eden/boş parça filtrelendi."
                )
        for idx, chunk in enumerate(unique_chunks):
            if debug_pages:
                log_info(
                    f"  - Parça {idx}: {len(chunk)} karakter"
                )
            chunk_id = f"{slugify_tr(metadata['employee_name'])}__p{page_no}__c{idx}"
            record = {
                "employee_name": metadata["employee_name"],
                "department": metadata["department"],
                "doc_title": metadata["doc_title"],
                "file_path": str(pdf_path),
                "page": int(page_no),
                "chunk_idx": int(idx),
                "chunk_id": chunk_id,
                "text": chunk,
            }
            all_records.append(record)

    if not all_records:
        raise SystemExit("Parçalanacak içerik bulunamadı.")

    if skipped_duplicates and debug_pages:
        log_info(f"Toplam {skipped_duplicates} tekrar eden parça atlandı.")

    with jsonl_path.open("w", encoding="utf-8") as fh:
        for record in all_records:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")

    df = pd.DataFrame(all_records)
    df.to_csv(csv_path, index=False, encoding="utf-8")

    log_info(f"{len(all_records)} adet parça üretildi.")
    log_info(f"JSONL çıktı: {jsonl_path}")
    log_info(f"CSV çıktı: {csv_path}")

    return all_records, jsonl_path, csv_path


def upsert_records_to_qdrant(
    client: QdrantClient,
    collection: str,
    records: Sequence[Dict[str, object]],
    model_dir: Optional[str],
    batch_size: int,
    show_progress: bool = True,
) -> int:
    """
    Parça kayıtlarını Trendyol embedding modeliyle vektörleyip Qdrant koleksiyonuna yükler.
    Koleksiyon boyutunu döndürür.
    """
    texts = [str(r["text"]) for r in records]
    embeddings = embed_texts_local_first(model_dir, texts, batch_size=batch_size)
    if embeddings.shape[0] != len(records):
        raise SystemExit("Embedding sayısı kayıt sayısıyla uyumsuz.")

    vector_size = embeddings.shape[1]

    try:
        client.get_collection(collection)
        log_info(f"'{collection}' koleksiyonu mevcut, upsert işlemi yapılacak.")
    except Exception:
        log_info(f"'{collection}' koleksiyonu oluşturuluyor (mesafe: cosine, boyut: {vector_size}).")
        
        client.create_collection(
            collection_name=collection,
            vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
        )

    points: List[PointStruct] = []
    for record, embedding in zip(records, embeddings):
        payload = {
            "employee_name": record.get("employee_name"),
            "department": record.get("department"),
            "doc_title": record.get("doc_title"),
            "file_path": record.get("file_path"),
            "page": record.get("page"),
            "chunk_id": record.get("chunk_id"),
            "text": record.get("text"),
        }
        chunk_id = str(record.get("chunk_id"))
        if not chunk_id:
            chunk_id = str(uuid.uuid4())
        point_uuid = str(uuid.uuid5(uuid.NAMESPACE_URL, chunk_id))
        points.append(
            PointStruct(
                id=point_uuid,
                vector=embedding.tolist(),
                payload=payload,
            )
        )

    for batch_points in tqdm(
        list(_iterate_batches(points, batch_size)),
        desc="Qdrant upsert işlemi",
        unit="paket",
        disable=not show_progress,
    ):
        client.upsert(collection_name=collection, points=batch_points)

    log_info("Upsert işlemi tamamlandı.")
    return vector_size


def search_qdrant(
    client: QdrantClient,
    collection: str,
    query: str,
    model_dir: Optional[str],
    top_k: int,
) -> List:
    """Trendyol modeli ile sorguyu vektörleyip Qdrant üzerinden arama yapar."""
    embeddings = embed_texts_local_first(model_dir, [query], batch_size=1)
    query_vector = embeddings[0].tolist()

    results = client.search(
        collection_name=collection,
        query_vector=query_vector,
        limit=top_k,
        with_payload=True,
    )
    return results


def _create_qdrant_client(config: RagServiceConfig) -> QdrantClient:
    """Konfigürasyona göre Qdrant istemcisi oluşturur."""
    return QdrantClient(url=config.qdrant_url, api_key=config.qdrant_api_key or None)


def _metadata_friendly_stem(pdf_path: Path) -> str:
    """Doc metaverisi çıkarımı için güvenli dosya adı üretir."""
    try:
        parse_filename_metadata(str(pdf_path))
        return pdf_path.stem
    except ValueError:
        stem = pdf_path.stem
        candidate_employee = stem
        candidate_department = "Genel"
        for separator in ("_", " ", "."):
            if separator in stem:
                parts = stem.split(separator, 1)
                candidate_employee = parts[0] or candidate_employee
                candidate_department = parts[1] or candidate_department
                break
        employee_slug = slugify_tr(candidate_employee or "YuklenenBelge").replace("-", "")
        if not employee_slug:
            employee_slug = "yuklenenbelge"
        department_slug = slugify_tr(candidate_department or "Genel").replace("-", "")
        if not department_slug:
            department_slug = "genel"
        return f"{employee_slug}-{department_slug}"


def _yield_text_chunks(text: str, chunk_size: int = 512) -> Iterator[str]:
    """Uzun çıktıları küçük parçalar halinde üretir."""
    if not text:
        return
    for idx in range(0, len(text), chunk_size):
        yield text[idx : idx + chunk_size]


def _format_search_results(results: Sequence) -> str:
    """Qdrant arama sonuçlarını okunabilir metne dönüştürür."""
    if not results:
        return "Eşleşme bulunamadı."
    lines: List[str] = []
    for idx, match in enumerate(results, 1):
        payload = getattr(match, "payload", {}) or {}
        doc_title = str(payload.get("doc_title", "bilinmiyor"))
        page = payload.get("page", "?")
        score = getattr(match, "score", None)
        header = f"{idx}. {doc_title} (sayfa {page})"
        if isinstance(score, (int, float)):
            header += f" | skor: {score:.3f}"
        lines.append(header)
        chunk_text = str(payload.get("text") or "").strip()
        if chunk_text:
            cleaned = re.sub(r"\s+", " ", chunk_text)
            if len(cleaned) > 900:
                cleaned = cleaned[:900].rstrip() + "..."
            lines.append(cleaned)
        lines.append("")
    return "\n".join(lines).strip()


_CHUNK_MARKER_RE = re.compile(r"\[SAYFA\s+\d+\s+(?:BAŞLANGIÇ|BİTİŞ)]")


def _chunk_fingerprint(text: str) -> str:
    """Parça karşılaştırması için sayfa işaretçilerini temizler ve normalize eder."""
    cleaned = _CHUNK_MARKER_RE.sub("", text)
    cleaned = re.sub(r"\s+", " ", cleaned).strip().lower()
    return cleaned


def _build_llm_context_blocks(results: Sequence, max_chunks: int) -> List[str]:
    """LLM'e gönderilecek bağlam bloklarını hazırlar."""
    blocks: List[str] = []
    for idx, match in enumerate(results, 1):
        if idx > max_chunks:
            break
        payload = getattr(match, "payload", {}) or {}
        doc_title = str(payload.get("doc_title", "Bilinmeyen doküman"))
        page = payload.get("page", "?")
        text = str(payload.get("text") or "")
        text = re.sub(r"\s+", " ", text).strip()
        if not text:
            continue
        if len(text) > 1200:
            text = text[:1200].rstrip() + "..."
        blocks.append(f"[Kaynak {idx}] Doküman: {doc_title} | Sayfa: {page}\n{text}")
    return blocks


def _compose_ollama_prompt(question: str, context_blocks: Sequence[str]) -> str:
    context_text = "\n\n".join(context_blocks) if context_blocks else "(Bağlam sağlanmadı)"
    return (
        "Sen Mitaş insan kaynakları için belge tabanlı çalışan bir yardımcı asistanısın."
        " Verilen bağlamdaki bilgilerle soruyu TÜRKÇE olarak yanıtla."
        " Bağlamda olmayan detayları üretme. Gerekirse 'bağlamda bilgi yok' de."
        " Cevap verirken kısa bir özet ve gerektiğinde maddeler halinde cevap ver."
        " Yanıtında her bilgi için ilgili kaynak sayfa numarasını belirt (örn. 'Kaynak: Sayfa 3')."
        "\n\nSoru:\n" + question.strip() +
        "\n\nBağlam:\n" + context_text +
        "\n\nYanıt:"
    )


def _stream_ollama_answer(
    config: RagServiceConfig,
    question: str,
    context_blocks: Sequence[str],
) -> Iterator[str]:
    """Ollama modeli üzerinden bağlama dayalı yanıt üretir."""

    prompt = _compose_ollama_prompt(question, context_blocks)
    base_url = config.ollama_url.rstrip("/")
    endpoint = f"{base_url}/api/generate"
    payload = {
        "model": config.ollama_model,
        "prompt": prompt,
        "stream": True,
        "options": {
            "temperature": config.ollama_temperature,
        },
    }

    try:
        response = requests.post(endpoint, json=payload, stream=True, timeout=(10, 600))
    except requests.RequestException as exc:
        yield f"Hata: Ollama isteği başarısız oldu ({exc}).\n"
        return

    if response.status_code != 200:
        try:
            error_body = response.json()
        except Exception:
            error_body = response.text
        yield f"Hata: Ollama {response.status_code} döndürdü: {error_body}\n"
        return

    for line in response.iter_lines(decode_unicode=True):
        if not line:
            continue
        try:
            data = json.loads(line)
        except json.JSONDecodeError:
            continue
        chunk = data.get("response")
        if chunk:
            yield chunk
        if data.get("done"):
            break

    yield "\n"


def process_and_run_stream(
    pdf_path: str,
    question: str = "",
    *,
    config: Optional[RagServiceConfig] = None,
) -> Iterator[str]:
    """Web arabirimi için uçtan uca RAG hattını parça parça çalıştırır."""

    cfg = config or RagServiceConfig.from_env()
    original_pdf = Path(pdf_path)
    if not original_pdf.exists():
        yield f"Hata: PDF bulunamadı ({original_pdf}).\n"
        return

    question = (question or "").strip()
    yield "RAG hattı başlatıldı.\n"
    yield "[PROGRESS 5]\n"

    try:
        with tempfile.TemporaryDirectory(prefix="hr_rag_ui_") as work_root_str:
            work_root = Path(work_root_str)
            safe_stem = _metadata_friendly_stem(original_pdf)
            suffix = original_pdf.suffix or ".pdf"
            safe_pdf_path = work_root / f"{safe_stem}{suffix}"
            shutil.copy2(original_pdf, safe_pdf_path)
            collection_name = _collection_name_for_document(cfg.collection, safe_stem)
            yield f"İşlenecek dosya: {original_pdf.name}\n"
            yield "[PROGRESS 10]\n"

            if cfg.persist_outputs:
                output_root = cfg.ensure_outputs_root()
                run_dir = output_root / f"{safe_stem}_{uuid.uuid4().hex[:8]}"
            else:
                run_dir = work_root / "docling"
            chunks_dir = run_dir / "chunks"

            yield "[PROGRESS 15]\n"
            md_path, json_path, artifacts_dir, preprocess_assets = parse_document_with_docling(
                pdf_path=safe_pdf_path,
                output_dir=run_dir,
                threads=cfg.threads,
                batch=cfg.docling_batch,
                lang=cfg.lang,
                preprocess=cfg.preprocess,
                preprocess_profile=cfg.preprocess_profile,
            )
            yield "Sayfa ayrıştırması tamamlandı.\n"
            yield "[PROGRESS 60]\n"
            if cfg.persist_outputs:
                yield f"OCR çıktıları: {run_dir}\n"

            records, jsonl_path, csv_path = build_page_chunks(
                md_path=md_path,
                pdf_path=safe_pdf_path,
                output_dir=chunks_dir,
                json_path=json_path,
                artifacts_dir=artifacts_dir,
                target_chars_min=cfg.target_chars_min,
                target_chars_max=cfg.target_chars_max,
                overlap_chars=cfg.overlap_chars,
                debug_pages=False,
            )

            source_name = original_pdf.name
            for record in records:
                record["file_path"] = source_name

            yield f"{len(records)} adet bilgi parçası üretildi.\n"
            yield "[PROGRESS 70]\n"
            if cfg.persist_outputs:
                yield f"Parça dosyaları: {jsonl_path.name}, {csv_path.name}\n"

            client = _create_qdrant_client(cfg)
            vector_size = upsert_records_to_qdrant(
                client=client,
                collection=collection_name,
                records=records,
                model_dir=cfg.model_dir,
                batch_size=cfg.embedding_batch_size,
                show_progress=False,
            )

            # İstemci ve sunucu koleksiyonu takip edebilsin diye makine-okur işaret bırakıyoruz
            yield f"[COLLECTION {collection_name}]\n"
            yield f"Qdrant koleksiyonu güncellendi (koleksiyon: {collection_name}, boyut: {vector_size}).\n"
            yield "[PROGRESS 85]\n"

            if question:
                yield f"Soru işleniyor: {question}\n"
                results = search_qdrant(
                    client=client,
                    collection=collection_name,
                    query=question,
                    model_dir=cfg.model_dir,
                    top_k=cfg.top_k,
                )
                formatted = _format_search_results(results)
                if not results:
                    for chunk in _yield_text_chunks(formatted):
                        yield chunk
                    yield "\n"
                    yield "[PROGRESS 100]\n"
                else:
                    if cfg.use_ollama:
                        yield "Kaynak parçalar (LLM bağlamı):\n"
                        for chunk in _yield_text_chunks(formatted):
                            yield chunk
                        yield "\nOllama yanıtı hazırlanıyor (model: "+cfg.ollama_model+")...\n"
                        context_blocks = _build_llm_context_blocks(
                            results,
                            max_chunks=cfg.ollama_max_context_chunks,
                        )
                        if context_blocks:
                            for chunk in _stream_ollama_answer(cfg, question, context_blocks):
                                yield chunk
                            yield "[PROGRESS 100]\n"
                        else:
                            yield "Hata: LLM için bağlam oluşturulamadı.\n"
                            yield "[PROGRESS 100]\n"
                    else:
                        for chunk in _yield_text_chunks(formatted):
                            yield chunk
                        yield "\n"
                        yield "[PROGRESS 100]\n"
            else:
                yield "Soru verilmedi, yalnızca indeksleme tamamlandı.\n"
                yield "[PROGRESS 100]\n"

            if preprocess_assets and not cfg.persist_outputs:
                shutil.rmtree(preprocess_assets.parent, ignore_errors=True)

    except SystemExit as exc:
        message = str(exc) or "İşlem kullanıcı tarafından sonlandırıldı."
        log_error(message)
        yield f"Hata: {message}\n"
    except Exception as exc:
        log_error(f"RAG hattı beklenmedik biçimde durdu: {exc}")
        yield f"Hata: {exc}\n"


def chat_stream_no_context(
    prompt: str,
    *,
    config: Optional[RagServiceConfig] = None,
) -> Iterator[str]:
    """Sadece vektör veritabanına dayanarak hızlı sorgu sağlar."""

    cfg = config or RagServiceConfig.from_env()
    query = (prompt or "").strip()
    if not query:
        query = "Merhaba!"

    # Kullanıcı deneyimi için başlangıçta 'Sorgu: ...' satırı yazdırmıyoruz.

    if not cfg.chat_use_collection:
        if cfg.use_ollama:
            yield f"Ollama yanıtı hazırlanıyor (model: {cfg.ollama_model})...\n"
            context_blocks: List[str] = []
            for chunk in _stream_ollama_answer(cfg, query, context_blocks):
                yield chunk
        else:
            yield (
                "Uyarı: Chat için etkin motor yok. LLM kapalı (HR_RAG_USE_OLLAMA=0)"
                " ve koleksiyon kullanımı devre dışı (HR_RAG_CHAT_USE_COLLECTION=0). "
                "LLM'i açın veya koleksiyon kullanımını etkinleştirin.\n"
            )
        return

    try:
        client = _create_qdrant_client(cfg)
        results = search_qdrant(
            client=client,
            collection=cfg.collection,
            query=query,
            model_dir=cfg.model_dir,
            top_k=cfg.top_k,
        )
        formatted = _format_search_results(results)
        if not results:
            for chunk in _yield_text_chunks(formatted):
                yield chunk
            yield "\n"
            return

        if cfg.use_ollama:
            yield "Kaynak parçalar (LLM bağlamı):\n"
            for chunk in _yield_text_chunks(formatted):
                yield chunk
            yield "\nOllama yanıtı hazırlanıyor (model: "+cfg.ollama_model+")...\n"
            context_blocks = _build_llm_context_blocks(
                results,
                max_chunks=cfg.ollama_max_context_chunks,
            )
            if context_blocks:
                for chunk in _stream_ollama_answer(cfg, query, context_blocks):
                    yield chunk
            else:
                yield "Hata: LLM için bağlam oluşturulamadı.\n"
        else:
            for chunk in _yield_text_chunks(formatted):
                yield chunk
            yield "\n"
    except SystemExit as exc:
        message = str(exc) or "Arama işlemi sonlandırıldı."
        log_error(message)
        yield f"Hata: {message}\n"
    except Exception as exc:
        log_error(f"Chat sorgusu başarısız: {exc}")
        yield f"Hata: {exc}\n"


def chat_stream_for_collection(
    prompt: str,
    collection_name: str,
    *,
    config: Optional[RagServiceConfig] = None,
) -> Iterator[str]:
    """Belirtilen koleksiyon üzerinde arama yaparak sohbet akışını döndürür."""
    cfg = config or RagServiceConfig.from_env()
    query = (prompt or "").strip() or "Merhaba!"

    try:
        client = _create_qdrant_client(cfg)
        results = search_qdrant(
            client=client,
            collection=collection_name,
            query=query,
            model_dir=cfg.model_dir,
            top_k=cfg.top_k,
        )
        formatted = _format_search_results(results)
        if not results:
            for chunk in _yield_text_chunks(formatted):
                yield chunk
            yield "\n"
            return

        if cfg.use_ollama:
            yield "Kaynak parçalar (LLM bağlamı):\n"
            for chunk in _yield_text_chunks(formatted):
                yield chunk
            yield "\nOllama yanıtı hazırlanıyor (model: "+cfg.ollama_model+")...\n"
            context_blocks = _build_llm_context_blocks(
                results,
                max_chunks=cfg.ollama_max_context_chunks,
            )
            if context_blocks:
                for chunk in _stream_ollama_answer(cfg, query, context_blocks):
                    yield chunk
            else:
                yield "Hata: LLM için bağlam oluşturulamadı.\n"
        else:
            for chunk in _yield_text_chunks(formatted):
                yield chunk
            yield "\n"
    except SystemExit as exc:
        message = str(exc) or "Arama işlemi sonlandırıldı."
        log_error(message)
        yield f"Hata: {message}\n"
    except Exception as exc:
        log_error(f"Belirli koleksiyon chat başarısız: {exc}")
        yield f"Hata: {exc}\n"


def build_arg_parser() -> argparse.ArgumentParser:
    """Argüman ayrıştırıcıyı tek adımlı çalışma için yapılandırır."""
    parser = argparse.ArgumentParser(
        description="PDF -> Docling -> RAG -> Qdrant -> Arama hattını tek adımda çalıştırır.",
    )
    parser.add_argument("--pdf", required=True, help="İşlenecek PDF dosyası (AdSoyad-Bölüm.pdf)")
    parser.add_argument(
        "--output-dir",
        help="Docling çıktılarının yazılacağı klasör (varsayılan: PDF klasörü içinde {stem}_docling)",
    )
    parser.add_argument(
        "--chunks-dir",
        help="Parça dosyalarının yazılacağı klasör (varsayılan: output-dir/chunks)",
    )
    parser.add_argument("--threads", type=int, default=4, help="Docling iş parçacığı sayısı")
    parser.add_argument("--batch", type=int, default=2, help="Docling sayfa işleme paket boyutu")
    parser.add_argument("--lang", default="tr", help="RapidOCR dili (varsayılan: tr)")
    parser.add_argument(
        "--preprocess",
        action="store_true",
        help="ImageMagick tabanlı 400 DPI gri ön işleme adımını uygula.",
    )
    parser.add_argument(
        "--preprocess-profile",
        choices=["auto", "color", "bw"],
        default="bw",
        help="OCR ön işleme profili: bw (varsayılan), auto veya color",
    )
    parser.add_argument(
        "--target-chars-min",
        type=int,
        default=1200,
        help="Parça alt karakter sınırı",
    )
    parser.add_argument(
        "--target-chars-max",
        type=int,
        default=1600,
        help="Parça üst karakter sınırı",
    )
    parser.add_argument(
        "--overlap-chars",
        type=int,
        default=220,
        help="Parçalar arası karakter çakışması",
    )
    parser.add_argument(
        "--debug-pages",
        action="store_true",
        help="Sayfa ve parça uzunluklarını ayrıntılı olarak günlüğe yaz.",
    )
    parser.add_argument("--collection", required=True, help="Qdrant koleksiyon adı")
    parser.add_argument("--model-dir", help="Yerel Trendyol model klasörü")
    parser.add_argument("--qdrant-url", help="Qdrant sunucu URL'si (varsayılan env veya localhost)")
    parser.add_argument("--qdrant-api-key", help="Qdrant API anahtarı (opsiyonel)")
    parser.add_argument(
        "--batch-size",
        type=int,
        default=256,
        help="Embedding ve upsert paket boyutu",
    )
    parser.add_argument("--top-k", type=int, default=3, help="Aramada döndürülecek sonuç sayısı")
    parser.add_argument("--query", help="Arama sorgusu (boşsa komut satırından istenir)")
    return parser


def run_full_pipeline(args: argparse.Namespace) -> None:
    """PDF'ten sorgu sonuçlarına kadar tüm hattı çalıştırır."""
    pdf_path = Path(args.pdf).resolve()
    if not pdf_path.exists():
        raise SystemExit(f"Girdi PDF bulunamadı: {pdf_path}")
    safe_stem = _metadata_friendly_stem(pdf_path)
    collection_name = _collection_name_for_document(args.collection, safe_stem)

    output_dir = (
        Path(args.output_dir).resolve()
        if args.output_dir
        else pdf_path.parent / f"{pdf_path.stem}_docling"
    )
    chunks_dir = (
        Path(args.chunks_dir).resolve()
        if args.chunks_dir
        else output_dir / "chunks"
    )

    log_info(f"PDF dosyası: {pdf_path}")
    log_info(f"Docling çıktı klasörü: {output_dir}")
    log_info(f"Parça çıktı klasörü: {chunks_dir}")
    log_info(f"Koleksiyon adı: {collection_name}")

    md_path, json_path, artifacts_dir, preprocess_assets = parse_document_with_docling(
        pdf_path=pdf_path,
        output_dir=output_dir,
        threads=args.threads,
        batch=args.batch,
        lang=args.lang,
        preprocess=args.preprocess,
        preprocess_profile=args.preprocess_profile,
    )

    if preprocess_assets:
        log_info(f"ImageMagick ara PNG klasörü: {preprocess_assets}")

    records, jsonl_path, csv_path = build_page_chunks(
        md_path=md_path,
        pdf_path=pdf_path,
        output_dir=chunks_dir,
        json_path=json_path,
        artifacts_dir=artifacts_dir,
        target_chars_min=args.target_chars_min,
        target_chars_max=args.target_chars_max,
        overlap_chars=args.overlap_chars,
        debug_pages=args.debug_pages,
    )

    log_info(f"Parça kayıt sayısı: {len(records)}")
    log_info(f"Parça dosyaları: {jsonl_path}, {csv_path}")

    url = args.qdrant_url or os.environ.get("QDRANT_URL", "http://localhost:6333")
    api_key = args.qdrant_api_key or os.environ.get("QDRANT_API_KEY", "")

    log_info(f"Qdrant sunucusu: {url}")
    if api_key:
        log_info("Qdrant API anahtarı kullanılacak.")
    else:
        log_info("Qdrant API anahtarı kullanılmıyor.")

    client = QdrantClient(url=url, api_key=api_key or None)
    vector_size = upsert_records_to_qdrant(
        client=client,
        collection=collection_name,
        records=records,
        model_dir=args.model_dir,
        batch_size=args.batch_size,
    )
    log_info(f"Embedding boyutu: {vector_size}")

    query = args.query
    if not query:
        try:
            query = input("Arama sorgusu girin: ").strip()
        except EOFError:
            query = ""

    if not query:
        log_info("Arama sorgusu verilmedi, işlem burada sonlandırıldı.")
        return

    results = search_qdrant(
        client=client,
        collection=collection_name,
        query=query,
        model_dir=args.model_dir,
        top_k=args.top_k,
    )

    if not results:
        log_info("Eşleşme bulunamadı.")
        return

    log_info(f"Arama sorgusu: {query}")
    log_info(f"En iyi {min(args.top_k, len(results))} sonuç:")
    for match in results:
        payload = match.payload or {}
        doc_title = payload.get("doc_title", "bilinmiyor")
        page = payload.get("page", "?")
        chunk_id = payload.get("chunk_id", "?")
        score = getattr(match, "score", None)
        if score is None:
            score = getattr(match, "payload", {}).get("score")
        if isinstance(score, (int, float)):
            log_info(
                f"Kaynak → {doc_title} | sayfa: {page} | chunk_id: {chunk_id} | skor: {score:.3f}"
            )
        else:
            log_info(
                f"Kaynak → {doc_title} | sayfa: {page} | chunk_id: {chunk_id}"
            )


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Komut satırı giriş noktası."""
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    try:
        run_full_pipeline(args)
    except KeyboardInterrupt:  # pragma: no cover - kullanıcı iptali
        log_error("Kullanıcı tarafından durduruldu.")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""Bounded *local* document decoders. The EvidenceStore owns resulting text.

No network fetch, OCR, JavaScript or executable document interpretation.
"""
import hashlib
import io
import json
import re
from html.parser import HTMLParser
from pathlib import Path

MAX_FILE_BYTES = 5_000_000
MAX_TEXT_CHARS = 1_500_000
EXTENSIONS = ('.txt', '.md', '.csv', '.json', '.html', '.htm', '.pdf')

class _HTMLText(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.parts = []
        self.omitted = 0
    def handle_starttag(self, tag, attrs):
        if tag in ('script', 'style', 'noscript'):
            self.omitted += 1
        elif tag in ('p','div','br','li','h1','h2','h3','tr'):
            self.parts.append('\n')
    def handle_endtag(self, tag):
        if tag in ('script', 'style', 'noscript') and self.omitted:
            self.omitted -= 1
        elif tag in ('p','li','div','tr'):
            self.parts.append('\n')
    def handle_data(self, data):
        if not self.omitted:
            self.parts.append(data)

def decode_document(filename, data):
    """Return (normalized_text, original_content_hash, mime_type) or refuse."""
    name = Path(str(filename)).name
    ext = Path(name).suffix.lower()
    if ext not in EXTENSIONS:
        raise ValueError('Unsupported file. Use TXT, MD, CSV, JSON, HTML or PDF.')
    if not isinstance(data, bytes) or not 20 <= len(data) <= MAX_FILE_BYTES:
        raise ValueError('File must contain 20 bytes to 5 MB of data')
    digest = hashlib.sha256(data).hexdigest()
    mime = {'.pdf':'application/pdf', '.html':'text/html', '.htm':'text/html',
            '.csv':'text/csv','.json':'application/json', '.md':'text/markdown'}.get(ext, 'text/plain')
    if ext == '.pdf':
        try:
            from pypdf import PdfReader
        except ImportError as exc:
            raise ValueError('PDF support requires: .venv/bin/pip install pypdf') from exc
        try:
            reader = PdfReader(io.BytesIO(data), strict=True)
            if reader.is_encrypted:
                raise ValueError('Encrypted PDF cannot be imported')
            if len(reader.pages) > 200:
                raise ValueError('PDF has more than the 200-page limit')
            text = '\n'.join((page.extract_text() or '') for page in reader.pages)
        except ValueError:
            raise
        except Exception as exc:
            raise ValueError('PDF could not be decoded (scanned/image-only PDFs need OCR)') from exc
    else:
        try:
            text = data.decode('utf-8-sig', errors='strict')
        except UnicodeDecodeError as exc:
            raise ValueError('Text files must use UTF-8 encoding') from exc
        if ext in ('.html', '.htm'):
            parser = _HTMLText()
            parser.feed(text)
            text = ' '.join(parser.parts)
        elif ext == '.json':
            try:
                parsed = json.loads(text)
            except json.JSONDecodeError as exc:
                raise ValueError('Invalid JSON document') from exc
            text = json.dumps(parsed, ensure_ascii=False, indent=2)
    text = re.sub(r'[ \t]+', ' ', text).strip()
    if not 20 <= len(text) <= MAX_TEXT_CHARS:
        raise ValueError('Extracted text must contain 20 to 1,500,000 characters')
    return text, digest, mime

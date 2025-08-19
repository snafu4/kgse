"""Utilities for compressing and decompressing graph data."""

import base64
import json
import zlib
from typing import Any, Dict


class KGCompressor:
    """Helper methods to compress and decompress knowledge graph structures."""

    @staticmethod
    def compress_data(data: Dict[str, Any]) -> bytes:
        """Compress a dictionary into base64-encoded zlib-compressed bytes."""
        json_string = json.dumps(data)
        compressed_data = zlib.compress(json_string.encode("utf-8"), level=9)
        return base64.b64encode(compressed_data)

    @staticmethod
    def decompress_data(encoded_data: bytes) -> Dict[str, Any]:
        """Decode base64-encoded zlib-compressed data back into a dictionary."""
        compressed_data = base64.b64decode(encoded_data)
        json_string = zlib.decompress(compressed_data).decode("utf-8")
        return json.loads(json_string)

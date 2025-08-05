import json
import zlib
import base64

class KGCompressor:
    @staticmethod
    def compress_data(data):
        """
        Compresses a dictionary of data.
        """
        json_string = json.dumps(data)
        compressed_data = zlib.compress(json_string.encode('utf-8'), level=9)
        return base64.b64encode(compressed_data)

    @staticmethod
    def decompress_data(encoded_data):
        """
        Decompresses data.
        """
        compressed_data = base64.b64decode(encoded_data)
        json_string = zlib.decompress(compressed_data).decode('utf-8')
        return json.loads(json_string)
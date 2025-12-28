from enum import Enum

SUPPORTED_FILE_EXTENSIONS = [".csv", ".parquet", ".h5"]

class DataReadingErrorMessages(Enum):
    INVALID_FILE_PATH_TYPE = "Invalid file path type: {type}. Expected str or Path."
    FILE_NOT_FOUND = "File not found at: {file_path}"
    EXT_NOT_SUPPORTED = "Extension {ext} not supported. Supported: {supported_extensions}"
    EMPTY_DATA_FILE = "The provided data file is empty."
    KAFKA_SEND_ERROR = "Failed to send data to Kafka: {error}"
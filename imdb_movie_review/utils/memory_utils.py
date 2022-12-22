# Define process object globally, so we don't reinitialize this everytime we get the memory usage
import psutil

process = psutil.Process()


def get_memory_usage(message: str) -> None:
    """
    Loading the data into memory, and the training process (fit) can take alot of memory.
    """
    global process
    memory_info = process.memory_info()
    print(
        f"{message}: Total memory used by process: {memory_info.rss / 1024**2:.2f} MB"
    )

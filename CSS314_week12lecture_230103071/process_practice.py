import multiprocessing as mp
import os


def worker():
    print("Worker process ID:", os.getpid())


if __name__ == "__main__":
    p = mp.Process(target=worker)
    p.start()
    p.join()

    print("Main process ID:", os.getpid())
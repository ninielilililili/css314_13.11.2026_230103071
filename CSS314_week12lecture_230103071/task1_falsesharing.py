
import ctypes
from multiprocessing import shared_memory


def worker(shm_name, slot, iterations):
    """
    Attach to shared memory and update one uint64 slot.
    """

    shm = shared_memory.SharedMemory(name=shm_name)

    value = ctypes.c_uint64.from_buffer(
        shm.buf,
        slot * 8
    )

    accumulator = value.value

    multiplier = 6364136223846793005
    mask = 0xFFFFFFFFFFFFFFFF

    for i in range(iterations):
        accumulator = (
            accumulator * multiplier + (i | 1)
        ) & mask

    value.value = accumulator

    del value
    shm.close()

if __name__ == "__main__":
    from multiprocessing import Process

    shm = shared_memory.SharedMemory(
        create=True,
        size=4096
    )

    try:
        processes = []

        for slot in range(2):
            p = Process(
                target=worker,
                args=(shm.name, slot, 1000)
            )
            processes.append(p)
            p.start()

        for p in processes:
            p.join()

        values = ctypes.cast(
            ctypes.addressof(
                ctypes.c_char.from_buffer(shm.buf)
            ),
            ctypes.POINTER(ctypes.c_uint64)
        )

        print("Slot 0:", values[0])
        print("Slot 1:", values[1])

        del values

    finally:
        shm.close()
        shm.unlink()

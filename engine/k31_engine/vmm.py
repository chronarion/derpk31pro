"""Shared staging rows through CUDA virtual memory: every MoE layer's kernel tensor is one virtual range whose first
`nslots` rows map the same physical HBM (the staging slots, shared by all layers) and whose remaining rows map the
layer's own memory (its resident experts). 92 layers x 16 private slots would cost 25.8 GB of HBM; 128 shared slots
cost 2.25 GB. 128 is what the 2 MiB mapping granularity asks for: the slot region must end on a granule for all four
tensors (w2's scale rows are 344,064 B: 128 of them are 21 granules). Only HBM pages are mapped (derpseek measured
VMM-mapped host pages at 91 GB/s for TMA; HBM pages are ordinary device memory).
"""
import ctypes

import torch

_cu = None
_KEEP = []                      # handles and reservations live for the process


class _Prop(ctypes.Structure):  # CUmemAllocationProp (CUDA 12/13 layout)
    _fields_ = [("type", ctypes.c_int), ("requestedHandleTypes", ctypes.c_int),
                ("loc_type", ctypes.c_int), ("loc_id", ctypes.c_int), ("win32HandleMetaData", ctypes.c_void_p),
                ("compressionType", ctypes.c_ubyte), ("gpuDirectRDMACapable", ctypes.c_ubyte),
                ("usage", ctypes.c_ushort), ("reserved", ctypes.c_ubyte * 4)]


class _Access(ctypes.Structure):  # CUmemAccessDesc
    _fields_ = [("loc_type", ctypes.c_int), ("loc_id", ctypes.c_int), ("flags", ctypes.c_int)]


def _lib():
    global _cu
    if _cu is None:
        _cu = ctypes.CDLL("libcuda.so.1")
    return _cu


def _ck(rc, what):
    if rc != 0:
        raise RuntimeError(f"K31 vmm: {what} returned CUresult {rc}")


def _prop(dev_index):
    p = _Prop()
    p.type = 1                  # CU_MEM_ALLOCATION_TYPE_PINNED
    p.loc_type = 1              # CU_MEM_LOCATION_TYPE_DEVICE
    p.loc_id = dev_index
    return p


def granularity(dev_index) -> int:
    g = ctypes.c_size_t()
    _ck(_lib().cuMemGetAllocationGranularity(ctypes.byref(g), ctypes.byref(_prop(dev_index)), 0), "granularity")
    return g.value


def _create(nbytes, dev_index):
    h = ctypes.c_uint64()
    _ck(_lib().cuMemCreate(ctypes.byref(h), ctypes.c_size_t(nbytes), ctypes.byref(_prop(dev_index)),
                           ctypes.c_uint64(0)), "cuMemCreate")
    _KEEP.append(h)
    return h


class _Arr:
    def __init__(self, ptr, shape, strides_bytes, typestr):
        self.__cuda_array_interface__ = {"shape": tuple(shape), "typestr": typestr, "data": (ptr, False),
                                         "strides": tuple(strides_bytes), "version": 3}


_TS = {torch.int8: "|i1", torch.uint8: "|u1", torch.int32: "<i4"}


class SharedSlots:
    """For one kernel tensor kind (shape tail and strides of one expert row): the shared physical slot memory, and
    per layer a tensor [nslots + own, ...] over a fresh virtual range."""

    def __init__(self, like: torch.Tensor, nslots: int):
        self.dev = like.device
        self.di = like.device.index if like.device.index is not None else torch.cuda.current_device()
        self.dtype, self.tail = like.dtype, tuple(like.shape[1:])
        self.esize = like.element_size()
        self.row = like[0].numel() * self.esize
        assert like.stride(0) * self.esize == self.row, "rows must be dense blocks"
        self.inner = tuple(s * self.esize for s in like.stride()[1:])
        self.g = granularity(self.di)
        self.nslots = nslots
        self.slot_bytes = nslots * self.row
        # the slot region must end on a granule: rows whose slots are not whole granules (6-bit scale rows, sf6rows.py:
        # 128 x 516,096 B = 31.5 granules) start `front` bytes into the shared mapping
        self.front = (-self.slot_bytes) % self.g if nslots else 0
        if self.front % 128:
            raise RuntimeError(f"K31 vmm: {nslots} slots of {self.row} B leave rows off a 128-byte boundary")
        self.slots = _create(self.front + self.slot_bytes, self.di) if nslots else None

    def tensor(self, own_rows: int) -> torch.Tensor:
        lib = _lib()
        own = -(-own_rows * self.row // self.g) * self.g
        shared = self.front + self.slot_bytes
        total = shared + own
        va = ctypes.c_uint64()
        _ck(lib.cuMemAddressReserve(ctypes.byref(va), ctypes.c_size_t(total), ctypes.c_size_t(0), ctypes.c_uint64(0),
                                    ctypes.c_uint64(0)), "cuMemAddressReserve")
        _KEEP.append(va)
        if self.nslots:
            _ck(lib.cuMemMap(va, ctypes.c_size_t(shared), ctypes.c_size_t(0), self.slots, ctypes.c_uint64(0)),
                "cuMemMap slots")
        if own:
            h = _create(own, self.di)
            _ck(lib.cuMemMap(ctypes.c_uint64(va.value + shared), ctypes.c_size_t(own), ctypes.c_size_t(0), h,
                             ctypes.c_uint64(0)), "cuMemMap own rows")
        acc = _Access(1, self.di, 3)          # device, read-write
        _ck(lib.cuMemSetAccess(va, ctypes.c_size_t(total), ctypes.byref(acc), ctypes.c_size_t(1)), "cuMemSetAccess")
        rows = self.nslots + own_rows
        arr = _Arr(va.value + self.front, (rows,) + self.tail, (self.row,) + self.inner, _TS[self.dtype])
        t = torch.as_tensor(arr, device=self.dev)
        if own_rows:                          # fresh physical memory is not zeroed
            t[self.nslots:].zero_()
        return t

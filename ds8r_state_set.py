import ctypes
import sys
import ctypes.util

D128API = ctypes.WinDLL(r"C:\Windows\System32\D128API.dll")

class CONTROLFLAGS(ctypes.Structure):
    _fields_ = [
        ("Enable",    ctypes.c_int, 2),   # 0x01=disable, 0x02=enable, 0x03=no change
        ("Mode",      ctypes.c_int, 3),   # 0x01=mono, 0x02=biphasic, 0x07=no change
        ("Polarity",  ctypes.c_int, 3),   # 0x01=pos, 0x02=neg, 0x03=alt, 0x07=no change
        ("Source",    ctypes.c_int, 3),   # 0x01=internal, 0x02=external, 0x07=no change
        ("Zero",      ctypes.c_int, 2),
        ("Trigger",   ctypes.c_int, 2),   # 0x01=trigger, 0x03=no change
        ("NoBuzzer",  ctypes.c_int, 2),
        ("Reserved",  ctypes.c_int, 15),
    ]

class D128STATE(ctypes.Structure):
    _fields_ = [
        ("Control",  CONTROLFLAGS),
        ("Demand",   ctypes.c_int),   # mA * 10 (e.g. 100 = 10mA)
        ("Width",    ctypes.c_int),   # µs
        ("Recovery", ctypes.c_int),   # % (10-100)
        ("Dwell",    ctypes.c_int),   # µs (1-99)
        ("CPULSE",   ctypes.c_uint),
        ("COOC",     ctypes.c_uint),
        ("SFlags",   ctypes.c_uint),
        ("_pad",     ctypes.c_uint),
    ]

class DEVHDR(ctypes.Structure):
    _fields_ = [("DeviceCount", ctypes.c_int)]

class D128DEVICESTATE(ctypes.Structure):
    _fields_ = [
        ("DeviceID",   ctypes.c_int),
        ("VersionID",  ctypes.c_int),
        ("Error",      ctypes.c_int),
        ("State",      D128STATE),
    ]

class D128(ctypes.Structure):
    _fields_ = [
        ("Header", DEVHDR),
        ("State",  D128DEVICESTATE * 1),
    ]

# --- Init ---
api_ref   = ctypes.c_int(0)
api_error = ctypes.c_int(0)
D128API.DGD128_Initialise(ctypes.byref(api_ref), ctypes.byref(api_error), None, None)
print(f"Initialised. Ref: {api_ref.value}")

# --- Read current state ---
def get_state():
    cb = ctypes.c_int(ctypes.sizeof(D128))
    buf = D128()
    D128API.DGD128_Update(api_ref, ctypes.byref(api_error), None, 0,
                          ctypes.byref(buf), ctypes.byref(cb), None, None)
    dev = buf.State[0]
    s = dev.State
    print(f"Serial:    {dev.DeviceID}")
    print(f"Demand:    {s.Demand / 10:.1f} mA")
    print(f"Width:     {s.Width} µs")
    print(f"Recovery:  {s.Recovery} %")
    print(f"Dwell:     {s.Dwell} µs")
    print(f"Enable:    {s.Control.Enable}")
    print(f"Mode:      {s.Control.Mode}")
    return buf

def set_demand(mA):
    if mA > 8.0:
        sys.exit("cannot exceed 8.0 mA")
    buf = get_state()
    buf.State[0].State.Demand = int(mA * 10)
    cb_new = ctypes.c_int(ctypes.sizeof(D128))
    cb_cur = ctypes.c_int(0)
    ret = D128API.DGD128_Update(api_ref, ctypes.byref(api_error),
                                ctypes.byref(buf), cb_new,
                                None, ctypes.byref(cb_cur),
                                None, None)
    print(f"Demand set to {mA} mA, ret={ret}, err={api_error.value}")

def set_pulse_width(us):
    buf = get_state()
    buf.State[0].State.Width = us
    cb_new = ctypes.c_int(ctypes.sizeof(D128))
    cb_cur = ctypes.c_int(0)
    ret = D128API.DGD128_Update(api_ref, ctypes.byref(api_error),
                                ctypes.byref(buf), cb_new,
                                None, ctypes.byref(cb_cur),
                                None, None)
    print(f"Width set to {us} µs, ret={ret}, err={api_error.value}")

def trigger():
    buf = get_state()
    buf.State[0].State.Control.Trigger = 0x01
    cb_new = ctypes.c_int(ctypes.sizeof(D128))
    cb_cur = ctypes.c_int(0)
    ret = D128API.DGD128_Update(api_ref, ctypes.byref(api_error),
                                ctypes.byref(buf), cb_new,
                                None, ctypes.byref(cb_cur),
                                None, None)
    print(f"Triggered, ret={ret}, err={api_error.value}")

 
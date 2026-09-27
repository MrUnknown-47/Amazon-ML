import ctypes
import sys

# Test whether PySetObject struct matches CPython 3.14 layout
s = {"apple", "banana", "cherry", "date"}
print("Set size:", len(s))

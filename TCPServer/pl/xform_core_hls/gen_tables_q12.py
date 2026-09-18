import math

# EXP_LUT: 512 entries, Q12
EXP_LUT = []
for i in range(512):
    val = int(round(math.exp(-i / 64.0) * 4096))
    if val > 4095: val = 4095
    EXP_LUT.append(val)

# SIG_LUT: 256 entries, Q12
SIG_LUT = []
for i in range(256):
    x_q12 = (i - 128) * 128  # -16384 to 16256
    x = x_q12 / 4096.0
    s = 1.0 / (1.0 + math.exp(-x))
    val = int(round(s * 4096))
    if val > 4095: val = 4095
    SIG_LUT.append(val)

# Print C arrays
print("// SIG_LUT: 256 entries, Q12")
print("static const int16_t SIG_LUT[256] = {")
for i, v in enumerate(SIG_LUT):
    if i % 16 == 0: print("  ", end="")
    print(f"{v},", end=" ")
    if i % 16 == 15: print()
print("};")
print()

print("// EXP_LUT: 512 entries, Q12")
print("static const int16_t EXP_LUT[512] = {")
for i, v in enumerate(EXP_LUT):
    if i % 16 == 0: print("  ", end="")
    print(f"{v},", end=" ")
    if i % 16 == 15: print()
print("};")
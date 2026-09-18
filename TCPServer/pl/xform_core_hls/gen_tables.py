import math

# Q10 scale = 1024
Q = 1024

# SIG_LUT: 256 entries for sigmoid
# Input range: -512..511 (Q10) -> maps to 0..255
# sigmoid(x) = 1/(1+exp(-x/Q10)) * Q10
SIG_LUT = []
for i in range(256):
    # x in Q10: -512 to 511
    x_q10 = i - 512
    x = x_q10 / 1024.0
    s = 1.0 / (1.0 + math.exp(-x))
    val = int(round(s * 1024))
    if val > 1023: val = 1023
    SIG_LUT.append(val)

# EXP_LUT: 512 entries for exp
# exp(-i/64) * 4096
EXP_LUT = []
for i in range(512):
    val = int(round(math.exp(-i / 64.0) * 4096))
    if val > 4095: val = 4095
    EXP_LUT.append(val)

# Print C arrays
print("// SIG_LUT: 256 entries, Q10")
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
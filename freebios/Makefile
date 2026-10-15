# OpenDS: builds the keyless ARM7 and ARM9 images and regenerates ../src/FreeBIOS.h.
# Needs clang and llvm-objcopy.
CLANG   ?= clang
OBJCOPY ?= llvm-objcopy

all: opends_bios7.bin opends_bios9.bin
	./make-header.sh

opends_bios7.o: opends_bios.S
	$(CLANG) --target=arm-none-eabi -x assembler-with-cpp -DBIOS_ARM7 -mcpu=arm7tdmi -c $< -o $@

opends_bios9.o: opends_bios.S
	$(CLANG) --target=arm-none-eabi -x assembler-with-cpp -DBIOS_ARM9 -mcpu=arm946e-s -c $< -o $@

%.bin: %.o
	$(OBJCOPY) -O binary $< $@

clean:
	rm -f opends_bios7.o opends_bios9.o

.PHONY: all clean

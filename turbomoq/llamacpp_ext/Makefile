CC = clang
CFLAGS = -O2 -Wall -fPIC
LDFLAGS = -lm

UNAME_S := $(shell uname -s)
ifeq ($(UNAME_S),Darwin)
    TARGET = libturbomoq.dylib
    SHARED = -shared -dynamiclib
else
    TARGET = libturbomoq.so
    SHARED = -shared
endif

SRC = turbomoq_kv.c

all: $(TARGET)

$(TARGET): $(SRC) turbomoq_kv.h
	$(CC) $(CFLAGS) $(SHARED) -o $@ $(SRC) $(LDFLAGS)

test: $(TARGET)
	@echo "C extension compiled successfully: $(TARGET)"
	@python3 -c "from turbomoq.llamacpp_ext import LlamaCppExtension, EXTENSION_AVAILABLE; print(f'Extension available: {EXTENSION_AVAILABLE}')"

clean:
	rm -f $(TARGET)

.PHONY: all test clean

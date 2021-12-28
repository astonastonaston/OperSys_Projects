CONFIG_MODULE_SIG=n
obj-m	:= program2.o
KVERSION := $(shell uname -r)
PWD	:= $(shell pwd)

all:
	$(MAKE) -C /lib/modules/$(KVERSION)/build M=$(PWD) modules
clean:
	$(MAKE) -C /lib/modules/$(KVERSION)/build M=$(PWD) clean

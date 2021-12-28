mname := mydev
$(mname)-objs := main.o
obj-m := $(mname).o

KERNELDIR := /lib/modules/`uname -r`/build

all:
# compile kernel module
	$(MAKE) -C $(KERNELDIR) M=`pwd` modules
# insert module
	sudo insmod mydev.ko
# compile test
	gcc -o test test.c

clean:
# clean kernel module
	$(MAKE) -C $(KERNELDIR) M=`pwd` clean
	sudo rmmod mydev
# clean test
	rm test
# show outputs on dmesg
	dmesg | grep OS_AS5



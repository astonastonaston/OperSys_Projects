#include <linux/module.h>
#include <linux/moduleparam.h>
#include <linux/kernel.h>
#include <linux/init.h>
#include <linux/stat.h>
#include <linux/fs.h>
#include <linux/workqueue.h>
#include <linux/sched.h>
#include <linux/interrupt.h>
#include <linux/slab.h>
#include <linux/cdev.h>
#include <linux/delay.h>
#include <asm/uaccess.h>
#include "ioc_hw5.h"
#include <linux/uaccess.h>
#include <linux/string.h>

MODULE_LICENSE("GPL");

#define PREFIX_TITLE "OS_AS5"
#define IRQ_NUM 1
// DMA
#define DMA_BUFSIZE 64
#define DMASTUIDADDR 0x0        // Student ID
#define DMARWOKADDR 0x4         // RW function complete
#define DMAIOCOKADDR 0x8        // ioctl function complete
#define DMAIRQOKADDR 0xc        // ISR function complete
#define DMACOUNTADDR 0x10       // interrupt count function complete
#define DMAANSADDR 0x14         // Computation answer
#define DMAREADABLEADDR 0x18    // READABLE variable for synchronize
#define DMABLOCKADDR 0x1c       // Blocking or non-blocking IO
#define DMAOPCODEADDR 0x20      // data.a opcode
#define DMAOPERANDBADDR 0x21    // data.b operand1
#define DMAOPERANDCADDR 0x25    // data.c operand2
void *dma_buf;
unsigned int dev_major, dev_minor;

#define DEV_BASEMINOR 1
#define DEV_COUNT 1
#define DEV_NAME "my_dev"

// Declaration for file operations
static ssize_t drv_read(struct file *filp, char __user *buffer, size_t, loff_t*);
static int drv_open(struct inode*, struct file*);
static ssize_t drv_write(struct file *filp, const char __user *buffer, size_t, loff_t*);
static int drv_release(struct inode*, struct file*);
static long drv_ioctl(struct file *, unsigned int , unsigned long );
dev_t dev;

// cdev file_operations
static struct file_operations fops = {
      owner: THIS_MODULE,
      read: drv_read,
      write: drv_write,
      unlocked_ioctl: drv_ioctl,
      open: drv_open,
      release: drv_release,
};

struct dataIn {
    char a;
    int b;
    short c;
};

// in and out function
void myoutc(unsigned char data,unsigned short int port);
void myouts(unsigned short data,unsigned short int port);
void myouti(unsigned int data,unsigned short int port);
unsigned char myinc(unsigned short int port);
unsigned short myins(unsigned short int port);
unsigned int myini(unsigned short int port);

// Work routine
static struct work_struct *work_routine;

// cdev declaration
struct cdev *dev_cdev;

// For input data structure
struct DataIn {
    char a;
    int b;
    short c;
} *dataIn;


// Arithmetic funciton
static void drv_arithmetic_routine(struct work_struct* ws);


// Input and output data from/to DMA
// input to DMA
void myoutc(unsigned char data,unsigned short int port) {
    *(volatile unsigned char*)(dma_buf+port) = data;
}
void myouts(unsigned short data,unsigned short int port) {
    *(volatile unsigned short*)(dma_buf+port) = data;
}
void myouti(unsigned int data,unsigned short int port) {
    *(volatile unsigned int*)(dma_buf+port) = data;
}

// output from DMA
unsigned char myinc(unsigned short int port) {
    return *(volatile unsigned char*)(dma_buf+port);
}
unsigned short myins(unsigned short int port) {
    return *(volatile unsigned short*)(dma_buf+port);
}
unsigned int myini(unsigned short int port) {
    return *(volatile unsigned int*)(dma_buf+port);
}


static int drv_open(struct inode* ii, struct file* ff) {
	try_module_get(THIS_MODULE);
    	printk("%s:%s(): device open\n", PREFIX_TITLE, __func__);
	return 0;
}

static int drv_release(struct inode* ii, struct file* ff) {
	module_put(THIS_MODULE);
    	printk("%s:%s(): device close\n", PREFIX_TITLE, __func__);
	return 0;
}

int prime(int base, short nth)
{
    int fnd=0;
    int i, num, isPrime;

    num = base;
    while(fnd != nth) {
        isPrime=1;
        num++;
        for(i=2;i<=num/2;i++) {
            if(num%i == 0) {
                isPrime=0;
                break;
            }
        }
        
        if(isPrime) {
            fnd++;
        }
    }
    return num;
}

static void drv_arithmetic_routine(struct work_struct* ws) {
    int ans, ope1, ope2;
	char optor;

	// fetch fron DMA formula
	ope1 = myini(DMAOPERANDBADDR);
	ope2 = myini(DMAOPERANDCADDR);
	optor = myinc(DMAOPCODEADDR);
	// printk("getting ope1(b): %d, ope2(c): %d, optor: %c\n", ope1, ope2, optor);

	// calculate result
    switch(optor) {
        case '+':
            ans = ope1 + ope2;
            break;
        case '-':
            ans = ope1 - ope2;
            break;
        case '*':
            ans = ope1 * ope2;
            break;
        case '/':
            ans = ope1 / ope2;
            break;
        case 'p':
            ans = prime(ope1, ope2);
            break;
        default:
            ans=0;
			break;
    }

	printk("%s:%s(): %d %c %d = %d\n\n",PREFIX_TITLE, __func__, ope1, optor, ope2, ans);
	// output answer to DMA
	// printk("the answer is %d\n", ans);
	myouti(ans, DMAANSADDR);

	// if non-blocking, set readable flag back to be True
	// printk("in arithmetic, now my block is %d\n", myini(DMABLOCKADDR));
	if (myini(DMABLOCKADDR)==0) myouti(1, DMAREADABLEADDR);
}

static ssize_t drv_read(struct file *filp, char __user *buffer, size_t ss, loff_t* lo) {
	// read answer and store to user space 
	int res = myini(DMAANSADDR);
	printk("%s,%s(): ans = %d\n",PREFIX_TITLE, __func__, res);

	copy_to_user(buffer, &res, ss);

	// clean result in DMA 
	myouti(0, DMAANSADDR);

	// reset readable as false
	myouti(0, DMAREADABLEADDR);
	return 0;
}

static ssize_t drv_write(struct file *filp, const char __user *buffer, size_t ss, loff_t* lo) {	
	// take IO mode from DMA
	int IOMode = myini(DMABLOCKADDR);
	
	// fetch arithmetic data from user space
	struct dataIn data;
 	copy_from_user(&data, (void *) buffer, ss);

	// transfer data into DMA buffer
	myoutc(data.a, DMAOPCODEADDR); // char (operator)
    myouti(data.b, DMAOPERANDBADDR); // int (operand B)
    myouti(data.c, DMAOPERANDCADDR); // short (operand C)
	printk("writing %c to OPCODE, %d to opeb, %d to opec\n", data.a, data.b, data.c);

	// init work to be submitted
	INIT_WORK(work_routine, drv_arithmetic_routine);

	// Decide io mode 
	if(IOMode) {
		// Blocking IO
		printk("%s:%s(): queue work\n",PREFIX_TITLE, __func__);

		// put in working queue
		schedule_work(work_routine);

		printk("%s:%s(): block\n",PREFIX_TITLE, __func__);

		// flush working queue
		flush_scheduled_work();

		// reset readable flag to 1
		myoutc(1, DMAREADABLEADDR);
    } 
	else {
		// Non-blocking IO
		// set unreadable
		myoutc(0, DMAREADABLEADDR);

		printk("%s:%s(): queue work\n",PREFIX_TITLE, __func__);

		// put in working queue
		schedule_work(work_routine);

		
   	}
	return 0;
}

static long drv_ioctl(struct file *filp, unsigned int cmd, unsigned long arg) {
	switch (cmd)
	{
	// switch begin
	case HW5_IOCSETSTUID: {
		int stuid = 0;
		// copy arg from usr space
		copy_from_user(&stuid, (void*)arg, 4);

		// store in DMA buffer
		myouti(stuid, DMASTUIDADDR);

		// print out kernel message
		printk("%s,%s(): My STUID is = %d\n",PREFIX_TITLE, __func__, myini(DMASTUIDADDR));
		return 1;
	}
	case HW5_IOCSETRWOK: {
		int rwok = 0;
		// copy arg from usr space
		copy_from_user(&rwok, (void*)arg, 4);

		// store in DMA buffer
		myouti(rwok, DMARWOKADDR);

		// print out kernel message
		printk("%s,%s(): RW OK\n",PREFIX_TITLE, __func__);
		return 1;		
	}
	case HW5_IOCSETIOCOK: {
		int iocok = 0;
		// copy arg from usr space
		copy_from_user(&iocok, (void*)arg, 4);

		// store in DMA buffer
		myouti(iocok, DMAIOCOKADDR);

		// print out kernel message
		printk("%s,%s(): IOC OK\n",PREFIX_TITLE, __func__);
		return 1;		
	}
	case HW5_IOCSETIRQOK: {
		int irqok = 0;
		// copy arg from usr space
		copy_from_user(&irqok, (void*)arg, 4);

		// store in DMA buffer
		myouti(irqok, DMAIRQOKADDR);

		// print out kernel message
		printk("%s,%s(): IRQ OK\n",PREFIX_TITLE, __func__);
		return 1;
	}
	case HW5_IOCSETBLOCK: {
		int blk = 0;
		// copy arg from usr space
		copy_from_user(&blk, (void*)arg, 4);

		// store in DMA buffer
		myouti(blk, DMABLOCKADDR);

		// print out kernel message
		if (blk==1) printk("%s,%s(): Blocking IO\n",PREFIX_TITLE, __func__);
		else printk("%s,%s(): Non-Blocking IO\n",PREFIX_TITLE, __func__);
		return 1;
	}
	case HW5_IOCWAITREADABLE: { // synchronization function
		printk("%s,%s(): wait readable 1\n",PREFIX_TITLE, __func__);
		
		// busy waiting
		while (myini(DMAREADABLEADDR) != 1) msleep(5);
		//msleep(5000);
		
		// copy flag to usr space
		int rdable;
		rdable = myini(DMAREADABLEADDR);
		copy_to_user((void*)arg, (void*)&rdable, 4);
		return 1;
	}
	default: {
		break;
	}
	// switch end
	}
	return 0;
}

static irqreturn_t isr_for_keyboard(int data, void *dev_id)
{
	int cnt = myini(DMACOUNTADDR);    
	myouti(cnt+1, DMACOUNTADDR); // addition by 1 for once isr execution
	return IRQ_HANDLED;               //interrupt handled successfully
}

static int __init init_modules(void) {
	// register device -> register driver -> register an ISR for the driver (for a given/keyboard IRQ)
	printk("%s:%s():...............Start...............\n", PREFIX_TITLE, __func__);

	/* Register chrdev */ 
    if (alloc_chrdev_region(&dev, DEV_BASEMINOR, DEV_COUNT, DEV_NAME) < 0)
	{
		printk(KERN_ALERT"Register chrdev failed!\n");
		return -1;
	}
	
	/* report error */
	else {
		printk("%s:%s(): register chrdev(%i,%i)\n", PREFIX_TITLE, __func__, MAJOR(dev), MINOR(dev));
	}

	dev_major = MAJOR(dev); 
	dev_minor = MINOR(dev);

	/* Init cdev and make it alive */	
	dev_cdev = cdev_alloc(); 
	dev_cdev->ops = &fops; // same as cdev_init
	dev_cdev->owner = THIS_MODULE;

	if (cdev_add(dev_cdev, dev, 1) < 0) // add dev to cdev
	{
		printk(KERN_ALERT"Add cdev failed!\n");
		return -1;
	}

	/* IRQ handler registration */
	int irqret = request_irq(IRQ_NUM, isr_for_keyboard, IRQF_SHARED, DEV_NAME, &dev);
	printk("%s:%s(): request_irq %d returns %d\n", PREFIX_TITLE, __func__, IRQ_NUM, irqret);

	/* Allocate work routine */
	work_routine = kmalloc(sizeof(typeof(*work_routine)), GFP_KERNEL);
	memset((void*)work_routine, 0, sizeof(typeof(*work_routine)));

	/* Allocate DMA buffer */
	dma_buf = kmalloc(DMA_BUFSIZE, GFP_KERNEL);
	memset(dma_buf, 0, DMA_BUFSIZE);
	printk("%s:%s(): allocate dma buffer\n", PREFIX_TITLE, __func__);

	return 0;
}

static void __exit exit_modules(void) {
	// release isr -> release dev -> release driver
	printk("%s,%s(): interrupt count = %d\n",PREFIX_TITLE, __func__, myini(DMACOUNTADDR));
	free_irq(IRQ_NUM, &dev);	

	/* Free DMA buffer when exit modules */
	printk("%s,%s(): free dma buffer\n",PREFIX_TITLE, __func__);	
	kfree(dma_buf);

	/* Delete character device */
	printk("%s,%s(): unregister chrdev\n",PREFIX_TITLE, __func__);
	unregister_chrdev_region(MKDEV(dev_major, dev_minor), DEV_COUNT);
	cdev_del(dev_cdev);

	/* Free work routine */
	kfree(work_routine);
	printk("%s:%s():..............End..............\n", PREFIX_TITLE, __func__);
}

module_init(init_modules);
module_exit(exit_modules);

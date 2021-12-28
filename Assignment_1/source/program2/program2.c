#include <linux/module.h>
#include <linux/sched.h>
#include <linux/pid.h>
#include <linux/kthread.h>
#include <linux/kernel.h>
#include <linux/err.h>
#include <linux/slab.h>
#include <linux/printk.h>
#include <linux/jiffies.h>
#include <linux/kmod.h>
#include <linux/fs.h>

MODULE_LICENSE("GPL");

// extern _do_fork
// extern do_execve
// extern getname

extern struct wait_opts;

extern int do_execve(struct filename *filename,
	const char __user *const __user *__argv,
	const char __user *const __user *__envp);

extern struct filename * getname(const char __user * filename);

extern long _do_fork(unsigned long clone_flags,
	      unsigned long stack_start,
	      unsigned long stack_size,
	      int __user *parent_tidptr,
	      int __user *child_tidptr,
	      unsigned long tls);

extern long do_wait(struct wait_opts *wo);

struct wait_opts {
	enum pid_type		wo_type;
	int			wo_flags;
	struct pid		*wo_pid;

	struct siginfo __user	*wo_info;
	int __user		*wo_stat;
	struct rusage __user	*wo_rusage;

	wait_queue_t		child_wait;
	int			notask_error;
};

void print_sig(int sig) {
	switch (sig) {
	case 0:
		printk(KERN_INFO "normal exit"); break;
	case 1:
		printk(KERN_INFO "get SIGHUP signal"); break;
	case 2:
		printk(KERN_INFO "get SIGINT signal"); break;
	case 3:
		printk(KERN_INFO "get SIGQUIT signal"); break;
	case 4:
		printk(KERN_INFO "get SIGILL signal"); break;
	case 5:
		printk(KERN_INFO "get SIGTRAP signal"); break;
	case 6:
		printk(KERN_INFO "get SIGABRT signal"); break;
	case 7:
		printk(KERN_INFO "get SIGBUS signal"); break;
	case 8:
		printk(KERN_INFO "get SIGFPE signal"); break;
	case 9:
		printk(KERN_INFO "get SIGKILL signal"); break;
	case 19:
		printk(KERN_INFO "get SIGSTOP signal"); break;
	case 11:
		printk(KERN_INFO "get SIGSEGV signal"); break;
	case 13:
		printk(KERN_INFO "get SIGPIPE signal"); break;
	case 14:
		printk(KERN_INFO "get SIGALRM signal"); break;
	case 15:
		printk(KERN_INFO "get SIGTERM signal"); break;
	case 31:
		printk(KERN_INFO "get SIGSTOP signal"); break;
	default:
		printk(KERN_INFO "can't find corresponding signal"); break;
	}
	return 0;
}


// child process executing file
int my_exec(void){
	int result;
	const char path[] = "/opt/test";
	const char *const argv[] = {path, NULL, NULL};
	const char *const envp[] = {"HOME=/", "PATH=/sbin:/user/sbin:/bin:/usr/bin", NULL};

	printk(KERN_INFO "The child process id is: %d", current->pid);
	printk(KERN_INFO "Child process starts\n");
	struct filename* my_filename = getname(path);

	result = do_execve(my_filename, argv, envp);
	
	// exec success
	if (!result) return 0;

	// exec failed
	do_exit(result);
}

// parent process waiting for execution
int my_wait(pid_t pid){
	int status;
	struct pid* wo_pid = find_get_pid(pid);

	enum pid_type type;
	type = PIDTYPE_PID;

	struct wait_opts wao;
	wao.wo_type = type;
	wao.wo_pid = wo_pid;
	wao.wo_flags = WEXITED | WUNTRACED;
	wao.wo_info = NULL;
	wao.wo_stat = (int __user*) &status;
	wao.wo_rusage = NULL;

	// call do_wait
	long a;
	a = do_wait(&wao);

	// print out child process info
	put_pid(wo_pid);
	print_sig(*wao.wo_stat & 0b11111);
	printk(KERN_INFO "child process terminated");

	// return child end/stop status
	return *wao.wo_stat & 0b11111;
}

//implement fork function
int my_fork(void *argc){
	//set default sigaction for current process
	int i;
	struct k_sigaction *k_action = &current->sighand->action[0];
	for(i=0;i<_NSIG;i++){
		k_action->sa.sa_handler = SIG_DFL;
		k_action->sa.sa_flags = 0;
		k_action->sa.sa_restorer = NULL;
		sigemptyset(&k_action->sa.sa_mask);
		k_action++;
	}
	
	printk(KERN_INFO "The parent process is inited with pid: %d", current->pid); 

	/* execute a test program in child process */	
	pid_t pid = _do_fork(SIGCHLD, (unsigned long)&my_exec, 0, NULL, NULL, 0);	
	
	/* wait until child process terminates */
	int status = my_wait(pid);

	/* child process state judgement */
	// change demo format of stop signal
	if (status == 31) status = 19;
	printk(KERN_INFO "The return signal is: %d", status);

	
	return 0;
}

static int __init program2_init(void){

	printk(KERN_INFO "[program2] : Module_init\n");
	printk(KERN_INFO "module_init create kthread start\n");
	
	//create a kthread
	struct task_struct* task = kthread_create(&my_fork,NULL,"ForkThread");


	//wake up new thread if ok
	if(!IS_ERR(task)){
		printk(KERN_INFO "Kthread starts\n");
		wake_up_process(task);
	}	

	return 0;
}

static void __exit program2_exit(void){
	printk(KERN_INFO "[program2] : Module_exit\n");
}

module_init(program2_init);
module_exit(program2_exit);


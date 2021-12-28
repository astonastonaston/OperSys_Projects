#include <stdio.h>
#include <stdlib.h>
#include <sys/types.h>
#include <wait.h>
#include <unistd.h>

#include <sys/stat.h>
#include <sys/mman.h>
#include <fcntl.h>

// allocate size bytes of shared memory
void* smalloc(size_t size) {
	void* p = mmap(NULL, size, PROT_READ|PROT_WRITE, MAP_SHARED|MAP_ANON, -1, 0);
	if (p == MAP_FAILED) p = NULL;
 	return p;
}

// free size bytes of shared memory
void sfree(void* p, size_t size) {
	munmap(p, size);
}

// print out process tree
void printProcTree(int* arr, int size) {
	printf("Process tree: ");
	for (int i = 0; i <= size-1; i++) {
		printf("%d", arr[i]);
		if (i != size-1) printf("->");
		else printf("\n");  
	}
}

char* getSigMacro(int i) {
	switch (i) {
		case 14: 
			return "SIGALAM";
		case 0:
			return "normal exit";
		case 1:
			return "SIGHUP";
		case 2:
			return "SIGINT";
		case 3:
			return "SIGQUIT";
		case 4:
			return "SIGILL";
		case 5:
			return "SIGTRAP";
		case 6:
			return "SIGABRT";
		case 7:
			return "SIGBUS";
		case 8:
			return "SIGFPE";
		case 9:
			return "SIGKILL";
		case 19:
			return "SIGSTOP";
		case 11:
			return "SIGSEGV";
		case 13:
			return "SIGPIPE";
		case 15:
			return "SIGTERM";
		default:
			return "unknown signal";
	};
}


// print out termination information
void printTermiInfo(int* procArr, int* statArr, int size) {
	for (int i = size-1; i >= 1; i--) {
		printf("Child process %d of parent process %d is terminated by signal %d (%s)\n", procArr[i], procArr[i-1], statArr[i] & 0b11111, getSigMacro(statArr[i] & 0b11111));
	}
	printf("Myfork process (%d) terminated normally\n", getpid());
}

int main(int argc,char *argv[]){
	// init PID array and return state array
	pid_t* shpidarr = (pid_t*)smalloc(argc * sizeof(pid_t));
	int* shstatarr = (int*)smalloc(argc * sizeof(int));

	// init temporary array for forking info use
	pid_t procarr[argc];
	int statarr[argc];

	int tot_child_num = 0, curr_chi = 0, curr_par = 0;
	
	// init my_fork process state and PID
	procarr[tot_child_num] = getpid();
	statarr[tot_child_num] = 0;

	while (tot_child_num != argc - 1) {
		curr_par = tot_child_num;
		curr_chi = tot_child_num + 1;
		procarr[curr_chi] = fork();

		// child process -> forking
		if (procarr[curr_chi] == 0) {
			procarr[curr_chi] = getpid();
			
			// furthest child proc
			if (curr_chi == argc - 1) {
				// Sync to shared array
				for (int i = 0; i <= argc-1; i++) {
					shpidarr[i] = procarr[i];
				}

				// exec furthest program (at argv[curr_chi])
				execve(argv[curr_chi], argv, NULL);
			}

			tot_child_num++;
			continue;
		}

		// parent process -> waiting and executing program
		else {
			waitpid(procarr[curr_chi], &statarr[curr_chi], WUNTRACED);

			// Sync status info to the shared state array 
			shstatarr[curr_chi] = statarr[curr_chi];

			// the first parent process ending
			if (curr_par == 0) break;

			// execute program at argv[curr_par] (as the child process of the process that invoked this process)
			execve(argv[curr_par], argv, NULL);
		}
	}

	// print out process tree and terminating information
	printProcTree(shpidarr, argc);
	printTermiInfo(shpidarr, shstatarr, argc);

	return 0;
}

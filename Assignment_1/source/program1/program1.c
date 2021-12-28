#include <stdlib.h>
#include <stdio.h>
#include <unistd.h>
#include <sys/wait.h>
#include <sys/types.h>
#include <signal.h>


void print_sig(int sig) {
	switch (sig) {
	case 0:
		printf("Normal exit\n"); break;
	case 1:
		printf("Child process get SIGHUP signal\nChild process is hung up\n"); break;
	case 2:
		printf("Child process get SIGINT signal\nChild process is interrupted\n"); break;
	case 3:
		printf("Child process get SIGQUIT signal\nChild process is quitted\n"); break;
	case 4:
		printf("Child process get SIGILL signal\nChild process comes across illegal instruction(s)\n"); break;
	case 5:
		printf("Child process get SIGTRAP signal\nChild process is trapped\n"); break;
	case 6:
		printf("Child process get SIGABRT signal\nChild process is aborted\n"); break;
	case 7:
		printf("Child process get SIGBUS signal\nChild process comes across bus error\n"); break;
	case 8:
		printf("Child process get SIGFPE signal\nChild process comes across floating-point exception\n"); break;
	case 9:
		printf("Child process get SIGKILL signal\nChild process is killed\n"); break;
	case 19:
		printf("Child process get SIGSTOP signal\nChild process is stopped\n"); break;
	case 11:
		printf("Child process get SIGSEGV signal\nChild process comes across segmentation fault\n"); break;
	case 13:
		printf("Child process get SIGPIPE signal\nChild process comes across broken pipe\n"); break;
	case 14:
		printf("Child process get SIGALRM signal\nChild process gets the alarm signal from timer\n"); break;
	case 15:
		printf("Child process get SIGTERM signal\nChild process terminated\n"); break;
	default:
		printf("can't find corresponding signal"); break;
	}
}

int main(int argc, char *argv[]){

	int state;

	/* fork a child process */
	printf("Process start to fork\n");	
	pid_t pid = fork();

	if (pid < 0) {
		printf("Fork error!\n");
	}

	else {
		if (pid == 0) {
			/* execute test program */ 
			printf("I'm the child process, my pid = %d\n", getpid());

			/* argument passing */
			char* passedArg[argc];

			for (int i = 0; i < argc-1; i++) {
				passedArg[i] = argv[i+1];
			}
			passedArg[argc-1] = NULL;

			/* execution */
			printf("Child process starts to execute the test procram:\n");
			execve(passedArg[0], passedArg, NULL);
		}
		else {
			/* wait for child process terminates */
			printf("I'm the parent process, my pid = %d\n", getpid());
			waitpid(pid, &state, WUNTRACED);
			printf("Parent process receives SIGCHLD signal\n");

			/* check child process'  termination status */
			if (WIFEXITED(state)) {
				print_sig(WEXITSTATUS(state));
				printf("Normal termination with EXIT STATUS = %d\n", WEXITSTATUS(state));
			}	
			else if (WIFSIGNALED(state)) {
				print_sig(WTERMSIG(state));
				printf("CHILD EXECUTION FAILED: %d\n", WTERMSIG(state));
			}
			else if (WIFSTOPPED(state)) {
				print_sig(WSTOPSIG(state));
				printf("CHILD PROCESS STOPPED: %d\n", WSTOPSIG(state));
			}		
			else {
				printf("CHILD PROCESS CONTINUED\n");
			}
			exit(0);
		}
	}
	return 0;
}

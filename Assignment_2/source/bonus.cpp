#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include <string.h>
#include <unistd.h>
#include <time.h>
#include <curses.h>
#include <termios.h>
#include <fcntl.h>
// #include <GL/glew.h>
#include <GL/glut.h>
#include <GL/freeglut.h>


// The range of the random length of logs is gened in interval:
// [LOG_LENGTH_OFFSET + 1, LOG_LENGTH_MAX + LOG_LENGTH_OFFSET]
#define LOG_LENGTH_MAX 10 
#define LOG_LENGTH_OFFSET 4 
#define ROW 10
#define COLUMN 50 	
#define NUM_OF_THREADS 11
#define NUM_OF_MUTEXES 9
#define GAME_RATIO 0.9

pthread_mutex_t mutexes[9];
char row_def_brick[11] = {'|', '=', '=', '=', '=', '=', '=', '=', '=', '=', '|'};
bool quit=0, win=0, lose=0;
int LATENCY = 90000;
int thread_nums[11] = {0,1,2,3,4,5,6,7,8,9,10};
int log_num=7;
int log_start_arr[ROW-1];
int row_log_nums[9];
int rivLog_stt_Index=0, rivLog_end_Index=0;
bool mouse_held=false;
int mouse_loc=330;

struct Node{
	int x , y; 
	Node( int _x , int _y ) : x( _x ) , y( _y ) {}; // node constructor defs
	Node(){} ; 
} frog ; 

// map[20][50]
char map[ROW+10][COLUMN] ; 

// Determine a keyboard is hit or not. If yes, return 1. If not, return 0. 
int kbhit(void){
	struct termios oldt, newt;
	int ch;
	int oldf;

	tcgetattr(STDIN_FILENO, &oldt);

	newt = oldt;
	newt.c_lflag &= ~(ICANON | ECHO);

	tcsetattr(STDIN_FILENO, TCSANOW, &newt);
	oldf = fcntl(STDIN_FILENO, F_GETFL, 0);

	fcntl(STDIN_FILENO, F_SETFL, oldf | O_NONBLOCK);

	ch = getchar();

	tcsetattr(STDIN_FILENO, TCSANOW, &oldt);
	fcntl(STDIN_FILENO, F_SETFL, oldf);

	if(ch != EOF)
	{
		ungetc(ch, stdin);
		return 1;
	}
	return 0;
}


// Effective size: index 0-48 
void *logs_move( void * my_id ){

	/*  Move the logs  */
	int log_row = *(int*)my_id;
	log_num = row_log_nums[log_row-1];
	pthread_mutex_t curr_lock = mutexes[log_row-1];
	// int curr_head=14, curr_tail=0;
	int curr_head, curr_tail;
	bool odd_row = log_row % 2;
	char temp;

	// left-running row init log position
	if (odd_row) {
		// curr_tail = (curr_head + log_num - 1) % (COLUMN-1);
		curr_tail = COLUMN-2-((log_start_arr[log_row-1] + log_num - 1) % (COLUMN-1));
		curr_head = (curr_tail + log_num - 1) % (COLUMN-1);
	}
	// right-running row init log position
	else {
		curr_tail = log_start_arr[log_row-1];
		curr_head = (curr_tail + log_num - 1) % (COLUMN-1);
	}

	while (true)
	{
		// speed specified
		usleep(LATENCY);

		// end status
		if (quit || win || lose) break;
		
		// left-running logs
		if (odd_row) {
			pthread_mutex_lock(&curr_lock);
			// log moving
			if (log_row == frog.x) {
				if (frog.y == 0) {
					lose = 1;
					pthread_mutex_unlock(&curr_lock);
					continue;
				}
				frog.y = frog.y - 1;
				map[log_row][frog.y] = '0';
				map[log_row][frog.y+1] = '=';
				if (frog.y+1 == COLUMN-2-curr_head) {
					map[log_row][COLUMN-2-curr_tail] = ' ';
					curr_tail = (curr_tail + 1) % (COLUMN-1);
					curr_head = (curr_head + 1) % (COLUMN-1);
					pthread_mutex_unlock(&curr_lock);
					continue;
				}
			}
			map[log_row][COLUMN-2-curr_tail] = ' ';
			curr_tail = (curr_tail + 1) % (COLUMN-1);
			curr_head = (curr_head + 1) % (COLUMN-1);

			map[log_row][COLUMN-2-curr_head] = '=';
			
			pthread_mutex_unlock(&curr_lock);
			// quit check
		}

		// right-running logs
		else {
			pthread_mutex_lock(&curr_lock);
			// log with frog moving
			if (log_row == frog.x) {
				if (frog.y == COLUMN-2) {
					lose = 1;
					pthread_mutex_unlock(&curr_lock);
					continue;
				}
				frog.y = frog.y + 1;
				map[log_row][frog.y] = '0';
				map[log_row][frog.y-1] = '=';
				if (frog.y-1 == curr_head) {
					map[log_row][curr_tail] = ' ';
					curr_tail = (curr_tail + 1) % (COLUMN-1);
					curr_head = (curr_head + 1) % (COLUMN-1);
					pthread_mutex_unlock(&curr_lock);
					continue;
				}
			}
			map[log_row][curr_tail] = ' ';
			curr_tail = (curr_tail + 1) % (COLUMN-1);
			curr_head = (curr_head + 1) % (COLUMN-1);
			
			map[log_row][curr_head] = '=';
			
			pthread_mutex_unlock(&curr_lock);
			// quit check
			
		}
	}
	/*  Print the map on the screen  */
	pthread_exit(NULL);
}


void pprint_map() {
	//Print the map into screen

	// ocean background
	glClearColor(0.0,0.0,0.9,0.75);
	glClear(GL_COLOR_BUFFER_BIT);

	// banks
	glBegin(GL_POLYGON);
	glColor3f(0.5, 0.5, 0.5); glVertex3f(-1.0, -1.0, 0);
	glColor3f(0.5, 0.5, 0.5); glVertex3f(-1.0, -9.0/11.0, 0);
	glColor3f(0.5, 0.5, 0.5); glVertex3f(1.0*GAME_RATIO, -9.0/11.0, 0);
	glColor3f(0.5, 0.5, 0.5); glVertex3f(1.0*GAME_RATIO, -1.0, 0);
	glEnd();
	
	glBegin(GL_POLYGON);
	glColor3f(0.5, 0.5, 0.5); glVertex3f(-1.0, 1.0, 0);
	glColor3f(0.5, 0.5, 0.5); glVertex3f(-1.0, 9.0/11.0, 0);
	glColor3f(0.5, 0.5, 0.5); glVertex3f(1.0*GAME_RATIO, 9.0/11.0, 0);
	glColor3f(0.5, 0.5, 0.5); glVertex3f(1.0*GAME_RATIO, 1, 0);
	glEnd();

	// lock before printing
	for (int i = 0; i < NUM_OF_MUTEXES; i++) {
		pthread_mutex_lock(&mutexes[i]);
	}
	
	// print out logs
	for (int j = 1; j <= ROW-1; j++) {

		// normal log case
		if (map[j][COLUMN-2] == ' ' || map[j][0] == ' ') {
			// ocean painted blue
			// glBegin(GL_POLYGON);
			// glColor3f(0.0, 0.0, 0.9); glVertex3f(-1.0, 1-j*2.0/11.0, 0);
			// glColor3f(0.0, 0.0, 0.9); glVertex3f(-1.0, 1-(j+1)*2.0/11.0, 0);
			// glColor3f(0.0, 0.0, 0.9); glVertex3f(1.0, 1-(j+1)*2.0/11.0, 0);
			// glColor3f(0.0, 0.0, 0.9); glVertex3f(1.0, 1-j*2.0/11.0, 0);
			// glEnd();
			// find log starting and ending index
			for (int k = 0; k < COLUMN-1; k++) {
				// if (false) {
				if (lose || win || quit) {
					if ((map[j][k]) == '=') {
						rivLog_stt_Index = k;
						break;
					}
				}
				else {
					if ((map[j][k]) == '=' || (map[j][k]) == '0') {
						rivLog_stt_Index = k;
						break;
					}
				}
			}
			
			for (int k = COLUMN-2; k >= 0; k--) {
				if (lose||win||quit) {
				// if (false) {
					if ((map[j][k]) == '=') {
						// if (map[j][k+1] == map[j][k-1] && map[j][k-1] == ' ') continue;
						rivLog_end_Index = k;
						break;
					}
				}
				else {
					if ((map[j][k]) == '=' || (map[j][k]) == '0') {
						rivLog_end_Index = k;
						break;
					}
				}
			}
			// log painted brown
			glBegin(GL_POLYGON);
			glColor3f(1, 0.7, 0.3); glVertex3f(-1.0+(rivLog_stt_Index*(2.0/49.0))*GAME_RATIO, 1-j*2.0/11.0, 0);
			glColor3f(1, 0.7, 0.3); glVertex3f(-1.0+(rivLog_stt_Index*(2.0/49.0))*GAME_RATIO, 1-(j+1)*2.0/11.0, 0);
			glColor3f(1, 0.7, 0.3); glVertex3f(-1.0+((rivLog_end_Index+1)*(2.0/49.0))*GAME_RATIO, 1-(j+1)*2.0/11.0, 0);
			glColor3f(1, 0.7, 0.3); glVertex3f(-1.0+((rivLog_end_Index+1)*(2.0/49.0))*GAME_RATIO, 1-j*2.0/11.0, 0);
			glEnd();
		}	
		
		// passing through edge case
		else {
			// log painted brown
			glBegin(GL_POLYGON);
			glColor3f(1, 0.7, 0.3); glVertex3f(-1.0, 1-j*2.0/11.0, 0);
			glColor3f(1, 0.7, 0.3); glVertex3f(-1.0, 1-(j+1)*2.0/11.0, 0);
			glColor3f(1, 0.7, 0.3); glVertex3f(1.0*GAME_RATIO, 1-(j+1)*2.0/11.0, 0);
			glColor3f(1, 0.7, 0.3); glVertex3f(1.0*GAME_RATIO, 1-j*2.0/11.0, 0);
			glEnd();

			// find ocean starting and ending index
			for (int k = 0; k < COLUMN-1; k++) {
				// if (false) {
				if ((win||lose||quit) && (frog.y!=0)) {
					if ((map[j][k] == ' ') || (map[j][k] == '0')) {
						rivLog_stt_Index = k;
						break;
					}
				}
				else {
					if ((map[j][k]) == ' ') {
						rivLog_stt_Index = k;
						break;
					}
				}
			}
			
			for (int k = COLUMN-2; k >= 0; k--) {
				// if (false) {
				if ((win||lose||quit) && (frog.y!=COLUMN-2)) {
					if ((map[j][k] == ' ') || (map[j][k] == '0')) {
						rivLog_end_Index = k;
						break;
					}
				}
				else {
					if ((map[j][k]) == ' ') {
						rivLog_end_Index = k;
						break;
					}
				}
			}
			
			// ocean painted blue
			glBegin(GL_POLYGON);
			glColor3f(0.0, 0.0, 0.9); glVertex3f(-1.0+(rivLog_stt_Index*(2.0/49.0))*GAME_RATIO, 1-j*2.0/11.0, 0);
			glColor3f(0.0, 0.0, 0.9); glVertex3f(-1.0+(rivLog_stt_Index*(2.0/49.0))*GAME_RATIO, 1-(j+1)*2.0/11.0, 0);
			glColor3f(0.0, 0.0, 0.9); glVertex3f(-1.0+((rivLog_end_Index+1)*(2.0/49.0))*GAME_RATIO, 1-(j+1)*2.0/11.0, 0);
			glColor3f(0.0, 0.0, 0.9); glVertex3f(-1.0+((rivLog_end_Index+1)*(2.0/49.0))*GAME_RATIO, 1-j*2.0/11.0, 0);
			glEnd();
		}
	}

	// print out frog
	glBegin(GL_POLYGON);
	glColor3f(0.0, 0.9, 0.0); glVertex3f(-(48.0/49.0)+(frog.y*(2.0/49.0))*GAME_RATIO, 1-frog.x*2.0/11.0, 0);
	glColor3f(0.0, 0.9, 0.0); glVertex3f(-1.0+(frog.y*(2.0/49.0))*GAME_RATIO, 1-(frog.x+1)*2.0/11.0, 0);
	glColor3f(0.0, 0.9, 0.0); glVertex3f(-1.0+((frog.y+1)*(2.0/49.0))*GAME_RATIO, 1-(frog.x+1)*2.0/11.0, 0);
	glEnd();	

	// print out the bar
	// separate zone
	glBegin(GL_POLYGON);
	glColor3f(0.9, 0.9, 0.9); glVertex3f(1.0-2*(1-GAME_RATIO), 1.0, 0);
	glColor3f(0.9, 0.9, 0.9); glVertex3f(1.0-2*(1-GAME_RATIO), -1.0, 0);
	glColor3f(0.9, 0.9, 0.9); glVertex3f(1.0, -1.0, 0);
	glColor3f(0.9, 0.9, 0.9); glVertex3f(1.0, 1.0, 0);
	glEnd();	

	// print words
	glColor3f(1.0, 0.0, 0.0); 
  	glRasterPos2f(0.83, 0.7);
  	glutBitmapString(GLUT_BITMAP_TIMES_ROMAN_24, (const unsigned char*)"Speed Bar");

	// print bar
	glBegin(GL_LINE_LOOP);
	glColor3f(0.1, 0.1, 0.1); 
	glVertex3f(GAME_RATIO*0.97, 0.5, 0);
	glVertex3f(GAME_RATIO*0.97, -0.5, 0);
	glVertex3f(GAME_RATIO*1.03, -0.5, 0);
	glVertex3f(GAME_RATIO*1.03, 0.5, 0);
	glEnd();	

	// print updated bar
	glBegin(GL_POLYGON);
	glColor3f(0.0, 0.0, 0.0); glVertex3f(GAME_RATIO*0.97,-0.5+(330.0-mouse_loc)/220.0, 0);
	glColor3f(0.0, 0.0, 0.0); glVertex3f(GAME_RATIO*0.97, -0.5, 0);
	glColor3f(0.0, 0.0, 0.0); glVertex3f(GAME_RATIO*1.03, -0.5, 0);
	glColor3f(0.0, 0.0, 0.0); glVertex3f(GAME_RATIO*1.03,-0.5+(330.0-mouse_loc)/220.0, 0);
	glEnd();	
	glutSwapBuffers();


	// terminating status
	if (win) {
		glColor3f(1.0, 0.0, 0.0); 
  		glRasterPos2f(-0.2, 0.0);
  		glutBitmapString(GLUT_BITMAP_TIMES_ROMAN_24, (const unsigned char*)"You win!");
	}
	else if (lose)
	{
		glColor3f(1.0, 0.0, 0.0); 
  		glRasterPos2f(-0.2, 0.0);
  		glutBitmapString(GLUT_BITMAP_TIMES_ROMAN_24, (const unsigned char*)"You lose!");
	}
	else if (quit)
	{
		glColor3f(1.0, 0.0, 0.0); 
  		glRasterPos2f(-0.2, 0.0);
  		glutBitmapString(GLUT_BITMAP_TIMES_ROMAN_24, (const unsigned char*)"You exit!");
	}

	// flush updates
    glutSwapBuffers();
	
	// unlock after printing
	for (int i = 0; i < NUM_OF_MUTEXES; i++) {
		pthread_mutex_unlock(&mutexes[i]);
	}
    
	usleep(LATENCY);
	glutPostRedisplay();
}


void *frog_move(void *my_id) {
	/*  Check keyboard hits, to change frog's position or quit the game. */
	char det;
	while (true){
		if (quit || win || lose) break;	
        if( kbhit() && !lose && !win && !quit){
            char dir = getchar() ;
            if( dir == 'w' || dir == 'W' ) {
				// determine win/lose state
				det = map[frog.x-1][frog.y];
				
				// move first
				if (frog.x != 1) pthread_mutex_lock(&mutexes[frog.x-2]);
				map[frog.x-1][frog.y] = '0';
				map[frog.x][frog.y] = row_def_brick[frog.x];
				if (frog.x != 1) pthread_mutex_unlock(&mutexes[frog.x-2]);
				frog.x = frog.x - 1;

				// river falling case
				if (det == ' ') { 
					lose = true;
					break;
				}

				// going to the opposite bank
				if (det == '|') {
					win = true;
					break;
				}
			}
            if( dir == 'a' || dir == 'A' ) {
				if (frog.x == ROW && frog.y == 0) continue;
				else if (frog.y == 0 || map[frog.x][frog.y-1] == ' ') { // edge crashing/ river falling case
					if (map[frog.x][frog.y-1] == ' ') {
						pthread_mutex_lock(&mutexes[frog.x-1]);
						map[frog.x][frog.y-1] = '0';
						map[frog.x][frog.y] = row_def_brick[frog.x];
						frog.y = frog.y - 1;
						pthread_mutex_unlock(&mutexes[frog.x-1]);
					}
					lose = true;
					break;
				}
				else { // moving case
					pthread_mutex_lock(&mutexes[frog.x-1]);
					map[frog.x][frog.y-1] = '0';
					map[frog.x][frog.y] = row_def_brick[frog.x];
					pthread_mutex_unlock(&mutexes[frog.x-1]);
					frog.y = frog.y - 1;
				}
			}
            if( dir == 'd' || dir == 'D' ) {
				if (frog.x == ROW && frog.y == COLUMN-2) continue;
				else if (frog.y == COLUMN-2 || map[frog.x][frog.y+1] == ' ') { // edge crashing/ river falling case
					pthread_mutex_lock(&mutexes[frog.x-1]);
					map[frog.x][frog.y+1] = '0';
					map[frog.x][frog.y] = row_def_brick[frog.x];
					frog.y = frog.y + 1;
					pthread_mutex_unlock(&mutexes[frog.x-1]);
					lose = true;
					break;
				}
				else { // moving case
					pthread_mutex_lock(&mutexes[frog.x-1]);
					map[frog.x][frog.y+1] = '0';
					map[frog.x][frog.y] = row_def_brick[frog.x];
					pthread_mutex_unlock(&mutexes[frog.x-1]);
					frog.y = frog.y + 1;
				}
			}                     
            if( dir == 's' || dir == 'S' ) {
				if (frog.x == ROW) continue;
				else if (map[frog.x+1][frog.y] == ' ') { // edge crashing or river falling case
					if (frog.x != ROW-1) pthread_mutex_lock(&mutexes[frog.x]);
					map[frog.x+1][frog.y] = '0';
					map[frog.x][frog.y] = row_def_brick[frog.x];
					if (frog.x != ROW-1) pthread_mutex_unlock(&mutexes[frog.x]);
					frog.x = frog.x + 1;
					lose = true;
					break;
				}
				else { // moving case
					if (frog.x != ROW-1) pthread_mutex_lock(&mutexes[frog.x]);
					map[frog.x+1][frog.y] = '0';
					map[frog.x][frog.y] = row_def_brick[frog.x];
					if (frog.x != ROW-1) pthread_mutex_unlock(&mutexes[frog.x]);
					frog.x = frog.x + 1;
				}
			}
            if( dir == 'q' || dir == 'Q' ){
                quit = true;
            }
        }

    }
	pthread_exit(NULL);
}


void click_bar(int button, int state, int x, int y) {
	// ensure left mouse clicked within the bar
	mouse_held = (button == GLUT_LEFT_BUTTON && state == GLUT_DOWN && x >= 1380 && x <= 1415 && y >= 110 && y <= 330);
	if (mouse_held) {
		mouse_loc = y;
		LATENCY = (y-30)*300;
	}
}


void drag_bar(int x, int y) {
	// free movement, but ensuring left-clicked
	if (mouse_held) {
		if (y >= 330) {mouse_loc = 330; LATENCY=250*460;}
		else if (y <= 110) {mouse_loc = 110; LATENCY=30*460;}
		else {mouse_loc = y; LATENCY = (y-30)*300;} // 
	}
}


void print_map() {
	// init GUI
	int argc = 0;
	char **argv = 0;
	glutInit(&argc, argv);
	glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGB);

	glutInitWindowPosition(80, 80);
	glutInitWindowSize(1470, 440); // (49*30) * (11*40)
	glClearColor(0.0,0.0,0.7,1.0);

	glutCreateWindow("Frog Crossing River");
	glutMouseFunc(click_bar);
	glutMotionFunc(drag_bar);
	glutDisplayFunc(pprint_map);

	glutMainLoop();
}

int main( int argc, char *argv[] ){

	// init random int
	srand (time(NULL));

	// Initialize the river map and frog's starting position
	memset( map , 0, sizeof( map ) ) ;
	pthread_t threads[NUM_OF_THREADS];
	int i , j ; 

	// init log_num array
	for (int i = 0; i <= 8; i++) row_log_nums[i] = (rand() % LOG_LENGTH_MAX) + 1 + LOG_LENGTH_OFFSET; 

	// init mutex locks
	for (int i = 0; i <= 8; i++) pthread_mutex_init(&mutexes[i], NULL); 

	// init starting log 
	for (int i = 0; i < ROW-1; i++) log_start_arr[i] = rand() % (COLUMN-1);

	// init river
	int log_stt;
	for( i = 1; i < ROW; ++i ){	// rows in [1,9]
		for( j = 0; j < COLUMN - 1; ++j ) // cols in [0,48]	
			map[i][j] = ' ' ;  
		// init log (using the index of the leftmost log)
		log_stt = log_start_arr[i-1];
		for (int j = 0; j < row_log_nums[i-1]; j++) map[i][(log_stt+j)%(COLUMN-1)] = '=';
	}	

	// init lower bank
	for( j = 0; j < COLUMN - 1; ++j )	
		map[ROW][j] = map[0][j] = '|' ;

	// init higher bamk
	for( j = 0; j < COLUMN - 1; ++j )	
		map[0][j] = map[0][j] = '|' ;

	// init frog and its position
	frog = Node( ROW, (COLUMN-1) / 2 ) ; 
	map[frog.x][frog.y] = '0' ; 

	// create pthreads for wood move and frog control.
	for (int i = 1; i <= NUM_OF_THREADS-2; i++) {
		pthread_create(&threads[i], NULL, logs_move,   (void *)&thread_nums[i]);
	}
	pthread_create(&threads[NUM_OF_THREADS-1], NULL, frog_move,   (void *)&thread_nums[NUM_OF_THREADS-1]);

	// GUI printing thread
	usleep(LATENCY/2);
	pthread_create(&threads[0], NULL, (void*(*)(void*))print_map, NULL);

	// thread joins
	for (int i = 0; i <= NUM_OF_THREADS-1; i++) {
		pthread_join(threads[i], NULL);
	}

	// lock destroy
	for (int i = 0; i <= NUM_OF_MUTEXES-1; i++) pthread_mutex_destroy(&mutexes[i]);

	pthread_exit(NULL);
	return 0;
}

main: main.cu file_system.cu file_system.h user_program.cu 
	nvcc --relocatable-device-code=true main.cu file_system.cu user_program.cu -o main.out
clean:
	rm -rf main.out

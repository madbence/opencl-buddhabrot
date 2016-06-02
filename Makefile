all: buddhabrot-cpu buddhabrot-gpu mandelbrot

buddhabrot-cpu: main.c
	g++ -DCPU -lOpenCL -lGLEW -lglfw -lGL main.c -o buddhabrot-cpu

buddhabrot-gpu: main.c
	g++ -DGPU -lOpenCL -lGLEW -lglfw -lGL main.c -o buddhabrot-gpu

mandelbrot: main.c
	g++ -DMANDELBROT -lOpenCL -lGLEW -lglfw -lGL main.c -o mandelbrot

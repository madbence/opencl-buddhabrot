#include <stdlib.h>
#define CL_USE_DEPRECATED_OPENCL_1_2_APIS
#include <CL/cl.h>
#include <stdio.h>
#define GLEW_STATIC
#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <unistd.h>

#include <complex>

float img_[512 * 512 * 3];
unsigned int img[512 * 512 * 3];
unsigned int max_r = 0;
unsigned int max_g = 0;
unsigned int max_b = 0;
cl_float2 output_results_data[512 * 2048];

float mandelbrot(std::complex<float> c, int limit) {
  std::complex<float> z = 0;
  int i = 0;
  for (; i < limit; i++) {
    z = z * z + c;
    if (std::abs(z) > 3) {
      return i * 1. / limit;
    }
  }
  return 0;
}

int buddhabrot(std::complex<float> c, std::complex<float> *t, int limit) {
  std::complex<float> z = 0;
  int i = 0;
  for (; i < limit; i++) {
    if (std::abs(z) > 3) {
      return i;
    }
    z = z * z + c;
    t[i] = z;
  }
  return limit;
}

int main() {
  cl_platform_id platform;
  clGetPlatformIDs(1, &platform, NULL);

  cl_device_id device;
  clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, NULL);

  cl_context context = clCreateContext(NULL, 1, &device, NULL, NULL, NULL);

  cl_command_queue queue = clCreateCommandQueue(context, device, 0, NULL);

  char buf[1000];

  clGetDeviceInfo(device, CL_DEVICE_NAME, 1000, buf, NULL);
  printf("Device Name: %s\n", buf);
  clGetDeviceInfo(device, CL_DEVICE_VERSION, 1000, buf, NULL);
  printf("Device Version: %s\n", buf);
  clGetDeviceInfo(device, CL_DRIVER_VERSION, 1000, buf, NULL);
  printf("Driver Version: %s\n", buf);
  clGetDeviceInfo(device, CL_DEVICE_OPENCL_C_VERSION, 1000, buf, NULL);
  printf("Supported C Version: %s\n", buf);
  cl_uint units;
  clGetDeviceInfo(device, CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(units), &units, NULL);
  printf("Computing Units: %d\n", units);

  srand(time(NULL));
  glfwInit();
  glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
  glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 2);
  glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
  glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);

  glfwWindowHint(GLFW_RESIZABLE, GL_FALSE);

  GLFWwindow* window = glfwCreateWindow(512, 512, "GPGPU", NULL, NULL);
  glfwMakeContextCurrent(window);
  glewExperimental = GL_TRUE;
  glewInit();

  float vertices[] = {
    -1, -1, 0, 0,
    -1,  1, 0, 1,
     1,  1, 1, 1,
    -1, -1, 0, 0,
     1,  1, 1, 1,
     1, -1, 1, 0,
  };

  GLuint vao;
  glGenVertexArrays(1, &vao);
  glBindVertexArray(vao);
  GLuint vbo;
  glGenBuffers(1, &vbo);
  glBindBuffer(GL_ARRAY_BUFFER, vbo);
  glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STATIC_DRAW);
  GLuint tex;
  glGenTextures(1, &tex);
  glBindTexture(GL_TEXTURE_2D, tex);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
  // glGenerateMipmap(GL_TEXTURE_2D);

  const char* vertex_source = R"(
    #version 150
    in vec2 position;
    in vec2 tex;

    out vec2 Tex;

    void main()
    {
      gl_Position = vec4(position, 0.0, 1.0);
      Tex = tex;
    }
  )";

  const char* fragment_source = R"(
    #version 150
    in vec2 Tex;
    out vec4 color;

    uniform sampler2D tex;

    void main()
    {
      color = texture(tex, Tex);
    }
  )";

  const char* kernel_source = R"(
    inline float2 mul(float2 a, float2 b) {
      return ((float2)(a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x));
    }

    inline float cabs(float2 a) {
      return sqrt(a.x * a.x + a.y * a.y);
    }

    __kernel void buddhabrot(__global float2* c, int limit, __global int* steps, __global float2* results) {
      int id = get_global_id(0);
      float2 z = ((float2)(0, 0));
      int i = 0;
      for (; i < limit; i++) {
        if (cabs(z) > 3) {
          steps[id] = i;
          break;
        }
        z = mul(z, z) + c[id];
        results[id * limit + i] = z;
      }
      steps[id] = i;
    }
  )";

  cl_program program = clCreateProgramWithSource(context, 1, &kernel_source, NULL, NULL);
  int err = clBuildProgram(program, 0, NULL, NULL, NULL, NULL);
  if (err != CL_SUCCESS) {
    char buffer[20000];
    printf("Error: Failed to build OpenCL program! (%d)\n", err);
    clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, sizeof(buffer), buffer, NULL);
    printf("%s\n", buffer);
    exit(1);
  }
  cl_kernel kernel = clCreateKernel(program, "buddhabrot", &err);
  if (err != CL_SUCCESS) {
    printf("Error: Cannot create compute kernel!\n");
    exit(1);
  }

  cl_mem input_c = clCreateBuffer(context,  CL_MEM_READ_ONLY,  sizeof(cl_float2) * 512, NULL, NULL);
  cl_mem output_steps = clCreateBuffer(context, CL_MEM_WRITE_ONLY, sizeof(int) * 512, NULL, NULL);
  cl_mem output_results = clCreateBuffer(context, CL_MEM_WRITE_ONLY, sizeof(cl_float2) * 512 * 2048, NULL, NULL);

  clSetKernelArg(kernel, 0, sizeof(cl_mem), &input_c);
  clSetKernelArg(kernel, 2, sizeof(cl_mem), &output_steps);
  clSetKernelArg(kernel, 3, sizeof(cl_mem), &output_results);

  GLuint vertex_shader = glCreateShader(GL_VERTEX_SHADER);
  glShaderSource(vertex_shader, 1, &vertex_source, NULL);
  glCompileShader(vertex_shader);
  GLint status;
  glGetShaderiv(vertex_shader, GL_COMPILE_STATUS, &status);
  if (status != GL_TRUE) {
    char buffer[512];
    glGetShaderInfoLog(vertex_shader, 512, NULL, buffer);
    printf("Cannot compile vertex shader: %s", buffer);
    return 1;
  }
  printf("Vertex shader compiled...\n");
  GLuint fragment_shader = glCreateShader(GL_FRAGMENT_SHADER);
  glShaderSource(fragment_shader, 1, &fragment_source, NULL);
  glCompileShader(fragment_shader);
  glGetShaderiv(fragment_shader, GL_COMPILE_STATUS, &status);
  if (status != GL_TRUE) {
    char buffer[512];
    glGetShaderInfoLog(fragment_shader, 512, NULL, buffer);
    printf("Cannot compile vertex shader: %s", buffer);
    return 1;
  }
  printf("Fragment shader compiled...\n");

  GLuint shader = glCreateProgram();
  glAttachShader(shader, vertex_shader);
  glAttachShader(shader, fragment_shader);
  glBindFragDataLocation(shader, 0, "color");
  glLinkProgram(shader);
  glUseProgram(shader);

  GLint pos_attr = glGetAttribLocation(shader, "position");
  GLint tex_attr = glGetAttribLocation(shader, "tex");
  glEnableVertexAttribArray(pos_attr);
  glVertexAttribPointer(pos_attr, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(float), 0);
  glEnableVertexAttribArray(tex_attr);
  glVertexAttribPointer(tex_attr, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(float), (void*)(2 * sizeof(float)));

  int i = 0;
  while(!glfwWindowShouldClose(window))
  {
    if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS)
      glfwSetWindowShouldClose(window, GL_TRUE);

#ifdef MANDELBROT
    for (int j = 0; j < 512; j++) {
      std::complex<float> c = std::complex<float>(j / 512. * 3 - 2.25, i / 512. * 3 - 1.5);
      float f = mandelbrot(c, 256);
      int color = f * 255;
      img[(i * 512 + j) * 3 + 0] = img[(i * 512 + j) * 3 + 1] = img[(i * 512 + j) * 3 + 2] = color;
    }
#endif

#ifdef CPU
    for (int j = 0; j < 512; j++) {
      const int sr = 512;
      const int sg = 1024;
      const int sb = 2048;
      std::complex<float> r[sr];
      std::complex<float> g[sg];
      std::complex<float> b[sb];
      float ri = i + (rand() * 1. / RAND_MAX - 0.5); // rand() % 512;
      float rj = j + (rand() * 1. / RAND_MAX - 0.5); // rand() % 512;
      std::complex<float> c = std::complex<float>(rj / 512. * 3 - 2.25, ri / 512. * 3 - 1.5);
      int cr = buddhabrot(c, r, sr);
      int cg = buddhabrot(c, g, sg);
      int cb = buddhabrot(c, b, sb);
      for (int k = 0; k < 2048; k++) {
        if (cr < sr && k < cr) {
          int x = (r[k].real() + 2.25) / 3 * 512;
          int y = (r[k].imag() + 1.5) / 3 * 512;
          if (x > 0 && x < 512 && y > 0 && y < 512) {
            int c = ++img[((512 - x) * 512 + y) * 3 + 0];
            if (max_r < c) max_r = c;
          }
        }
        if (cg < sg && k < cg) {
          int x = (g[k].real() + 2.25) / 3 * 512;
          int y = (g[k].imag() + 1.5) / 3 * 512;
          if (x > 0 && x < 512 && y > 0 && y < 512) {
            int c = ++img[((512 - x) * 512 + y) * 3 + 1];
            if (max_g < c) max_g = c;
          }
        }
        if (cb < sb && k < cb) {
          int x = (b[k].real() + 2.25) / 3 * 512;
          int y = (b[k].imag() + 1.5) / 3 * 512;
          if (x > 0 && x < 512 && y > 0 && y < 512) {
            int c = ++img[((512 - x) * 512 + y) * 3 + 2];
            if (max_b < c) max_b = c;
          }
        }
      }
    }
#endif

#ifdef GPU
    cl_float2 points[2048];
    int output_steps_data[2048];
    int limit;
    const size_t limit_r = 512;
    const size_t limit_g = 1024;
    const size_t limit_b = 2048;
    for (int j = 0; j < 512; j++) {
      float ri = i + (rand() * 1. / RAND_MAX - 0.5); // rand() % 512;
      float rj = j + (rand() * 1. / RAND_MAX - 0.5); // rand() % 512;
      points[j] = {rj / 512.f * 3 - 2.25f, ri / 512.f * 3 - 1.5f};
    }

    clEnqueueWriteBuffer(queue, input_c, CL_FALSE, 0, sizeof(cl_float2) * 512, points, 0, NULL, NULL);

    limit = 512;
    clSetKernelArg(kernel, 1, sizeof(int), &limit);
    clEnqueueNDRangeKernel(queue, kernel, 1, NULL, &limit_r, NULL, 0, NULL, NULL);
    clEnqueueReadBuffer(queue, output_steps, CL_FALSE, 0, sizeof(int) * 512, output_steps_data, 0, NULL, NULL);
    clEnqueueReadBuffer(queue, output_results, CL_FALSE, 0, sizeof(cl_float2) * limit * 512, output_results_data, 0, NULL, NULL);
    clFinish(queue);
    for (int j = 0; j < 512; j++) {
      int steps = output_steps_data[j];
      if (steps >= limit) continue;
      for (int k = 0; k < steps; k++) {
        int x = (output_results_data[j * limit + k].x + 2.25) / 3 * 512;
        int y = (output_results_data[j * limit + k].y + 1.5) / 3 * 512;
        if (x > 0 && x < 512 && y > 0 && y < 512) {
          int c = ++img[((512 - x) * 512 + y) * 3 + 0];
          if (max_r < c) max_r = c;
        }
      }
    }

    limit = 1024;
    clSetKernelArg(kernel, 1, sizeof(int), &limit);
    clEnqueueNDRangeKernel(queue, kernel, 1, NULL, &limit_g, NULL, 0, NULL, NULL);
    clEnqueueReadBuffer(queue, output_steps, CL_FALSE, 0, sizeof(int) * 512, output_steps_data, 0, NULL, NULL);
    clEnqueueReadBuffer(queue, output_results, CL_FALSE, 0, sizeof(cl_float2) * limit * 512, output_results_data, 0, NULL, NULL);
    clFinish(queue);
    for (int j = 0; j < 512; j++) {
      int steps = output_steps_data[j];
      if (steps >= limit) continue;
      for (int k = 0; k < steps; k++) {
        int x = (output_results_data[j * limit + k].x + 2.25) / 3 * 512;
        int y = (output_results_data[j * limit + k].y + 1.5) / 3 * 512;
        if (x > 0 && x < 512 && y > 0 && y < 512) {
          int c = ++img[((512 - x) * 512 + y) * 3 + 1];
          if (max_g < c) max_g = c;
        }
      }
    }

    limit = 2048;
    clSetKernelArg(kernel, 1, sizeof(int), &limit);
    clEnqueueNDRangeKernel(queue, kernel, 1, NULL, &limit_b, NULL, 0, NULL, NULL);
    clEnqueueReadBuffer(queue, output_steps, CL_FALSE, 0, sizeof(int) * 512, output_steps_data, 0, NULL, NULL);
    clEnqueueReadBuffer(queue, output_results, CL_FALSE, 0, sizeof(cl_float2) * limit * 512, output_results_data, 0, NULL, NULL);
    clFinish(queue);
    for (int j = 0; j < 512; j++) {
      int steps = output_steps_data[j];
      if (steps >= limit) continue;
      for (int k = 0; k < steps; k++) {
        int x = (output_results_data[j * limit + k].x + 2.25) / 3 * 512;
        int y = (output_results_data[j * limit + k].y + 1.5) / 3 * 512;
        if (x > 0 && x < 512 && y > 0 && y < 512) {
          int c = ++img[((512 - x) * 512 + y) * 3 + 2];
          if (max_b < c) max_b = c;
        }
      }
    }
#endif

    i = (i + 1) % 512;
    for (int j = 0; j < 512 * 512 * 3; j++) {
      float m;
      switch (j % 3) {
        case 0: m = max_r; break;
        case 1: m = max_g; break;
        case 2: m = max_b; break;
      }
      img_[j] = img[j] / m;
      if (j / 3 % 512 == i) {
        img_[j] = 0.5;
      }
    }
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, 512, 512, 0, GL_RGB, GL_FLOAT, img_);
    glDrawArrays(GL_TRIANGLES, 0, 6);
    glfwSwapBuffers(window);
    glfwPollEvents();
  }

  glfwTerminate();

  return 0;
}

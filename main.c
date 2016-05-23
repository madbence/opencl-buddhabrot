#include <stdlib.h>
#include <CL/cl.h>
#include <stdio.h>
#define GLEW_STATIC
#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <unistd.h>

#include <complex>

unsigned char img[512 * 512 * 3];

float mandelbrot(std::complex<float> c, int limit) {
  std::complex<float> z = 0;
  int i = 0;
  for (; i < limit; i++) {
    z = z * z + c;
    if (std::abs(z) > 2) {
      return i * 1. / limit;
    }
  }
  return 0;
}

int buddhabrot(std::complex<float> c, std::complex<float> *t, int limit) {
  std::complex<float> z = 0;
  int i = 0;
  for (; i < limit; i++) {
    z = z * z + c;
    t[i] = z;
    if (std::abs(z) > 2) {
      return i + 1;
    }
  }
  return limit;
}

int main() {
  cl_platform_id platform;
  clGetPlatformIDs(1, &platform, NULL);

  cl_device_id device;
  clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, NULL);

  cl_context context = clCreateContext(NULL, 1, &device, NULL, NULL, NULL);

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
    /*
    for (int j = 0; j < 512; j++) {
      std::complex<float> c = std::complex<float>(j / 512. * 3 - 2.25, i / 512. * 3 - 1.5);
      float f = mandelbrot(c, 128);
      int color = f * 255;
      img[(i * 512 + j) * 3 + 0] = img[(i * 512 + j) * 3 + 1] = img[(i * 512 + j) * 3 + 2] = color;
    }
    */

    for (int j = 0; j < 512; j++) {
      std::complex<float> t[4096];
      float ri = i + (rand() * 1. / RAND_MAX - 0.5); // rand() % 512;
      float rj = j + (rand() * 1. / RAND_MAX - 0.5); // rand() % 512;
      int steps;
      std::complex<float> c = std::complex<float>(rj / 512. * 3 - 2.25, ri / 512. * 3 - 1.5);
      steps = buddhabrot(c, t, 1024);
      if (steps < 1024) {
        for (int k = 0; k < steps - 1; k++) {
          int x = (t[k].real() + 2.25) / 3 * 512;
          int y = (t[k].imag() + 1.5) / 3 * 512;
          if (x > 0 && x < 512 && y > 0 && y < 512)
            img[((512 - x) * 512 + y) * 3 + 0]++;
        }
      }
    }
    for (int j = 0; j < 512; j++) {
      std::complex<float> t[4096];
      float ri = i + (rand() * 1. / RAND_MAX - 0.5); // rand() % 512;
      float rj = j + (rand() * 1. / RAND_MAX - 0.5); // rand() % 512;
      int steps;
      std::complex<float> c = std::complex<float>(rj / 512. * 3 - 2.25, ri / 512. * 3 - 1.5);
      steps = buddhabrot(c, t, 2048);
      if (steps < 2048) {
        for (int k = 0; k < steps - 1; k++) {
          int x = (t[k].real() + 2.25) / 3 * 512;
          int y = (t[k].imag() + 1.5) / 3 * 512;
          if (x > 0 && x < 512 && y > 0 && y < 512)
            img[((512 - x) * 512 + y) * 3 + 1]++;
        }
      }
    }
    for (int j = 0; j < 512; j++) {
      std::complex<float> t[4096];
      float ri = i + (rand() * 1. / RAND_MAX - 0.5); // rand() % 512;
      float rj = j + (rand() * 1. / RAND_MAX - 0.5); // rand() % 512;
      int steps;
      std::complex<float> c = std::complex<float>(rj / 512. * 3 - 2.25, ri / 512. * 3 - 1.5);
      steps = buddhabrot(c, t, 4096);
      if (steps < 4096) {
        for (int k = 0; k < steps - 1; k++) {
          int x = (t[k].real() + 2.25) / 3 * 512;
          int y = (t[k].imag() + 1.5) / 3 * 512;
          if (x > 0 && x < 512 && y > 0 && y < 512)
            img[((512 - x) * 512 + y) * 3 + 2]++;
        }
      }
    }
    i = (i + 1) % 512;
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, 512, 512, 0, GL_RGB, GL_UNSIGNED_BYTE, img);
    glDrawArrays(GL_TRIANGLES, 0, 6);
    glfwSwapBuffers(window);
    glfwPollEvents();
  }

  glfwTerminate();

  return 0;
}

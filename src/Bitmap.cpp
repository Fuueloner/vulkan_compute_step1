#include <vector>
#include <fstream>
#include <iostream>
#include <cstring>

struct Pixel { unsigned char r, g, b; };

void WriteBMP(const char* fname, Pixel* a_pixelData, int width, int height)
{
  int paddedsize = (width*height) * sizeof(Pixel);

  unsigned char bmpfileheader[14] = {'B','M', 0,0,0,0, 0,0, 0,0, 54,0,0,0};
  unsigned char bmpinfoheader[40] = {40,0,0,0, 0,0,0,0, 0,0,0,0, 1,0, 24,0};

  bmpfileheader[ 2] = (unsigned char)(paddedsize    );
  bmpfileheader[ 3] = (unsigned char)(paddedsize>> 8);
  bmpfileheader[ 4] = (unsigned char)(paddedsize>>16);
  bmpfileheader[ 5] = (unsigned char)(paddedsize>>24);

  bmpinfoheader[ 4] = (unsigned char)(width    );
  bmpinfoheader[ 5] = (unsigned char)(width>> 8);
  bmpinfoheader[ 6] = (unsigned char)(width>>16);
  bmpinfoheader[ 7] = (unsigned char)(width>>24);
  bmpinfoheader[ 8] = (unsigned char)(height    );
  bmpinfoheader[ 9] = (unsigned char)(height>> 8);
  bmpinfoheader[10] = (unsigned char)(height>>16);
  bmpinfoheader[11] = (unsigned char)(height>>24);

  std::ofstream out(fname, std::ios::out | std::ios::binary);
  out.write((const char*)bmpfileheader, 14);
  out.write((const char*)bmpinfoheader, 40);
  out.write((const char*)a_pixelData, paddedsize);
  out.flush();
  out.close();
}

void SaveBMP(const char* fname, const unsigned int* pixels, int w, int h)
{
  std::vector<Pixel> pixels2(w*h);

  for (size_t i = 0; i < pixels2.size(); i++)
  {
    Pixel px;
    px.r       = (pixels[i] & 0x00FF0000) >> 16;
    px.g       = (pixels[i] & 0x0000FF00) >> 8;
    px.b       = (pixels[i] & 0x000000FF);
    pixels2[i] = px;
  }

  WriteBMP(fname, &pixels2[0], w, h);
}

unsigned int* LoadBMP(const char* fname, int& w, int& h)
{
  char bmpFileHeader[14];
  char bmpInfoHeader[40];

  std::ifstream inputFileStream(fname, std::ios::in | std::ios::binary);
  
  inputFileStream.read(bmpFileHeader, 14);
  inputFileStream.read(bmpInfoHeader, 40);
  
  int paddedSize(0);
  for (size_t i(5); i > 1; --i)
  {
    paddedSize <<= 8;
    paddedSize += (int)(bmpFileHeader[i]);
  }
  for (size_t i(7); i > 3; --i)
  {
    w <<= 8;
    w += (int)(bmpInfoHeader[i]);
  }
  for (size_t i(11); i > 7; --i)
  {
    h <<= 8;
    h += (int)(bmpInfoHeader[i]);
  }
  
  std::cout << "paddedSize: " << paddedSize << std::endl;
  std::cout << "Width: " << w << std::endl;
  std::cout << "Height: " << h << std::endl;

  size_t count(w * h);
  Pixel* internalPixels(new Pixel[count]);
  
  inputFileStream.read((char*)internalPixels, paddedSize);
  
  unsigned int* pixels(new unsigned int[count]);
  for (size_t i(0); i < count; i++)
  {
    unsigned int px(0);
    px += ((Pixel*)internalPixels)[i].r;
    px <<= 8;
    px += ((Pixel*)internalPixels)[i].g;
    px <<= 8;
    px += ((Pixel*)internalPixels)[i].b;
    pixels[i] = px;
  }
  delete[] internalPixels;
  return pixels;
}


#ifndef _UNSHARP_MASK_HPP_
#define _UNSHARP_MASK_HPP_

#include "blur.hpp"
#include "add_weighted.hpp"
#include "ppm.hpp"

void unsharp_mask(unsigned char *out, const unsigned char *in,
                  const int blur_radius,
                  const unsigned w, const unsigned h, const unsigned nchannels)
{
  const auto alpha = 1.5f; const auto beta = -0.5f;
  std::vector<unsigned char> blur1, blur2, blur3;

  blur1.resize(w * h * nchannels);
  blur2.resize(w * h * nchannels);
  blur3.resize(w * h * nchannels);

  auto serialBlur1TimeStart = std::chrono::steady_clock::now();
  blur(blur1.data(),   in,           blur_radius, w, h, nchannels);
  auto serialBlur1TimeStop = std::chrono::steady_clock::now();
  std::cout << "Serial blur 1 took: " << std::chrono::duration<double>(serialBlur1TimeStop - serialBlur1TimeStart).count() << " seconds.\n";

  auto serialBlur2TimeStart = std::chrono::steady_clock::now();
  blur(blur2.data(),   blur1.data(), blur_radius, w, h, nchannels);
  auto serialBlur2TimeStop = std::chrono::steady_clock::now();
  std::cout << "Serial blur 2 took: " << std::chrono::duration<double>(serialBlur2TimeStop - serialBlur2TimeStart).count() << " seconds.\n";

  auto serialBlur3TimeStart = std::chrono::steady_clock::now();
  blur(blur3.data(),   blur2.data(), blur_radius, w, h, nchannels);
  auto serialBlur3TimeStop = std::chrono::steady_clock::now();
  std::cout << "Serial blur 3 took: " << std::chrono::duration<double>(serialBlur3TimeStop - serialBlur3TimeStart).count() << " seconds.\n";

  auto serialAddWeightedimeStart = std::chrono::steady_clock::now();
  add_weighted(out, in, alpha, blur3.data(), beta, 0.0f, w, h, nchannels);
  auto serialAddWeightedimeStop = std::chrono::steady_clock::now();
  std::cout << "Serial addWeighted took: " << std::chrono::duration<double>(serialAddWeightedimeStop - serialAddWeightedimeStart).count() << " seconds.\n";
}

#endif // _UNSHARP_MASK_HPP_

#include <chrono>
#include "unsharp_mask.hpp"
#include <SYCL/sycl.hpp>
using namespace cl::sycl;

// Apply an unsharp mask to the 24-bit PPM loaded from the file path of
// the first input argument; then write the sharpened output to the file path
// of the second argument. The third argument provides the blur radius.

//SYCL
class kernelBlurOne;
class kernelBlurTwo;
class kernelBlurThree;
class kernelAddWeighted;
/* Classes can inherit from the device_selector class to allow users
 * to dictate the criteria for choosing a device from those that might be
 * present on a system. This example looks for a device with SPIR support
 * and prefers GPUs over CPUs. */
class custom_selector : public device_selector {
public:
	custom_selector() : device_selector() {}

	/* The selection is performed via the () operator in the base
	 * selector class.This method will be called once per device in each
	 * platform. Note that all platforms are evaluated whenever there is
	 * a device selection. */
	int operator()(const device& device) const override {
		/* We only give a valid score to devices that support SPIR. */
		if (device.has_extension(cl::sycl::string_class("cl_khr_spir"))) {
			if (device.get_info<info::device::device_type>() ==
				info::device_type::cpu) {
				return 50;
			}
			if (device.get_info<info::device::device_type>() ==
				info::device_type::gpu) {
				return 100;
			}
		}
		/* Devices with a negative score will never be chosen. */
		return -1;
	}
};

struct uchar_three {
	unsigned char r, g, b;
};

int main(int argc, char *argv[])
{

	//User Input
	int blurInt;
	std::cout << "Please enter the integer value you would like to use for the blur radius in the unsharp mask code: ";
	std::cin >> blurInt;
	std::cout << "You have selected the integer " << blurInt << " to be used for the blur radius.\n";

	//DATA USED BY SYCL and SERIAL code
	const char *ifilename = argc > 1 ? argv[1] : "../../unsharp_mask/ghost-town-8k.ppm";
	const char *oFileNameSYCL = argc > 2 ? argv[2] : "../../unsharp_mask/outSYCL.ppm";
	const char *oFileNameSerial = argc > 3 ? argv[3] : "../../unsharp_mask/outSerial.ppm";
	const int blur_radius = blurInt;
	ppm img;
	std::vector<unsigned char> data_in, data_sharp;
	img.read(ifilename, data_in);
	unsigned w = img.w;//for use within SYCL block
	unsigned h = img.h;//for use within SYCL blick
	data_sharp.resize(img.w * img.h * img.nchannels);
	std::vector<unsigned char> blur1, blur2, blur3;
	blur1.resize(img.w * img.h * img.nchannels);
	blur2.resize(img.w * img.h * img.nchannels);
	blur3.resize(img.w * img.h * img.nchannels);

  //SYCL block
	try {
		custom_selector selector;
		queue myQueue(selector, async_handler{});
		std::cout << "\nRunning on "
			<< myQueue.get_device().get_info<cl::sycl::info::device::name>()
			<< "\n";

		auto syclTotalTimeStart = std::chrono::steady_clock::now();
		
	    //BLUR 1
		buffer<uchar_three, 2> bufIBlur1(reinterpret_cast<uchar_three *>(data_in.data()), range<2>(w, h));
		buffer<uchar_three, 2> bufOBlur1(reinterpret_cast<uchar_three *>(blur1.data()), range<2>(w, h));
		auto blur1TimeStart = std::chrono::steady_clock::now();
		myQueue.submit([&](handler &cgh) {
			auto inABlur1 = bufIBlur1.get_access<access::mode::read>(cgh);
			auto outBlur1 = bufOBlur1.get_access<access::mode::write>(cgh);
			cgh.parallel_for<kernelBlurOne>(range<2>(w, h),
				[=](id<2> ik) {
						float red_total = 0, green_total = 0, blue_total = 0;

						for (int j = ik.get(0) - blur_radius + 1; j < ik.get(0) + blur_radius; ++j) {
							for (int i = ik.get(1) - blur_radius + 1; i < ik.get(1) + blur_radius; ++i) {
								const unsigned r_i = i < 0 ? 0 : i >= w ? w - 1 : i;
								const unsigned r_j = j < 0 ? 0 : j >= h ? h - 1 : j;
								red_total += inABlur1[r_i][r_j].r;
								green_total += inABlur1[r_i][r_j].g;
								blue_total += inABlur1[r_i][r_j].b;
							}
						}

						const unsigned nsamples = (blur_radius * 2 - 1) * (blur_radius * 2 - 1);
						outBlur1[ik.get(0)][ik.get(1)].r = red_total / nsamples;
						outBlur1[ik.get(0)][ik.get(1)].g = green_total / nsamples;
						outBlur1[ik.get(0)][ik.get(1)].b = blue_total / nsamples;
				});
		});
		auto blur1TimeStop = std::chrono::steady_clock::now();
		std::cout << "SYCL blur 1 took: " << std::chrono::duration<double>(blur1TimeStop - blur1TimeStart).count() << " seconds.\n";
		//BLUR 2
		buffer<uchar_three, 2> bufIBlur2(reinterpret_cast<uchar_three *>(blur1.data()), range<2>(w, h));
		buffer<uchar_three, 2> bufOBlur2(reinterpret_cast<uchar_three *>(blur2.data()), range<2>(w, h));
		auto blur2TimeStart = std::chrono::steady_clock::now();
		myQueue.submit([&](handler &cgh) {
			auto inABlur2 = bufIBlur2.get_access<access::mode::read>(cgh);
			auto outBlur2 = bufOBlur2.get_access<access::mode::write>(cgh);
			cgh.parallel_for<kernelBlurTwo>(range<2>(w, h),
				[=](id<2> ik) {
						float red_total = 0, green_total = 0, blue_total = 0;

						for (int j = ik.get(0) - blur_radius + 1; j < ik.get(0) + blur_radius; ++j) {
							for (int i = ik.get(1) - blur_radius + 1; i < ik.get(1) + blur_radius; ++i) {
								const unsigned r_i = i < 0 ? 0 : i >= w ? w - 1 : i;
								const unsigned r_j = j < 0 ? 0 : j >= h ? h - 1 : j;
								red_total += inABlur2[r_i][r_j].r;
								green_total += inABlur2[r_i][r_j].g;
								blue_total += inABlur2[r_i][r_j].b;
							}
						}

						const unsigned nsamples = (blur_radius * 2 - 1) * (blur_radius * 2 - 1);
						outBlur2[ik.get(0)][ik.get(1)].r = red_total / nsamples;
						outBlur2[ik.get(0)][ik.get(1)].g = green_total / nsamples;
						outBlur2[ik.get(0)][ik.get(1)].b = blue_total / nsamples;
				});
		});
		auto blur2TimeStop = std::chrono::steady_clock::now();
		std::cout << "SYCL blur 2 took: " << std::chrono::duration<double>(blur2TimeStop - blur2TimeStart).count() << " seconds.\n";
		//BLUR 3
		buffer<uchar_three, 2> bufIBlur3(reinterpret_cast<uchar_three *>(blur2.data()), range<2>(w, h));
		buffer<uchar_three, 2> bufOBlur3(reinterpret_cast<uchar_three *>(blur3.data()), range<2>(w, h));
		auto blur3TimeStart = std::chrono::steady_clock::now();
		myQueue.submit([&](handler &cgh) {
			auto inABlur3 = bufIBlur3.get_access<access::mode::read>(cgh);
			auto outBlur3 = bufOBlur3.get_access<access::mode::write>(cgh);
			cgh.parallel_for<kernelBlurThree>(range<2>(w, h),
				[=](id<2> ik) {
						float red_total = 0, green_total = 0, blue_total = 0;

						for (int j = ik.get(0) - blur_radius + 1; j < ik.get(0) + blur_radius; ++j) {
							for (int i = ik.get(1) - blur_radius + 1; i < ik.get(1) + blur_radius; ++i) {
								const unsigned r_i = i < 0 ? 0 : i >= w ? w - 1 : i;
								const unsigned r_j = j < 0 ? 0 : j >= h ? h - 1 : j;
								red_total += inABlur3[r_i][r_j].r;
								green_total += inABlur3[r_i][r_j].g;
								blue_total += inABlur3[r_i][r_j].b;
							}
						}

						const unsigned nsamples = (blur_radius * 2 - 1) * (blur_radius * 2 - 1);
						outBlur3[ik.get(0)][ik.get(1)].r = red_total / nsamples;
						outBlur3[ik.get(0)][ik.get(1)].g = green_total / nsamples;
						outBlur3[ik.get(0)][ik.get(1)].b = blue_total / nsamples;
				});
		});
		auto blur3TimeStop = std::chrono::steady_clock::now();
		std::cout << "SYCL blur 3 took: " << std::chrono::duration<double>(blur3TimeStop - blur3TimeStart).count() << " seconds.\n";
		//ADD WEIGHTED
		buffer<uchar_three, 2> bufIAWeight1(reinterpret_cast<uchar_three *>(data_in.data()), range<2>(w, h));
		buffer<uchar_three, 2> bufIAWeight2(reinterpret_cast<uchar_three *>(blur3.data()), range<2>(w, h));
		buffer<uchar_three, 2> bufOAWeight(reinterpret_cast<uchar_three *>(data_sharp.data()), range<2>(w, h));
		auto addWeightedTimeStart = std::chrono::steady_clock::now();
		myQueue.submit([&](handler &cgh) {
			auto inAAWeight1 = bufIAWeight1.get_access<access::mode::read>(cgh);
			auto inAAWeight2 = bufIAWeight2.get_access<access::mode::read>(cgh);
			auto outAWeight = bufOAWeight.get_access<access::mode::write>(cgh);
			cgh.parallel_for<kernelAddWeighted>(range<2>(w, h),
				[=](id<2> ik) {
						const auto alpha = 1.5f; const auto beta = -0.5f; const auto gamma = 0.0f;

						float tmp = inAAWeight1[ik.get(0)][ik.get(1)].r * alpha + inAAWeight2[ik.get(0)][ik.get(1)].r * beta + gamma;
						outAWeight[ik.get(0)][ik.get(1)].r = tmp < 0 ? 0 : tmp > UCHAR_MAX ? UCHAR_MAX : tmp;

						tmp = inAAWeight1[ik.get(0)][ik.get(1)].g * alpha + inAAWeight2[ik.get(0)][ik.get(1)].g * beta + gamma;
						outAWeight[ik.get(0)][ik.get(1)].g = tmp < 0 ? 0 : tmp > UCHAR_MAX ? UCHAR_MAX : tmp;

						tmp = inAAWeight1[ik.get(0)][ik.get(1)].b * alpha + inAAWeight2[ik.get(0)][ik.get(1)].b * beta + gamma;
						outAWeight[ik.get(0)][ik.get(1)].b = tmp < 0 ? 0 : tmp > UCHAR_MAX ? UCHAR_MAX : tmp;
				});
		});
		auto addWeightedTimeStop = std::chrono::steady_clock::now();
		std::cout << "SYCL addWeighted took: " << std::chrono::duration<double>(addWeightedTimeStop - addWeightedTimeStart).count() << " seconds.\n";
		
		auto syclTotalTimeStop = std::chrono::steady_clock::now();
		std::cout << "SYCL code in total took: " << std::chrono::duration<double>(syclTotalTimeStop - syclTotalTimeStart).count() << " seconds.\n";

		myQueue.wait_and_throw();
	}catch (exception e) { std::cout << "exception caught : " << e.what() << "\n"; }
	img.write(oFileNameSYCL, data_sharp);


    //SERIAL block
    std::vector<unsigned char>  data_sharp_serial;
    data_sharp_serial.resize(img.w * img.h * img.nchannels);

    std::cout << "\nRunning on CPU\n";

    auto serialTotalTimeStart = std::chrono::steady_clock::now();
    unsharp_mask(data_sharp_serial.data(), data_in.data(), blur_radius,
                 img.w, img.h, img.nchannels);
    auto serialTotalTimeStop = std::chrono::steady_clock::now();

    std::cout << "Serial code in total took: " << std::chrono::duration<double>(serialTotalTimeStop - serialTotalTimeStart).count() << " seconds.\n";
    img.write(oFileNameSerial, data_sharp_serial);
  
    return 0;
}


/**
 * File              : NetVLAD.cpp
 * Author            : Long Vuong <vuong@wheel.me>
 * Date              : 29.08.2023
 * Last Modified Date: 29.08.2023
 * Last Modified By  : Long Vuong <vuong@wheel.me>
 */

#include <chrono>
#include <torch_cpp/NetVLAD.hpp>

netvlad_torch::netvlad_torch(std::string_view checkpoint_path) {
  try {
    // Load TorchScript trace of PyTorch implementation of NetVLAD based on https://github.com/Nanne/pytorch-NetVlad
    script_net_ = torch::jit::load(std::string(checkpoint_path));
    script_net_.to(DTYPE); // conversion probably not necessary, but will avoid errors
    script_net_.eval();
  } catch (const c10::Error& e) {
    LOG(FATAL) << "Failed to load NetVLAD model.";
  }
  LOG(INFO) << "NetVLAD loaded successfully.";
  printf("NetVLAD loaded successfully.\n");
}

void netvlad_torch::transform(const cv::Mat& img, at::Tensor& rep) {
  auto start = std::chrono::high_resolution_clock::now();
  // Assert that the returned tensor is normalized and detached
  std::vector<torch::jit::IValue> inputs;
  // Interprets the raw mono8 image data as a 1xRxC matrix of bytes (uint8)
  // Source: https://github.com/pytorch/pytorch/issues/12506#issuecomment-429573396
  at::Tensor tensor_img = torch::from_blob(img.data, {img.channels(), img.rows, img.cols}, at::TensorOptions(at::kByte));
  tensor_img = tensor_img.to(DTYPE); // Matching model
  // Expand to (1, 3, R, C) to pretend we have a color image with batch size 1.
  // This was included in the training/testing code, but I haven't seen examples of its use, so performance is unclear
  tensor_img = tensor_img.expand({1, 3, -1, -1});
  // Normalize the image based on constants from https://github.com/Nanne/pytorch-NetVlad/blob/master/pittsburgh.py input_transform()
  tensor_img = (tensor_img/255.f - at::tensor({0.485, 0.456, 0.406}).to(DTYPE).expand({1, 1, 1, 3}).permute({0, 3, 1, 2}))
                 / at::tensor({0.229, 0.224, 0.225}).to(DTYPE).expand({1, 1, 1, 3}).permute({0, 3, 1, 2});
  inputs.push_back(tensor_img);

  // Run NetVLAD
  at::Tensor output = script_net_.forward(inputs).toTensor();
  if(output.dim() != 2 || output.size(0) != 1 || output.size(1) != FULL_VECTOR_SIZE) {
    LOG(FATAL) << "NetVLAD output is not expected size: dim: " << output.dim() << " [" << output.size(0) << ", " << output.size(-1) << "]";
  }

  // TODO: determine an appropriate PCA reduction for output tensor to avoid storing 16MB per image

  rep = output.squeeze();
  auto stop = std::chrono::high_resolution_clock::now();
  auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start);
  std::cout << "NetVLAD transform took " << duration.count() << "ms" << std::endl;
}

netvlad_torch::~netvlad_torch() = default;


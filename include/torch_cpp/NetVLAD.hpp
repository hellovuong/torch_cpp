/**
 * File              : NetVLAD.hpp
 * Author            : Long Vuong <vuong@wheel.me>
 * Date              : 29.08.2023
 * Last Modified Date: 29.08.2023
 * Last Modified By  : Long Vuong <vuong@wheel.me>
 */

#include <torch/torch.h>
#include <torch/script.h>
#include <opencv2/opencv.hpp>

constexpr int FULL_VECTOR_SIZE = 32768;
constexpr auto DTYPE = at::kFloat;

class netvlad_torch {
public:
  explicit netvlad_torch(std::string_view checkpoint_path);
  netvlad_torch(netvlad_torch &&) = default;
  netvlad_torch(const netvlad_torch &) = default;
  ~netvlad_torch();

  void transform(const cv::Mat& img, at::Tensor& rep);
  static double score(const at::Tensor& rep1, const at::Tensor& rep2)  {
  // Can assume that tensors are normalized, so Euclidean norm^2 is in [0, 2], with 0 best.
  // This score was designed to be interpreted via norm, so we return 1 - norm^2 / 2 to
  // follow convention of score in [0, 1] with 1 best
  if(rep1.size(0) == 0 || rep2.size(0) == 0) {
    // one rep is uninitialized, so the score should be bad
    return 0;
  } else if(rep1.size(0) != FULL_VECTOR_SIZE || rep2.size(0) != FULL_VECTOR_SIZE) {
    LOG(FATAL) << "NetVLAD::score had vector inputs of different sizes: " << rep1.size(0) << " " << rep2.size(0);
  }
  const double norm = (rep1 - rep2).norm().item<double>();
  if(norm > 2) {
    LOG(WARNING) << "NetVLAD::score had large norm: " << norm << " from inputs with norms: "
                 << rep1.norm().item<double>() << " " << rep2.norm().item<double>();
  }
  return std::max(1 - norm/2., 0.);
}

private:
 std::map<unsigned int, at::Tensor> database_;
 torch::jit::script::Module script_net_;
 torch::NoGradGuard grad_guard_;
};


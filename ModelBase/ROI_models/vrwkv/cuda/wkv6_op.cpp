#include <torch/extension.h>
#include "ATen/ATen.h"
typedef float fp32;

void cuda_forward(int B, int T, int C, int H, fp32 *r, fp32 *k, fp32 *v, fp32 *w, fp32 *u, fp32 *y);
void cuda_backward(int B, int T, int C, int H, fp32 *r, fp32 *k, fp32 *v, fp32 *w, fp32 *u, fp32 *gy, fp32 *gr, fp32 *gk, fp32 *gv, fp32 *gw, fp32 *gu);

void forward(int64_t B, int64_t T, int64_t C, int64_t H, torch::Tensor &r, torch::Tensor &k, torch::Tensor &v, torch::Tensor &w, torch::Tensor &u, torch::Tensor &y) {
    cuda_forward(B, T, C, H, r.data_ptr<fp32>(), k.data_ptr<fp32>(), v.data_ptr<fp32>(), w.data_ptr<fp32>(), u.data_ptr<fp32>(), y.data_ptr<fp32>());
}
void backward(int64_t B, int64_t T, int64_t C, int64_t H, torch::Tensor &r, torch::Tensor &k, torch::Tensor &v, torch::Tensor &w, torch::Tensor &u, torch::Tensor &gy, torch::Tensor &gr, torch::Tensor &gk, torch::Tensor &gv, torch::Tensor &gw, torch::Tensor &gu) {
    cuda_backward(B, T, C, H, r.data_ptr<fp32>(), k.data_ptr<fp32>(), v.data_ptr<fp32>(), w.data_ptr<fp32>(), u.data_ptr<fp32>(), gy.data_ptr<fp32>(), gr.data_ptr<fp32>(), gk.data_ptr<fp32>(), gv.data_ptr<fp32>(), gw.data_ptr<fp32>(), gu.data_ptr<fp32>());
}

TORCH_LIBRARY(wkv6, m) {
    m.def("forward(int B, int T, int C, int H, Tensor r, Tensor k, Tensor v, Tensor w, Tensor u, Tensor(a!) y) -> ()");
    m.def("backward(int B, int T, int C, int H, Tensor r, Tensor k, Tensor v, Tensor w, Tensor u, Tensor gy, Tensor(a!) gr, Tensor(b!) gk, Tensor(c!) gv, Tensor(d!) gw, Tensor(e!) gu) -> ()");
}

TORCH_LIBRARY_IMPL(wkv6, CUDA, m) {
    m.impl("forward", &forward);
    m.impl("backward", &backward);
}

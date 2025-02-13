#include <torch/extension.h>

void cuda_forward(int B, int T, int H, float*w, float*q, float*k, float*v, float*z, float*a, float*y, float*s, float*sa);

void forward(torch::Tensor &w, torch::Tensor &q, torch::Tensor &k, torch::Tensor &v, torch::Tensor &z, torch::Tensor &a, torch::Tensor &y, torch::Tensor &s, torch::Tensor &sa) {
    int B = w.sizes()[0], T = w.sizes()[1], H = w.sizes()[2];
    cuda_forward(B, T, H, (float*)w.data_ptr(), (float*)q.data_ptr(), (float*)k.data_ptr(), (float*)v.data_ptr(), (float*)z.data_ptr(), (float*)a.data_ptr(), (float*)y.data_ptr(), (float*)s.data_ptr(), (float*)sa.data_ptr());
}

void cuda_backward(int B, int T, int H, float*w, float*q, float*k, float*v, float*z, float*a, float*dy, float*s, float*sa, float*dw, float*dq, float*dk, float*dv, float*dz, float*da);

void backward(torch::Tensor &w, torch::Tensor &q, torch::Tensor &k, torch::Tensor &v, torch::Tensor &z, torch::Tensor &a, torch::Tensor &dy,
        torch::Tensor &s, torch::Tensor &sa, torch::Tensor &dw, torch::Tensor &dq, torch::Tensor &dk, torch::Tensor &dv, torch::Tensor &dz, torch::Tensor &da) {
    int B = w.sizes()[0], T = w.sizes()[1], H = w.sizes()[2];
    cuda_backward(B, T, H, (float*)w.data_ptr(), (float*)q.data_ptr(), (float*)k.data_ptr(), (float*)v.data_ptr(), (float*)z.data_ptr(), (float*)a.data_ptr(), (float*)dy.data_ptr(), 
            (float*)s.data_ptr(), (float*)sa.data_ptr(), (float*)dw.data_ptr(), (float*)dq.data_ptr(), (float*)dk.data_ptr(), (float*)dv.data_ptr(), (float*)dz.data_ptr(), (float*)da.data_ptr());
}

TORCH_LIBRARY(wind_backstepping, m) {
    m.def("forward(Tensor w, Tensor q, Tensor k, Tensor v, Tensor z, Tensor a, Tensor(a!) y, Tensor(b!) s, Tensor(c!) sa) -> ()");
    m.def("backward(Tensor w, Tensor q, Tensor k, Tensor v, Tensor z, Tensor a, Tensor dy, Tensor s, Tensor sa, Tensor(a!) dw, Tensor(b!) dq, Tensor(c!) dk, Tensor(d!) dv, Tensor(e!) dz, Tensor(f!) da) -> ()");
}

TORCH_LIBRARY_IMPL(wind_backstepping, CUDA, m) {
    m.impl("forward", &forward);
    m.impl("backward", &backward);
}

#include <assert.h>
using fp32 = float;
typedef fp32 * __restrict__ F_;

__global__ void forward_kernel(int T, int H, F_ w_, F_ q_, F_ k_, F_ v_, F_ a_, F_ b_, fp32* y_, float* s_, float* sa_) {
    constexpr int C = _C_;
    int bb = blockIdx.y, hh = blockIdx.x, i = threadIdx.x;

    float state[C] = {0};
    __shared__ float q[C], k[C], w[C], a[C], b[C];

    for (int t = 0; t < T; t++) {
        int ind = bb*T*H*C + t*H*C + hh * C + i;
        __syncthreads();
        q[i] = q_[ind];
        w[i] = __expf(-__expf(w_[ind]));
        k[i] = k_[ind];
        a[i] = a_[ind];
        b[i] = b_[ind];
        __syncthreads();

        float sa = 0;
#pragma unroll
        for (int j = 0; j < C; j++) {
            sa += a[j] * state[j];
        }
        sa_[ind] = sa;

        float v = v_[ind];
        float y = 0;
#pragma unroll
        for (int j = 0; j < C; j++) {
            float& s = state[j];
            s = s * w[j] + sa * b[j] + k[j] * v;
            y += s * q[j];
        }
        y_[ind] = y;

        if ((t+1)%_CHUNK_LEN_ == 0) {
            int base = (bb*H+hh)*(T/_CHUNK_LEN_)*C*C + (t/_CHUNK_LEN_)*C*C + i;
#pragma unroll
            for (int j = 0; j < C; j++) {
                s_[base + j*C] = state[j];
            }
        }
    }
}

__global__ void backward_kernel(int T, int H, F_ w_, F_ q_, F_ k_, F_ v_, F_ a_, F_ b_, F_ dy_, float * __restrict__ s_, float * __restrict__ sa_, fp32* dw_, fp32* dq_, fp32* dk_, fp32* dv_, fp32* da_, fp32* db_) {
    constexpr int C = _C_;
    int bb = blockIdx.y, hh = blockIdx.x, i = threadIdx.x;

    float stateT[C] = {0}, dstate[C] = {0}, dstateT[C] = {0};
    __shared__ float w[C], q[C], k[C], v[C], a[C], b[C], dy[C], sa[C], dSb_shared[C];
    float qi, wi, ki, ai, bi, dyi;

    for (int t = T-1; t >= 0; t--) {
        int ind = bb*T*H*C + t*H*C + hh * C + i;
        __syncthreads();
        q[i] = qi = q_[ind];
        float wi_fac = -__expf(w_[ind]);
        w[i] = wi = __expf(wi_fac);
        k[i] = ki = k_[ind];
        a[i] = ai = a_[ind];
        b[i] = bi = b_[ind];
        v[i] = v_[ind];
        dy[i] = dyi = dy_[ind];
        sa[i] = sa_[ind];
        __syncthreads();

        if ((t+1)%_CHUNK_LEN_ == 0) {
            int base = (bb*H+hh)*(T/_CHUNK_LEN_)*C*C + (t/_CHUNK_LEN_)*C*C + i*C;
#pragma unroll
            for (int j = 0; j < C; j++) {
                stateT[j] = s_[base + j];
            }
        }

        float dq = 0;
#pragma unroll
        for (int j = 0; j < C; j++) {
            dq += stateT[j]*dy[j];
        }
        dq_[ind] = dq;

        float iwi = 1.0f/wi;
#pragma unroll        
        for (int j = 0; j < C; j++) {
            stateT[j] = (stateT[j] - ki*v[j] - bi*sa[j]) * iwi;
            dstate[j] += dyi * q[j];
            dstateT[j] += qi * dy[j];
        }

        float dw = 0, dk = 0, dv = 0, db = 0, dSb = 0;
#pragma unroll
        for (int j = 0; j < C; j++) {
            dw += dstateT[j]*stateT[j];
            dk += dstateT[j]*v[j];
            dv += dstate[j]*k[j];
            dSb += dstate[j]*b[j];
            db += dstateT[j]*sa[j];
        }
        dw_[ind] = dw * wi * wi_fac;
        dk_[ind] = dk;
        dv_[ind] = dv;
        db_[ind] = db;

        __syncthreads();
        dSb_shared[i] = dSb;
        __syncthreads();

        float da = 0;
#pragma unroll
        for (int j = 0; j < C; j++) {
            da += stateT[j]*dSb_shared[j];
        }
        da_[ind] = da;

#pragma unroll        
        for (int j = 0; j < C; j++) {
            dstate[j] = dstate[j]*w[j] + dSb * a[j];
            dstateT[j] = dstateT[j]*wi + ai * dSb_shared[j];
        }
    }
}

void cuda_forward(int B, int T, int H, fp32*w, fp32*q, fp32*k, fp32*v, fp32*z, fp32*a, fp32*y, float*s, float*sa) {
    forward_kernel<<<dim3(H,B), dim3(_C_)>>>(T,H,w,q,k,v,z,a,y,s,sa);
}
void cuda_backward(int B, int T, int H, fp32*w, fp32*q, fp32*k, fp32*v, fp32*z, fp32*a, fp32*dy, float*s, float*sa, fp32*dw, fp32*dq, fp32*dk, fp32*dv, fp32*dz, fp32*da) {
    assert(T%_CHUNK_LEN_ == 0);
    backward_kernel<<<dim3(H,B), dim3(_C_)>>>(T,H,w,q,k,v,z,a,dy,s,sa,dw,dq,dk,dv,dz,da);
}

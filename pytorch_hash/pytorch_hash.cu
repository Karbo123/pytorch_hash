/*
    fnv-1a hashing for: Byte, Char, Double, Float, Int, Long, Short, Half, ComplexFloat, ComplexDouble
      input data must have shape == (num, dim)
      output data has shape == (num, )
*/
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <pybind11/pybind11.h>
#include <thrust/for_each.h>
#include <thrust/execution_policy.h>
#include <thrust/iterator/counting_iterator.h>

template <typename scalar_t>
struct hash_functor {
    scalar_t* ptr_in;
    int64_t* ptr_out;
    uint64_t num, dim;
    hash_functor(scalar_t* ptr_in, int64_t* ptr_out, uint64_t num, uint64_t dim): 
                 ptr_in(ptr_in), ptr_out(ptr_out), num(num), dim(dim) {}
    __host__ __device__ void operator()(uint64_t index_row) {
        auto ptr_in_i = reinterpret_cast<u_char*>(ptr_in + index_row * dim);
        uint64_t number = dim * sizeof(scalar_t);
        uint64_t result = 0XCBF29CE484222325;
        while (number--) result = (result ^ (*ptr_in_i++)) * 0x00000100000001B3;
        ptr_out[index_row] = reinterpret_cast<int64_t&>(result);
    }
};

torch::Tensor hash(torch::Tensor data) {
    TORCH_CHECK(data.is_contiguous(), "data must be contiguous");
    auto num = data.size(0);
    auto dim = data.size(1);
    auto is_cuda = data.is_cuda();

    auto options = torch::TensorOptions().dtype(torch::kInt64);
    torch::Tensor out = data.new_empty({num}, options);

    AT_DISPATCH_ALL_TYPES_AND(at::ScalarType::Half, data.type(), "hash", [&] {
        auto functor = hash_functor<scalar_t>(data.data_ptr<scalar_t>(), 
                                              out .data_ptr<int64_t>(),
                                              num, dim);
        auto iter_rows = thrust::counting_iterator<uint64_t>(0);
        if (is_cuda) {
            auto stream = at::cuda::getCurrentCUDAStream();
            auto device = thrust::cuda::par.on(stream);
            thrust::for_each_n(device, iter_rows, num, functor);
            cudaStreamSynchronize(stream);
        }
        else {
            thrust::for_each_n(thrust::host, iter_rows, num, functor);
        }
    });
    
    return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("hash", &hash, "fnv-1a hashing for pytorch", pybind11::arg("data"));
}
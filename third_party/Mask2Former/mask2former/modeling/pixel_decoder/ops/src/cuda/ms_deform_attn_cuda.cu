/*!
**************************************************************************************************
* Deformable DETR
* Copyright (c) 2020 SenseTime. All Rights Reserved.
* Licensed under the Apache License, Version 2.0 [see LICENSE for details]
**************************************************************************************************
* Modified from https://github.com/chengdazhi/Deformable-Convolution-V2-PyTorch/tree/pytorch_1.0.0
**************************************************************************************************
*/

/*!
* Copyright (c) Facebook, Inc. and its affiliates.
* Modified by Bowen Cheng from https://github.com/fundamentalvision/Deformable-DETR
*/

#include <vector>
#include "cuda/ms_deform_im2col_cuda.cuh"

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda.h>
#include <cuda_runtime.h>


at::Tensor ms_deform_attn_cuda_forward(
    const at::Tensor &value, 
    const at::Tensor &spatial_shapes,
    const at::Tensor &level_start_index,
    const at::Tensor &sampling_loc,
    const at::Tensor &attn_weight,
    const int im2col_step)
{
    TORCH_CHECK(value.is_contiguous(), "value tensor has to be contiguous");
    TORCH_CHECK(spatial_shapes.is_contiguous(), "spatial_shapes tensor has to be contiguous");
    TORCH_CHECK(level_start_index.is_contiguous(), "level_start_index tensor has to be contiguous");
    TORCH_CHECK(sampling_loc.is_contiguous(), "sampling_loc tensor has to be contiguous");
    TORCH_CHECK(attn_weight.is_contiguous(), "attn_weight tensor has to be contiguous");

    TORCH_CHECK(value.is_cuda(), "value must be a CUDA tensor");
    TORCH_CHECK(spatial_shapes.is_cuda(), "spatial_shapes must be a CUDA tensor");
    TORCH_CHECK(level_start_index.is_cuda(), "level_start_index must be a CUDA tensor");
    TORCH_CHECK(sampling_loc.is_cuda(), "sampling_loc must be a CUDA tensor");
    TORCH_CHECK(attn_weight.is_cuda(), "attn_weight must be a CUDA tensor");

    const int batch = value.size(0);
    const int spatial_size = value.size(1);
    const int num_heads = value.size(2);
    const int channels = value.size(3);

    const int num_levels = spatial_shapes.size(0);

    const int num_query = sampling_loc.size(1);
    const int num_point = sampling_loc.size(4);

    const int im2col_step_ = std::min(batch, im2col_step);

    TORCH_CHECK(
        batch % im2col_step_ == 0,
        "batch(",
        batch,
        ") must divide im2col_step(",
        im2col_step_,
        ")"
    );
    
    auto dispatch_forward_impl = [&](auto scalar_type_tag) {
        using scalar_t = decltype(scalar_type_tag);

        auto output = at::zeros({batch, num_query, num_heads, channels}, value.options());

        const int batch_n = im2col_step_;
        auto output_n = output.view({batch / im2col_step_, batch_n, num_query, num_heads, channels});
        auto per_value_size = spatial_size * num_heads * channels;
        auto per_sample_loc_size = num_query * num_heads * num_levels * num_point * 2;
        auto per_attn_weight_size = num_query * num_heads * num_levels * num_point;
        for (int n = 0; n < batch / im2col_step_; ++n)
        {
            auto columns = output_n.select(0, n);
            ms_deformable_im2col_cuda(
                at::cuda::getCurrentCUDAStream().stream(),
                value.data_ptr<scalar_t>() + n * im2col_step_ * per_value_size,
                spatial_shapes.data_ptr<int64_t>(),
                level_start_index.data_ptr<int64_t>(),
                sampling_loc.data_ptr<scalar_t>() + n * im2col_step_ * per_sample_loc_size,
                attn_weight.data_ptr<scalar_t>() + n * im2col_step_ * per_attn_weight_size,
                batch_n, spatial_size, num_heads, channels, num_levels, num_query, num_point,
                columns.data_ptr<scalar_t>());
        }

        return output.view({batch, num_query, num_heads * channels});
    };

    switch (value.scalar_type())
    {
        case at::ScalarType::Float:
            return dispatch_forward_impl(float());
        case at::ScalarType::Double:
            return dispatch_forward_impl(double());
        case at::ScalarType::Half:
            return dispatch_forward_impl(at::Half());
        case at::ScalarType::BFloat16:
            return dispatch_forward_impl(at::BFloat16());
        default:
            TORCH_CHECK(false, "ms_deform_attn_cuda_forward supports only float, double, half, bfloat16");
    }

    throw std::runtime_error("Unsupported dtype for ms_deform_attn_cuda_forward");
}


std::vector<at::Tensor> ms_deform_attn_cuda_backward(
    const at::Tensor &value, 
    const at::Tensor &spatial_shapes,
    const at::Tensor &level_start_index,
    const at::Tensor &sampling_loc,
    const at::Tensor &attn_weight,
    const at::Tensor &grad_output,
    const int im2col_step)
{

    TORCH_CHECK(value.is_contiguous(), "value tensor has to be contiguous");
    TORCH_CHECK(spatial_shapes.is_contiguous(), "spatial_shapes tensor has to be contiguous");
    TORCH_CHECK(level_start_index.is_contiguous(), "level_start_index tensor has to be contiguous");
    TORCH_CHECK(sampling_loc.is_contiguous(), "sampling_loc tensor has to be contiguous");
    TORCH_CHECK(attn_weight.is_contiguous(), "attn_weight tensor has to be contiguous");
    TORCH_CHECK(grad_output.is_contiguous(), "grad_output tensor has to be contiguous");

    TORCH_CHECK(value.is_cuda(), "value must be a CUDA tensor");
    TORCH_CHECK(spatial_shapes.is_cuda(), "spatial_shapes must be a CUDA tensor");
    TORCH_CHECK(level_start_index.is_cuda(), "level_start_index must be a CUDA tensor");
    TORCH_CHECK(sampling_loc.is_cuda(), "sampling_loc must be a CUDA tensor");
    TORCH_CHECK(attn_weight.is_cuda(), "attn_weight must be a CUDA tensor");
    TORCH_CHECK(grad_output.is_cuda(), "grad_output must be a CUDA tensor");

    const int batch = value.size(0);
    const int spatial_size = value.size(1);
    const int num_heads = value.size(2);
    const int channels = value.size(3);

    const int num_levels = spatial_shapes.size(0);

    const int num_query = sampling_loc.size(1);
    const int num_point = sampling_loc.size(4);

    const int im2col_step_ = std::min(batch, im2col_step);

    TORCH_CHECK(
        batch % im2col_step_ == 0,
        "batch(",
        batch,
        ") must divide im2col_step(",
        im2col_step_,
        ")"
    );

    auto dispatch_backward_impl = [&](auto scalar_type_tag) {
        using scalar_t = decltype(scalar_type_tag);

        auto grad_value = at::zeros_like(value);
        auto grad_sampling_loc = at::zeros_like(sampling_loc);
        auto grad_attn_weight = at::zeros_like(attn_weight);

        const int batch_n = im2col_step_;
        auto per_value_size = spatial_size * num_heads * channels;
        auto per_sample_loc_size = num_query * num_heads * num_levels * num_point * 2;
        auto per_attn_weight_size = num_query * num_heads * num_levels * num_point;
        auto grad_output_n = grad_output.view({batch / im2col_step_, batch_n, num_query, num_heads, channels});

        for (int n = 0; n < batch / im2col_step_; ++n)
        {
            auto grad_output_g = grad_output_n.select(0, n);
            ms_deformable_col2im_cuda(
                at::cuda::getCurrentCUDAStream().stream(),
                grad_output_g.data_ptr<scalar_t>(),
                value.data_ptr<scalar_t>() + n * im2col_step_ * per_value_size,
                spatial_shapes.data_ptr<int64_t>(),
                level_start_index.data_ptr<int64_t>(),
                sampling_loc.data_ptr<scalar_t>() + n * im2col_step_ * per_sample_loc_size,
                attn_weight.data_ptr<scalar_t>() + n * im2col_step_ * per_attn_weight_size,
                batch_n, spatial_size, num_heads, channels, num_levels, num_query, num_point,
                grad_value.data_ptr<scalar_t>() + n * im2col_step_ * per_value_size,
                grad_sampling_loc.data_ptr<scalar_t>() + n * im2col_step_ * per_sample_loc_size,
                grad_attn_weight.data_ptr<scalar_t>() + n * im2col_step_ * per_attn_weight_size);
        }

        return std::vector<at::Tensor>({grad_value, grad_sampling_loc, grad_attn_weight});
    };

    switch (value.scalar_type())
    {
        case at::ScalarType::Float:
            return dispatch_backward_impl(float());
        case at::ScalarType::Double:
            return dispatch_backward_impl(double());
        case at::ScalarType::Half:
            return dispatch_backward_impl(at::Half());
        case at::ScalarType::BFloat16:
            return dispatch_backward_impl(at::BFloat16());
        default:
            TORCH_CHECK(false, "ms_deform_attn_cuda_backward supports only float, double, half, bfloat16");
    }

    throw std::runtime_error("Unsupported dtype for ms_deform_attn_cuda_backward");
}

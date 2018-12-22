
#include <random>
#include <cmath>
#include <limits>
#include <vector>
#include <iostream>
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/lib/core/errors.h"

using namespace tensorflow;

using shape_inference::InferenceContext;
using shape_inference::ShapeHandle;

REGISTER_OP("SSMatMulGrad")
    .Input("grad: float64")
    .Input("sparse: float64")
    .Input("column_ind: int32")
    .Output("output: float64")
    .SetShapeFn([](InferenceContext* c) {

      ShapeHandle mat_shape;
      c->WithRankAtLeast(c->input(1), 2, &mat_shape);
      c->set_output(0, mat_shape);

      return Status::OK();
    });

using namespace tensorflow;

class SSMatMulGradOp : public OpKernel {
    public:
        explicit SSMatMulGradOp(OpKernelConstruction* context) : OpKernel(context) {}

        void row_add(typename TTypes<double, 1>::ConstTensor& matrix, typename TTypes<int, 1>::ConstTensor& indices, \
                     int row_i_base, int row_j_base, \
                     double grad_ij, typename TTypes<double, 1>::Tensor& output, int64 ncols) {

            int k1 = row_i_base; 
            int k2 = row_j_base; 

            while (true) {

                if (indices(k1) == indices(k2)) {
                    output(k1) += matrix(k2) * grad_ij; 

                    //std::cout << output(k1) << ' ' << matrix(k2) << '\n'; 

                    k1++;
                    k2++; 

                } else if (indices(k1) < indices(k2)) {
                    k1++; 
                } else if (indices(k1) > indices(k2)) {
                    k2++;
                }

                if (k1 >= row_i_base + ncols || k2 >= row_j_base + ncols)
                    break;
            }
        }

        void dense_sparse_product(typename TTypes<double, 1>::ConstTensor& matrix, \
                              typename TTypes<int, 1>::ConstTensor& indices, \
                              int slice_base1, \
                              typename TTypes<double, 1>::ConstTensor& grad, \
                              int slice_base2, \
                              typename TTypes<double, 1>::Tensor& output, \
                              int64 irow, \
                              int64 nrows, \
                              int64 ncols) {
        
        int row_i_base = slice_base1 + irow * ncols;

        for (int jrow = 0; jrow < nrows; jrow++) {
            
            int row_j_base = slice_base1 + jrow * ncols;

            double grad_ij = 0;

            if (irow == jrow)
                grad_ij = 2.0 * grad(slice_base2 + irow * nrows + jrow);
            else
                grad_ij = grad(slice_base2 + irow * nrows + jrow) + grad(slice_base2 + jrow * nrows + irow);


            //std::cout << "Working on rows: "  << irow << ' ' << jrow << ' ' <<  grad_ij <<  '\n'; 

            row_add(matrix, indices, row_i_base, row_j_base, grad_ij, output, ncols); 

        }
        return;
    }
                             

    void Compute(OpKernelContext* context) override {
        // Grab the input tensor
        const Tensor& grad_tensor = context->input(0);
        const Tensor& mat_tensor = context->input(1);
        const Tensor& ind_tensor = context->input(2);

        const TensorShape& mat_shape = mat_tensor.shape(); 
        const TensorShape& ind_shape = ind_tensor.shape(); 

        //make shape for the output
        TensorShape output_shape(mat_shape); 
        Tensor* p_tensor = NULL;
        OP_REQUIRES_OK(context, context->allocate_output(0, output_shape,
                                                         &p_tensor));
        Tensor& output_tensor(*p_tensor);

        auto grad = grad_tensor.flat<double>();
        auto matrix = mat_tensor.flat<double>();
        auto indices = ind_tensor.flat<int>();
        auto output = output_tensor.flat<double>();
        

        int64 nrows = mat_shape.dim_size(mat_shape.dims() - 2);
        int64 ncols = mat_shape.dim_size(mat_shape.dims() - 1);
        int64 num_elems = mat_tensor.NumElements();
        int64 num_slices = num_elems / nrows / ncols;

        int64 slice_size = nrows * ncols;
        int64 num_iter = num_slices * nrows;

        for (int64 itr = 0; itr < num_elems; itr++) {
            output(itr) = 0;
        }

        for (int64 it = 0; it < num_iter; it++) {
            
            int64 islice = it / nrows;
            int64 irow = it % nrows;

            int64 slice_base1 = islice * (nrows * ncols);
            int64 slice_base2 = islice * (nrows * nrows);

            dense_sparse_product(matrix, indices, slice_base1, grad, slice_base2, output, irow, nrows, ncols);

        }
    }
};

REGISTER_KERNEL_BUILDER(Name("SSMatMulGrad").Device(DEVICE_CPU), SSMatMulGradOp);











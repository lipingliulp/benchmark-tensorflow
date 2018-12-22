
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
using shape_inference::DimensionHandle;

REGISTER_OP("SSMatMul")
    .Input("sparse: float64")
    .Input("column_ind: int32")
    .Output("output: float64")
    .SetShapeFn([](InferenceContext* c) {

      ShapeHandle mat_shape;
      ShapeHandle ind_shape;

      c->WithRankAtLeast(c->input(0), 2, &mat_shape);
      c->WithRankAtLeast(c->input(1), 2, &ind_shape);

      if (!c->RankKnown(mat_shape)) {
          c->set_output(0, c->UnknownShape());
          return Status::OK();
      }

      ShapeHandle output_shape;
      //c->ReplaceDim(mat_shape, -1, c->Dim(mat_shape, -2), &output_shape);

      ShapeHandle all_but_last;
      TF_RETURN_IF_ERROR(c->Subshape(mat_shape, 0, -1, &all_but_last));

      DimensionHandle row_dim = c->Dim(mat_shape, -2);
      ShapeHandle num_rows = c->Vector(row_dim);
      TF_RETURN_IF_ERROR(c->Concatenate(all_but_last, num_rows, &output_shape));

      c->set_output(0, output_shape);

      return Status::OK();
    });

using namespace tensorflow;

class SSMatMulOp : public OpKernel {
    public:
        explicit SSMatMulOp(OpKernelConstruction* context) : OpKernel(context) {}

        void sparse_inner_product(typename TTypes<double, 1>::ConstTensor& matrix, \
                              typename TTypes<int, 1>::ConstTensor& indices, \
                              int base1, \
                              int base2, \
                              typename TTypes<double, 1>::Tensor& output, \
                              int pointer, \
                              int64 ncols) {

        int i = base1; 
        int j = base2; 

        output(pointer) = 0;
        while (true) {

            if (indices(i) == indices(j)) {
                output(pointer) += matrix(i) * matrix(j); 
                i++;
                j++; 
            } else if (indices(i) < indices(j)) {
                i++; 
            } else if (indices(i) > indices(j)) {
                j++;
            }

            if (i >= base1 + ncols || j >= base2 + ncols)
                break;
        }
        
        return;
    }
                             

    void Compute(OpKernelContext* context) override {
        // Grab the input tensor
        const Tensor& mat_tensor = context->input(0);
        const Tensor& ind_tensor = context->input(1);

        const TensorShape& mat_shape = mat_tensor.shape(); 
        const TensorShape& ind_shape = ind_tensor.shape(); 

        for (int i = 0; i < mat_shape.dims(); i++) { 
          if (mat_shape.dim_size(i) != ind_shape.dim_size(i)) {
            context->SetStatus(errors::InvalidArgument("param and indices have different shapes"));
          }
        }

        //make shape for the output
        TensorShape output_shape(mat_shape); 
        output_shape.RemoveLastDims(1);
        output_shape.AddDim(mat_shape.dim_size(mat_shape.dims() - 2));

        Tensor* p_tensor = NULL;
        OP_REQUIRES_OK(context, context->allocate_output(0, output_shape,
                                                         &p_tensor));
        Tensor& output_tensor(*p_tensor);

        auto matrix = mat_tensor.flat<double>();
        auto indices = ind_tensor.flat<int>();
        auto output = output_tensor.flat<double>();
        

        int64 nrows = mat_shape.dim_size(mat_shape.dims() - 2);
        int64 ncols = mat_shape.dim_size(mat_shape.dims() - 1);
        int64 num_slices = mat_tensor.NumElements() / nrows / ncols;

        int64 num_output_elems = output_tensor.NumElements();
        int64 slice_size = nrows * nrows;

        for (int64 it = 0; it < num_output_elems; it++) {

            
            int64 i_slice = it / slice_size;
            int64 i_in_slice = it % slice_size;

            int64 i_row1 = i_in_slice / nrows;
            int64 i_row2 = i_in_slice % nrows;


            //std::cout << it << ' ' << i_slice << ' ' << i_in_slice << ' ' << i_row1 << ' ' << i_row2 << '\n';

            int64 base1 = i_slice * (nrows * ncols) + i_row1 * ncols;
            int64 base2 = i_slice * (nrows * ncols) + i_row2 * ncols;
            int64 pointer = it;

            sparse_inner_product(matrix, indices, base1, base2, output, pointer, ncols); 
        }

    }
};

REGISTER_KERNEL_BUILDER(Name("SSMatMul").Device(DEVICE_CPU), SSMatMulOp);











/*
// @HEADER
// ***********************************************************************
// 
// Moocho: Multi-functional Object-Oriented arCHitecture for Optimization
//                  Copyright (2003) Sandia Corporation
// 
// Under terms of Contract DE-AC04-94AL85000, there is a non-exclusive
// license for use of this work by or on behalf of the U.S. Government.
// 
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are
// met:
//
// 1. Redistributions of source code must retain the above copyright
// notice, this list of conditions and the following disclaimer.
//
// 2. Redistributions in binary form must reproduce the above copyright
// notice, this list of conditions and the following disclaimer in the
// documentation and/or other materials provided with the distribution.
//
// 3. Neither the name of the Corporation nor the names of the
// contributors may be used to endorse or promote products derived from
// this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY SANDIA CORPORATION "AS IS" AND ANY
// EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
// PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL SANDIA CORPORATION OR THE
// CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
// EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
// PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
// PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
// LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
// NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
// SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//
// Questions? Contact Roscoe A. Bartlett (rabartl@sandia.gov) 
// 
// ***********************************************************************
// @HEADER
*/

#include "RTOp_reduct_sum_values.h"

int RTOp_reduct_sum_values(
  const struct RTOp_RTOp_vtbl_t* vtbl, const void* obj_data
  , RTOp_ReductTarget in_targ_obj, RTOp_ReductTarget inout_targ_obj )
{
  int num_values, k;
#ifdef RTOp_DEBUG
  assert(obj_data);
#endif
  num_values = *(RTOp_index_type*)obj_data;
  /* inout_dot_prod += in_dot_prod */
  for( k = 0; k < num_values; ++k )
    ((RTOp_value_type*)inout_targ_obj)[k] += ((RTOp_value_type*)in_targ_obj)[k];
  return 0;
}

static void CALL_API external_reduct_op( void* in_targ_array, void* inout_targ_array
  , int* len, RTOp_Datatype* datatype )
{
  /* inout_dot_prod += in_dot_prod */
  RTOp_index_type
    num_values   = *(RTOp_value_type*)in_targ_array; /* num_values is first size member */
  RTOp_value_type /* index past the size members */
    *in_targs    = (RTOp_value_type*)in_targ_array    + 3,
    *inout_targs = (RTOp_value_type*)inout_targ_array + 3;
  int i, k;
  for( i = 0; i < *len; ++i, inout_targs += (3 + num_values), in_targs += (3 + num_values) ) {
    for( k = 0; k < num_values; ++k )
      inout_targs[k] += in_targs[k];
  }
}

int RTOp_get_reduct_sum_values_op(
  const struct RTOp_RTOp_vtbl_t* vtbl, const void* obj_data
  , RTOp_reduct_op_func_ptr_t* reduct_op_func_ptr )
{
  *reduct_op_func_ptr = external_reduct_op;
  return 0;
}

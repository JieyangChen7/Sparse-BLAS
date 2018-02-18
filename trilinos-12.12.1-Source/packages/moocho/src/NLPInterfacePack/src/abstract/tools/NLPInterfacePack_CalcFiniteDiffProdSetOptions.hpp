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

#ifndef CALC_FINITE_DIFF_PROD_SET_OPTIONS_H
#define CALC_FINITE_DIFF_PROD_SET_OPTIONS_H

#include "NLPInterfacePack_CalcFiniteDiffProd.hpp"
#include "OptionsFromStreamPack_SetOptionsFromStreamNode.hpp"
#include "OptionsFromStreamPack_SetOptionsToTargetBase.hpp"

namespace NLPInterfacePack {

/** \brief Set options for \c CalcFiniteDiffProd from an
 * \c OptionsFromStream object.
 *
 * The default options group name is CalcFiniteDiffProd.
 *
 * The options group is:
 *
 \verbatim
 options_group CalcFiniteDiffProdSetOptions {
 *    fd_method_order = FD_ORDER_ONE;
 *    fd_method_order = FD_ORDER_TWO;
 *    fd_method_order = FD_ORDER_TWO_CENTRAL;
 *    fd_method_order = FD_ORDER_TWO_AUTO;
 *    fd_method_order = FD_ORDER_FOUR;
 *    fd_method_order = FD_ORDER_FOUR_CENTRAL;
 *    fd_method_order = FD_ORDER_FOUR_AUTO; *** (Default)
 *    fd_step_select = FD_STEP_ABSOLUTE; *** (Default)
 *    fd_step_select = FD_STEP_RELATIVE;
 *    fd_step_size = -1.0; *** (default)
 *    fd_step_size_min = -1.0; *** (default)
 *    fd_step_size_f = -1.0; *** (default)
 *    fd_step_size_c = -1.0; *** (default)
 }
 \endverbatim
 */
class CalcFiniteDiffProdSetOptions
  : public OptionsFromStreamPack::SetOptionsFromStreamNode 
    , public OptionsFromStreamPack::SetOptionsToTargetBase<
      CalcFiniteDiffProd >
{
public:

  /** \brief . */
  CalcFiniteDiffProdSetOptions(
    CalcFiniteDiffProd* target = 0
    ,const char opt_grp_name[] = "CalcFiniteDiffProd" );
  
protected:

  /// Overridden from SetOptionsFromStreamNode
  void setOption( int option_num, const std::string& option_value );

};	// end class CalcFiniteDiffProdSetOptions

}	// end namespace NLPInterfacePack

#endif	// CALC_FINITE_DIFF_PROD_SET_OPTIONS_H

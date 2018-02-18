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

#include <assert.h>
#include <math.h>

#include "NLPInterfacePack_NLPFirstDerivTesterSetOptions.hpp"

// Define the options
namespace {

  const int local_num_options = 4;

  enum local_EOptions {
    FD_TESTING_METHOD
    ,NUM_FD_DIRECTIONS
      ,WARNING_TOL
      ,ERROR_TOL
  };

  const char* local_SOptions[local_num_options]	= {
      "fd_testing_method"
      ,"num_fd_directions"
      ,"warning_tol"
      ,"error_tol"
  };

}

namespace NLPInterfacePack {

NLPFirstDerivTesterSetOptions::NLPFirstDerivTesterSetOptions(
        NLPFirstDerivTester* target
      , const char opt_grp_name[] )
  :	OptionsFromStreamPack::SetOptionsFromStreamNode(
        opt_grp_name, local_num_options, local_SOptions )
    , OptionsFromStreamPack::SetOptionsToTargetBase<
      NLPFirstDerivTester >( target )
{}

void NLPFirstDerivTesterSetOptions::setOption(
  int option_num, const std::string& option_value )
{
  typedef NLPFirstDerivTester target_t;
  switch( (local_EOptions)option_num ) {
      case FD_TESTING_METHOD:
    {
      const std::string &option = option_value.c_str();
      if( option == "FD_COMPUTE_ALL" )
        target().fd_testing_method( target_t::FD_COMPUTE_ALL );
      else if( option == "FD_DIRECTIONAL" )
        target().fd_testing_method( target_t::FD_DIRECTIONAL );
      else
        throw std::invalid_argument( "Error, incorrect value for "
          "\"fd_testing_method\".  Only the options "
          "FD_COMPUTE_ALL and FD_DIRECTIONAL are available" );
      break;
    }
      case NUM_FD_DIRECTIONS:
      target().num_fd_directions(std::atoi(option_value.c_str()));
      break;
      case WARNING_TOL:
      target().warning_tol(std::fabs(std::atof(option_value.c_str())));
      break;
      case ERROR_TOL:
      target().error_tol(std::fabs(std::atof(option_value.c_str())));
      break;
    default:
      TEUCHOS_TEST_FOR_EXCEPT(true);	// Local error only?
  }
}

}	// end namespace NLPInterfacePack

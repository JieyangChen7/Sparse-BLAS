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

#ifndef MERIT_FUNC_PENALTY_PARAMS_H
#define MERIT_FUNC_PENALTY_PARAMS_H

#include "ConstrainedOptPack_Types.hpp"
#include "AbstractLinAlgPack_VectorSpace.hpp"

namespace ConstrainedOptPack {

/** \brief This class provides interface for setting and retrieving a penalty parameter
  * that many merit functions use {abstract}.
  */
class MeritFuncPenaltyParams {
public:

  /** \brief . */
  class CanNotResize : public std::logic_error
  {public: CanNotResize(const std::string& what_arg) : std::logic_error(what_arg) {}};

  /** \brief . */
  virtual ~MeritFuncPenaltyParams() {}

  /** @name To be overridden by subclasses */
  //@{

  /** \brief Set the vector space for \c to use for the penalty parameters.
    */
  virtual void set_space_c( const VectorSpace::space_ptr_t& space_c ) = 0;

  /// Get the vector of penalty parameters for setting them
  virtual VectorMutable& set_mu() = 0;

  /// Get the vector of penalty parameters for viewing them
  virtual const Vector& get_mu() const = 0;

  //@}

};	// end class MeritFuncPenaltyParams

}	// end namespace ConstrainedOptPack

#endif	// MERIT_FUNC_PENALTY_PARAMS_H

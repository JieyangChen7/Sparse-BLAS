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

#ifndef POST_PROCESS_BARRIER_LINE_SEARCH_STEP_H
#define POST_PROCESS_BARRIER_LINE_SEARCH_STEP_H

#include "MoochoPack_Types.hpp"
#include "IterationPack_AlgorithmStep.hpp"
#include "Teuchos_StandardCompositionMacros.hpp"
#include "Teuchos_StandardMemberCompositionMacros.hpp"

#include "OptionsFromStreamPack_SetOptionsFromStreamNode.hpp"
#include "OptionsFromStreamPack_SetOptionsToTargetBase.hpp"

namespace MoochoPack {

/** \brief Fraction to boundary rule for calculating alpha max
 *   
 *
 * This class updates alpha_vl, alpha_vu, and alpha
 *  and x_kp1, Vl_kp1, and Vu_kp1
 *
 */

class PostProcessBarrierLineSearch_Step
  : public IterationPack::AlgorithmStep // doxygen needs full path
  {
  public:

    /** @name Constructors / initializers */
    //@{

    /** \brief Constructor.
     */
    PostProcessBarrierLineSearch_Step(
      Teuchos::RCP<NLPInterfacePack::NLPBarrier> barrier_nlp
      );
    //@}

    /** @name Overridden from AlgorithmStep */
    //@{
    /** \brief . */
    bool do_step(Algorithm& algo, poss_type step_poss, IterationPack::EDoStepType type
           , poss_type assoc_step_poss);
    
    
    void print_step( const IterationPack::Algorithm& algo, poss_type step_poss, IterationPack::EDoStepType type
             , poss_type assoc_step_poss, std::ostream& out, const std::string& leading_str ) const;
    //@}

  private:
    Teuchos::RCP<NLPInterfacePack::NLPBarrier> barrier_nlp_;

  }; // end class PostProcessBarrierLineSearch_Step


} // end namespace MoochoPack

#endif // #if !defined POST_PROCESS_BARRIER_LINE_SEARCH_STEP_H

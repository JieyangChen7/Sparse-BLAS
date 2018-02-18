#ifndef CAZTECOO_CPP_HPP
#define CAZTECOO_CPP_HPP

/*! @HEADER */
/*
************************************************************************

                CTrilinos:  C interface to Trilinos
                Copyright (2009) Sandia Corporation

Under terms of Contract DE-AC04-94AL85000, there is a non-exclusive
license for use of this work by or on behalf of the U.S. Government.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are
met:

1. Redistributions of source code must retain the above copyright
notice, this list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright
notice, this list of conditions and the following disclaimer in the
documentation and/or other materials provided with the distribution.

3. Neither the name of the Corporation nor the names of the
contributors may be used to endorse or promote products derived from
this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY SANDIA CORPORATION "AS IS" AND ANY
EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL SANDIA CORPORATION OR THE
CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

Questions? Contact M. Nicole Lemaster (mnlemas@sandia.gov)

************************************************************************
*/
/*! @HEADER */


#include "CTrilinos_config.h"


#ifdef HAVE_CTRILINOS_AZTECOO


#include "CTrilinos_enums.h"
#include "Teuchos_RCP.hpp"
#include "AztecOO.h"


namespace CAztecOO {


using Teuchos::RCP;


/*! get AztecOO from non-const table using CT_AztecOO_ID */
const RCP<AztecOO>
getAztecOO( CT_AztecOO_ID_t id );

/*! get AztecOO from non-const table using CTrilinos_Universal_ID_t */
const RCP<AztecOO>
getAztecOO( CTrilinos_Universal_ID_t id );

/*! get const AztecOO from either the const or non-const table
 * using CT_AztecOO_ID */
const RCP<const AztecOO>
getConstAztecOO( CT_AztecOO_ID_t id );

/*! get const AztecOO from either the const or non-const table
 * using CTrilinos_Universal_ID_t */
const RCP<const AztecOO>
getConstAztecOO( CTrilinos_Universal_ID_t id );

/*! store AztecOO (owned) in non-const table */
CT_AztecOO_ID_t
storeNewAztecOO( AztecOO *pobj );

/*! store AztecOO in non-const table */
CT_AztecOO_ID_t
storeAztecOO( AztecOO *pobj );

/*! store const AztecOO in const table */
CT_AztecOO_ID_t
storeConstAztecOO( const AztecOO *pobj );

/* remove AztecOO from table using CT_AztecOO_ID */
void
removeAztecOO( CT_AztecOO_ID_t *id );

/* remove AztecOO from table using CTrilinos_Universal_ID_t */
void
removeAztecOO( CTrilinos_Universal_ID_t *aid );

/* purge AztecOO table */
void
purgeAztecOO(  );

} // namespace CAztecOO


#endif /* HAVE_CTRILINOS_AZTECOO */
#endif // CAZTECOO_CPP_HPP



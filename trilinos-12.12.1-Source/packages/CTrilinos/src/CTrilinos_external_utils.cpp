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

#include "CTrilinos_external_utils.h"
#include "CTrilinos_test_utils.hpp"
#include "CTrilinos_exceptions.hpp"
#include "CEpetra_MpiComm.h"


#ifdef HAVE_MPI
#include "mpi.h"
#endif


extern "C" {


#ifdef HAVE_MPI

/*! Create an Epetra_MpiComm from Fortran */
CT_Epetra_MpiComm_ID_t Epetra_MpiComm_Fortran_Create ( int fcomm )
{
    MPI_Fint mfcomm = (MPI_Fint) fcomm;
    MPI_Comm ccomm = MPI_Comm_f2c(mfcomm);

    /* duplicate the communicator so that we won't get any cross-talk
     * from the application */
    MPI_Comm dupcomm;
    int ret = MPI_Comm_dup(ccomm, &dupcomm);
    if (ret != MPI_SUCCESS)
        throw CTrilinos::CTrilinosMiscException("Error on MPI_Comm_dup");

    return Epetra_MpiComm_Create(dupcomm);
}

#endif /* HAVE_MPI */

/* Clear the tables between tests */
void CTrilinos_CleanSlate (  )
{
  CTrilinos::purgeAllTables();
}

} // extern "C"

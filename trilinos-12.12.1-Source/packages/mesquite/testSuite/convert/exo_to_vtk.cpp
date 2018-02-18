/* ***************************************************************** 
    MESQUITE -- The Mesh Quality Improvement Toolkit

    Copyright 2004 Sandia Corporation and Argonne National
    Laboratory.  Under the terms of Contract DE-AC04-94AL85000 
    with Sandia Corporation, the U.S. Government retains certain 
    rights in this software.

    This library is free software; you can redistribute it and/or
    modify it under the terms of the GNU Lesser General Public
    License as published by the Free Software Foundation; either
    version 2.1 of the License, or (at your option) any later version.

    This library is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
    Lesser General Public License for more details.

    You should have received a copy of the GNU Lesser General Public License 
    (lgpl.txt) along with this library; if not, write to the Free Software
    Foundation, Inc., 51 Franklin St, Fifth Floor, Boston, MA 02110-1301, USA
 
    diachin2@llnl.gov, djmelan@sandia.gov, mbrewer@sandia.gov, 
    pknupp@sandia.gov, tleurent@mcs.anl.gov, tmunson@mcs.anl.gov      
   
  ***************************************************************** */
// -*- Mode : c++; tab-width: 3; c-tab-always-indent: t; indent-tabs-mode: nil; c-basic-offset: 3 -*-
//
//   SUMMARY: 
//     USAGE:
//
// ORIG-DATE: 19-Feb-02 at 10:57:52
//  LAST-MOD: 23-May-03 at 18:04:38 by Thomas Leurent
//
//
// DESCRIPTION:
// ============
/*! \file main.cpp

describe main.cpp here

 */
// DESCRIP-END.
//

#include <iostream>
using std::cout;
using std::endl;
#include <cstdlib>

#include "Mesquite.hpp"
#include "Mesquite_MeshImpl.hpp"
#include "Mesquite_MsqError.hpp"

using namespace Mesquite;

int main(int argc, char* argv[])
{
  Mesquite::MsqPrintError err(cout);
  char in_file_name[256];
  char out_file_name[256];
  
  // command line arguments
  if (argc!=3){
    cout << "Input meshfile name needed as first argument.\n"
      "Output meshfile name needed as second argument.\n" << endl;
    return -1;
  }
  else{
    cout << " given 2 command line arguments.\n";
    strcpy(in_file_name, argv[1]);
    strcpy(out_file_name, argv[2]);
  }

  Mesquite::MeshImpl *mesh = new Mesquite::MeshImpl;
  cout<<"\nReading Exodus file.\n";
  mesh->read_exodus(in_file_name, err);
  if (err) return 1;
  cout<<"Writing VTK file.\n";
  mesh->write_vtk(out_file_name,err);
  if (err) return 1;

  return 0;
}

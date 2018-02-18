/* ***************************************************************** 
    MESQUITE -- The Mesh Quality Improvement Toolkit

    Copyright 2006 Sandia National Laboratories.  Developed at the
    University of Wisconsin--Madison under SNL contract number
    624796.  The U.S. Government and the University of Wisconsin
    retian certain rights to this software.

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
 
    (2006) kraftche@cae.wisc.edu
   
  ***************************************************************** */


/** \file EdgeLengthMetricTest.cpp
 *  \brief unit tests for EdgeLengthQualityMetric
 *  \author Jason Kraftcheck 
 */

#include "Mesquite.hpp"
#include "Mesquite_EdgeLengthQualityMetric.hpp"
#include "Mesquite_cppunit/extensions/HelperMacros.h"
#include "QualityMetricTester.hpp"

using namespace Mesquite;

class EdgeLengthMetricTest : public CppUnit::TestFixture
{
private:  

  CPPUNIT_TEST_SUITE(EdgeLengthMetricTest);

  CPPUNIT_TEST (test_supported_types);
  CPPUNIT_TEST (test_ideal_element_eval);
  CPPUNIT_TEST (test_inverted_elements);
  CPPUNIT_TEST (test_degenerate_elements);
  CPPUNIT_TEST (test_get_evaluations);
  CPPUNIT_TEST (test_get_vertex_indices);
  CPPUNIT_TEST (test_get_fixed_indices);
  CPPUNIT_TEST (test_eval_with_indices);
  CPPUNIT_TEST (test_location_invariant);
  CPPUNIT_TEST (test_orient_invariant);
  
  CPPUNIT_TEST_SUITE_END();
  
  EdgeLengthQualityMetric mMetric;
  QualityMetricTester tester;
  
public:

  EdgeLengthMetricTest() 
    : tester(QualityMetricTester::ALL_FE_EXCEPT_SEPTAHEDRON)
    { }

  void test_supported_types()
    { tester.test_supported_element_types( &mMetric ); }
    
  void test_ideal_element_eval()
  {
    tester.test_evaluate_unit_edge_tris_about_vertex( &mMetric, 6.0 );
    tester.test_evaluate_unit_quads_about_vertex( &mMetric, 4.0 );
    tester.test_evaluate_unit_hexes_about_vertex( &mMetric, 6.0 );
  }
  
  void test_inverted_elements()
    { tester.test_evaluate_inverted_element( &mMetric, true ); }
    
  void test_degenerate_elements()
    { tester.test_evaluate_degenerate_element( &mMetric, true ); }
    
  void test_get_evaluations()
    { tester.test_get_vertex_evaluations( &mMetric ); }
    
  void test_get_vertex_indices()
    { tester.test_get_vertex_indices( &mMetric ); }
  
  void test_get_fixed_indices()
    { tester.test_get_indices_fixed( &mMetric ); }
  
  void test_eval_with_indices()
    { tester.compare_eval_and_eval_with_indices( &mMetric ); }
  
  void test_location_invariant()
  {
    tester.test_location_invariant( &mMetric );
    tester.test_grad_location_invariant( &mMetric );
  }
  
  void test_orient_invariant()
  {
    tester.test_orient_invariant( &mMetric );
    tester.test_grad_orient_invariant( &mMetric );
  }
};


CPPUNIT_TEST_SUITE_NAMED_REGISTRATION(EdgeLengthMetricTest, "EdgeLengthQualityMetricTest");
CPPUNIT_TEST_SUITE_NAMED_REGISTRATION(EdgeLengthMetricTest, "Unit");

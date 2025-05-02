#!/usr/bin/env python
import numpy as np
import os
import shutil
import sys
import unittest
from pathlib import Path

from dpgen.generator.run import (
    _read_plumed_colvar_file,
    _select_by_plumed_colvar,
)

class TestPlumedColvar(unittest.TestCase):
    def setUp(self):
        # Create a temporary directory for test data
        self.test_dir = Path("tmp.plumed_colvar_test")
        self.test_dir.mkdir(exist_ok=True)
        
        # Create a mock task directory
        self.task_dir = self.test_dir / "task.000.000000"
        self.task_dir.mkdir(exist_ok=True)
        
        # Create a mock model_devi.out file
        model_devi_data = np.array([
            [1, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06],  # frame 1: within thresholds (candidate)
            [2, 0.02, 0.03, 0.04, 0.10, 0.06, 0.07],  # frame 2: force too high (failed)
            [3, 0.002, 0.003, 0.004, 0.005, 0.006, 0.007],  # frame 3: all below low thresholds (accurate)
            [4, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07],  # frame 4: within thresholds (candidate)
            [5, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06],  # frame 5: within thresholds (candidate)
            [6, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06],  # frame 6: within thresholds (candidate)
            [7, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06],  # frame 7: within thresholds (candidate)
            [8, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06],  # frame 8: within thresholds (candidate)
            [9, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06],  # frame 9: within thresholds (candidate)
            [10, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06], # frame 10: within thresholds (candidate)
        ])
        np.savetxt(self.task_dir / "model_devi.out", model_devi_data, fmt="%f")
        
        # Create a mock COLVAR file with multiple columns
        # Format: time, cv1, cv2, cv3
        colvar_data = np.array([
            [1, 1.1, 5.5, 0.2],  # frame 1: cv2 within range [5, 10), cv3 within range [0, 0.5)
            [2, 2.2, 4.5, 0.3],  # frame 2: cv2 below range, cv3 within range
            [3, 3.3, 7.8, 0.4],  # frame 3: cv2 within range, cv3 within range
            [4, 4.4, 10.5, 0.6], # frame 4: cv2 above range, cv3 above range
            [5, 5.5, 8.0, 0.6],  # frame 5: cv2 within range, cv3 above range
            [6, 6.6, 2.5, 0.2],  # frame 6: cv2 below 5 but in multi-range, cv3 in range
            [7, 7.7, 9.0, 0.3],  # frame 7: cv2 within range, cv3 within range
            [8, 8.8, 11.0, 0.4], # frame 8: cv2 outside range, cv3 within range
            [9, 9.9, 1.5, 0.1],  # frame 9: cv2 below 3 and in multi-range, cv3 in range
            [10, 10.0, 9.5, 0.2], # frame 10: cv2 within range, cv3 within range
        ])
        
        with open(self.task_dir / "COLVAR", "w") as f:
            f.write("# time cv1 cv2 cv3\n")
            np.savetxt(f, colvar_data, fmt="%f")

    def tearDown(self):
        # Remove temporary directory
        shutil.rmtree(self.test_dir)

    def test_read_plumed_colvar_file_single_column(self):
        # Test reading COLVAR file with a single column
        colvar = _read_plumed_colvar_file(str(self.task_dir), model_devi_colvar_columns=1)
        
        # Check if data was loaded correctly
        self.assertIsNotNone(colvar)
        self.assertEqual(colvar.shape, (10, 2))  # 10 frames, 2 columns (time + 1 CV)
        self.assertEqual(colvar[0, 0], 1)  # First frame number
        self.assertEqual(colvar[0, 1], 5.5)  # First CV value (column 2 in the file)

    def test_read_plumed_colvar_file_multiple_columns(self):
        # Test reading COLVAR file with multiple columns
        colvar = _read_plumed_colvar_file(str(self.task_dir), model_devi_colvar_columns=[1, 2])
        
        # Check if data was loaded correctly
        self.assertIsNotNone(colvar)
        self.assertEqual(colvar.shape, (10, 3))  # 10 frames, 3 columns (time + 2 CVs)
        self.assertEqual(colvar[0, 0], 1)      # First frame number
        self.assertEqual(colvar[0, 1], 5.5)    # First CV value (column 2 in the file)
        self.assertEqual(colvar[0, 2], 0.2)    # Second CV value (column 3 in the file)
        
    def test_select_by_plumed_colvar_single_column(self):
        # Test selection with both model deviation and a single colvar criterion
        fp_rest_accurate, fp_candidate, fp_rest_failed, counter = _select_by_plumed_colvar(
            [str(self.task_dir)],  # modd_system_task
            0.01,  # f_trust_lo
            0.08,  # f_trust_hi
            0.01,  # v_trust_lo
            0.05,  # v_trust_hi
            5.0,   # colvar_lo
            10.0,  # colvar_hi
            1,     # colvar_column (second column in COLVAR, 0-indexed)
            None,  # cluster_cutoff
            "lammps",  # model_devi_engine
            0,     # model_devi_skip
            False, # model_devi_f_avg_relative
            False, # model_devi_merge_traj
            True,  # detailed_report_make_fp
        )
        
        # Check selection results
        # Frames with cv2 within [5, 10): 1, 3, 5, 7, 10
        # Frames with forces within thresholds: 1, 3, 4, 5, 6, 7, 8, 9, 10
        # Frames with forces too high: 2
        
        # Check candidate frames - force OK and CV in range
        self.assertEqual(len(fp_candidate), 5)
        candidate_frames = sorted([x[1] for x in fp_candidate])
        self.assertEqual(candidate_frames, [1, 5, 7, 10])
        
        # Check accurate frames - force below threshold and CV in range
        self.assertEqual(len(fp_rest_accurate), 1)
        self.assertEqual(fp_rest_accurate[0][1], 3)  # Frame 3
        
        # Check counter
        self.assertEqual(counter["candidate"], 5)
        self.assertEqual(counter["accurate"], 1)

    def test_select_by_plumed_colvar_multiple_columns(self):
        # Test selection with both model deviation and multiple colvar criteria
        fp_rest_accurate, fp_candidate, fp_rest_failed, counter = _select_by_plumed_colvar(
            [str(self.task_dir)],  # modd_system_task
            0.01,  # f_trust_lo
            0.08,  # f_trust_hi
            0.01,  # v_trust_lo
            0.05,  # v_trust_hi
            [5.0, 0.0],   # colvar_lo for columns 1 and 2
            [10.0, 0.5],  # colvar_hi for columns 1 and 2
            [1, 2],       # colvar_columns (second and third columns in COLVAR, 0-indexed)
            None,  # cluster_cutoff
            "lammps",  # model_devi_engine
            0,     # model_devi_skip
            False, # model_devi_f_avg_relative
            False, # model_devi_merge_traj
            True,  # detailed_report_make_fp
        )
        
        # Check selection results - force OK and both CVs in range
        self.assertEqual(len(fp_candidate), 4)
        candidate_frames = sorted([x[1] for x in fp_candidate])
        self.assertEqual(candidate_frames, [1, 7, 10])
        
        # Check accurate frames - force below threshold and both CVs in range
        self.assertEqual(len(fp_rest_accurate), 1)
        self.assertEqual(fp_rest_accurate[0][1], 3)  # Frame 3
        
        # Check counter
        self.assertEqual(counter["candidate"], 4)
        self.assertEqual(counter["accurate"], 1)

    def test_select_by_plumed_colvar_multi_range(self):
        # Test selection with multi-range CV criteria
        fp_rest_accurate, fp_candidate, fp_rest_failed, counter = _select_by_plumed_colvar(
            [str(self.task_dir)],  # modd_system_task
            0.01,  # f_trust_lo
            0.08,  # f_trust_hi
            0.01,  # v_trust_lo
            0.05,  # v_trust_hi
            [[0.0, 8.0], [0.0]],   # colvar_lo for columns 1: two ranges [0-3) and [8-10), column 2: one range [0-0.5)
            [[3.0, 10.0], [0.5]],  # colvar_hi for columns 1 and 2
            [1, 2],       # colvar_columns (second and third columns in COLVAR, 0-indexed)
            None,  # cluster_cutoff
            "lammps",  # model_devi_engine
            0,     # model_devi_skip
            False, # model_devi_f_avg_relative
            False, # model_devi_merge_traj
            True,  # detailed_report_make_fp
        )
        
        # Check selection results - should select frames with cv2 in [0-3) OR [8-10) AND cv3 in [0-0.5)
        # These would be frames 6, 9 (below 3) and 7, 10 (above 8), all with cv3 < 0.5
        self.assertEqual(len(fp_candidate), 4)
        candidate_frames = sorted([x[1] for x in fp_candidate])
        self.assertEqual(candidate_frames, [6, 7, 9, 10])

    def test_select_by_plumed_colvar_uniform(self):
        # Test uniform selection across CV range
        fp_rest_accurate, fp_candidate, fp_rest_failed, counter = _select_by_plumed_colvar(
            [str(self.task_dir)],  # modd_system_task
            0.01,  # f_trust_lo
            0.08,  # f_trust_hi
            0.01,  # v_trust_lo
            0.05,  # v_trust_hi
            5.0,   # colvar_lo
            10.0,  # colvar_hi
            1,     # colvar_column (second column in COLVAR, 0-indexed)
            None,  # cluster_cutoff
            "lammps",  # model_devi_engine
            0,     # model_devi_skip
            False, # model_devi_f_avg_relative
            False, # model_devi_merge_traj
            True,  # detailed_report_make_fp
            True,  # uniform_selection
        )
        
        # The fp_task_max parameter would be applied in the main code flow
        # So the above just produces all candidates with CV values
        # Check that candidate frames are returned
        self.assertGreater(len(fp_candidate), 0)
        
        # Check counter - should reflect all candidates
        self.assertEqual(counter["candidate"], len(fp_candidate))

if __name__ == "__main__":
    unittest.main() 
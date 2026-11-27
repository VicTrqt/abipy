"""Tests for dfpt_works module"""

import os
import shutil
import tempfile

from abipy import flowtk
from abipy.core.testing import AbipyTest
from abipy.flowtk.tasks import DdeTask, DdkTask, DkdeTask, DkdkTask, DteTask


class TestDfptWorks(AbipyTest):
    def test_nscfddkswork(self):
        """Testing NscfDdksWork."""
        scf_task = self.get_gsinput_si(as_task=True)
        work = flowtk.NscfDdksWork.from_scf_task(scf_task, ddk_ngkpt=[8, 8, 8], ddk_shiftk=[0, 0, 0], ddk_nband=10)
        assert len(work) == 4
        self.abivalidate_work(work)

    def test_elastic_work(self):
        """Testing ElasticWork."""
        scf_task = self.get_gsinput_si(as_task=True)
        scf_input = scf_task.input
        den_deps = {scf_task: "DEN"}
        tolerances = dict(nscf={"tolwfr": 1.0e-10}, ddk={"tolwfr": 1.0e-12}, strain={"tolvrs": 1.0e-10})
        work = flowtk.ElasticWork.from_scf_input(
            scf_input,
            with_relaxed_ion=True,
            with_piezo=True,
            with_dde=True,
            tolerances=tolerances,
            den_deps=den_deps,
            manager=None,
        )
        self.abivalidate_work(work)

        assert work[0].input["iscf"] == -2
        assert work[0].input["tolwfr"] == tolerances["nscf"]["tolwfr"]
        assert isinstance(work[0], flowtk.NscfTask)

        for task in work[1:]:
            assert task.input["kptopt"] == 2
        assert all(isinstance(task, flowtk.ElasticTask) for task in work[-6:])
        for task in work[-6:]:
            assert task.input["tolvrs"] == tolerances["strain"]["tolvrs"]


class TestDteWorkFullDfpt(AbipyTest):
    """Tests for DteWorkFullDfpt: full-DFPT (optdriver=5, usepead=0) third-order response."""

    MANAGER = """\
qadapters:
    - priority: 1
      queue:
        qtype: shell
        qname: localhost
      job:
        mpi_runner: mpirun
      limits:
        timelimit: 1:00:00
        max_cores: 4
      hardware:
        num_nodes: 1
        sockets_per_node: 1
        cores_per_socket: 4
        mem_per_node: 4 GB
"""

    def setUp(self):
        super().setUp()
        self.manager = flowtk.TaskManager.from_string(self.MANAGER)
        flowtk.set_user_config_taskmanager(self.manager)
        self.workdir = tempfile.mkdtemp()

    def tearDown(self):
        flowtk.set_user_config_taskmanager(None)
        shutil.rmtree(self.workdir)
        super().tearDown()

    def build_flow(self):
        """Build a NonLinearCoeffFullDfptFlow in self.workdir and return (flow, scf_task, work)."""
        scf_input = self.get_gsinput_si()
        # GS-SCF settings required by the full-DFPT chain (tnlo_6 dataset 1).
        scf_input.set_vars(kptopt=1, prtwf=1, prtden=1)
        flow = flowtk.NonLinearCoeffFullDfptFlow.from_scf_input(self.workdir, scf_input, manager=self.manager)
        scf_task = flow[0][0]
        work = flow[1]
        assert isinstance(work, flowtk.DteWorkFullDfpt)
        return flow, scf_task, work

    def test_invalid_scf_input(self):
        """A GS input that cannot produce the WFK/DEN on the IBZ must raise a clear error."""
        scf_input = self.get_gsinput_si()
        scf_input.set_vars(kptopt=1, prtwf=1, prtden=1)
        for bad_vars in ({"prtwf": 0}, {"prtden": 0}, {"kptopt": 3}, {"occopt": 7}):
            scf_input.set_vars(**bad_vars)
            with self.assertRaises(ValueError):
                flowtk.NonLinearCoeffFullDfptFlow.from_scf_input(self.workdir, scf_input, manager=self.manager)

    def test_from_scf_task(self):
        """Test the task graph and the inputs of DteWorkFullDfpt."""
        flow, scf_task, work = self.build_flow()
        # The SCF input is used as given by the user (no silent modification).
        assert scf_task.input["kptopt"] == 1
        assert scf_task.input["prtwf"] == 1 and scf_task.input["prtden"] == 1

        # 1 NSCF + 3 DDK + 3 DDE + 1 DKDK + 1 DKDE + 1 DTE
        assert len(work) == 10
        assert isinstance(work[0], flowtk.NscfTask)
        assert all(isinstance(t, DdkTask) for t in work[1:4])
        assert all(isinstance(t, DdeTask) for t in work[4:7])
        assert isinstance(work[7], DkdkTask)
        assert isinstance(work[8], DkdeTask)
        assert isinstance(work[9], DteTask)

        # Si has 4 valence bands: nband must be restricted to the valence, nbdbuf removed, occopt=1.
        for task in work:
            assert task.input["occopt"] == 1
            assert task.input["nband"] == 4
            assert "nbdbuf" not in task.input
            if not isinstance(task, flowtk.NscfTask):
                assert task.input["kptopt"] == 2

        nscf_task = work.nscf_task
        assert nscf_task.input["iscf"] == -2
        assert nscf_task.input["tolwfr"] == 1.0e-22
        assert nscf_task.input["prtwf"] == 1
        assert {tuple(d.exts) for d in nscf_task.deps} == {("WFK", "DEN")}

        for ddk_task in work.ddk_tasks:
            assert ddk_task.input["rfelfd"] == 2
            assert ddk_task.input["iscf"] == -3
            assert ddk_task.input["tolwfr"] == 1.0e-22
            assert {tuple(d.exts) for d in ddk_task.deps} == {("WFK",)}

        for dde_task in work.dde_tasks:
            assert dde_task.input["rfelfd"] == 3
            assert dde_task.input["prepanl"] == 1
            assert dde_task.input["tolvrs"] == 1.0e-12
            assert dde_task.input["prtwf"] == 1 and dde_task.input["prtden"] == 1
            exts = {tuple(d.exts) for d in dde_task.deps}
            assert ("WFK",) in exts and ("DDK",) in exts

        dkdk_task = work.dkdk_task
        assert dkdk_task.input["rf2_dkdk"] == 1
        assert dkdk_task.input["prepanl"] == 1
        assert dkdk_task.input["useylm"] == 1
        assert dkdk_task.input["tolwfr"] == 1.0e-22

        dkde_task = work.dkde_task
        assert dkde_task.input["rf2_dkde"] == 1
        assert dkde_task.input["prepanl"] == 1
        assert dkde_task.input["useylm"] == 1
        assert dkde_task.input["tolwfr"] == 1.0e-22
        # getdkdk equivalent in flow mode
        assert dkde_task.input["irddkdk"] == 1
        exts = {tuple(d.exts) for d in dkde_task.deps}
        assert ("WFK",) in exts and ("DDK",) in exts and ("1DEN", "DELFD") in exts and ("DKDK",) in exts

        dte_task = work.dte_task
        assert dte_task.input["optdriver"] == 5
        assert dte_task.input["usepead"] == 0
        assert dte_task.input["d3e_pert1_elfd"] == 1
        assert dte_task.input["d3e_pert2_elfd"] == 1
        assert dte_task.input["d3e_pert3_elfd"] == 1
        assert dte_task.input["d3e_pert1_phon"] == 0
        assert dte_task.input["d3e_pert2_phon"] == 0
        assert dte_task.input["d3e_pert3_phon"] == 0
        exts = {tuple(d.exts) for d in dte_task.deps}
        assert ("WFK",) in exts and ("DDK",) in exts and ("1WF", "1DEN") in exts and ("DKDE",) in exts

        # Connecting variables that will be added to the inputs at start time.
        cvars = {}
        for d in dte_task.deps:
            cvars.update(d.connecting_vars())
        assert cvars == {"irdwfk": 1, "irdddk": 1, "ird1wf": 1, "ird1den": 1, "irddkde": 1}
        cvars = {}
        for d in dkde_task.deps:
            cvars.update(d.connecting_vars())
        assert cvars == {"irdwfk": 1, "irdddk": 1, "ird1den": 1, "irddelfd": 1}

        if self.has_abinit("10.5"):
            self.abivalidate_work(work)

    def test_make_links(self):
        """Test the symlinks created by DfptTask.make_links for the DKDE and DTE tasks."""
        flow, scf_task, work = self.build_flow()

        # Si: natom=2. DDK pertcases are idir + 3*natom = 7,8,9.
        # ELFD (dde) pertcases are idir + 3*(natom+1) = 10,11,12.
        # DKDK pertcases are idir + 3*(natom+6) = 25..33.
        # DKDE pertcases are idir + 9 + 3*(natom+6) = 34..42.
        def touch(task, name):
            os.makedirs(task.outdir.path, exist_ok=True)
            path = task.outdir.path_in(name)
            with open(path, "w") as fh:
                fh.write("x")
            return path

        for task in work:
            os.makedirs(task.indir.path, exist_ok=True)

        touch(scf_task, "out_WFK")
        touch(scf_task, "out_DEN")
        touch(work.nscf_task, "out_WFK")
        for i, ddk_task in enumerate(work.ddk_tasks):
            touch(ddk_task, "out_1WF%d" % (7 + i))
            os.symlink(ddk_task.outdir.path_in("out_1WF%d" % (7 + i)), ddk_task.outdir.path_in("out_DDK"))
        for i, dde_task in enumerate(work.dde_tasks):
            touch(dde_task, "out_1WF%d" % (10 + i))
            touch(dde_task, "out_DEN%d" % (10 + i))
        for p in (25, 26, 30):
            touch(work.dkdk_task, "out_1WF%d" % p)
        for p in (34, 35, 42):
            touch(work.dkde_task, "out_1WF%d" % p)

        work.dkde_task.make_links()
        work.dte_task.make_links()

        def indir_names(task):
            return sorted(os.path.basename(p) for p in task.indir.list_filepaths())

        expected_dkde = sorted(
            ["in_WFK", "in_1WF7", "in_1WF8", "in_1WF9", "in_1WF10", "in_1WF11", "in_1WF12",
             "in_DEN10", "in_DEN11", "in_DEN12", "in_1WF25", "in_1WF26", "in_1WF30"]
        )
        expected_dte = sorted(
            ["in_WFK", "in_1WF7", "in_1WF8", "in_1WF9", "in_1WF10", "in_1WF11", "in_1WF12",
             "in_DEN10", "in_DEN11", "in_DEN12", "in_1WF34", "in_1WF35", "in_1WF42"]
        )
        assert indir_names(work.dkde_task) == expected_dkde
        assert indir_names(work.dte_task) == expected_dte

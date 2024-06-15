#!/usr/bin/env python
# Copyright 2014-2021 The PySCF Developers. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Author: Tianyu Zhu <zhutianyu1991@gmail.com>
#

'''
Periodic k-point spin-restricted random phase approximation
(direct RPA/dRPA in chemistry) with N^4 scaling

Method:
    Main routines are based on GW-AC method descirbed in:
    T. Zhu and G.K.-L. Chan, J. Chem. Theory. Comput. 17, 727-741 (2021)
    X. Ren et al., New J. Phys. 14, 053020 (2012)
'''

from functools import reduce
import time, h5py, os
import numpy
import numpy as np
from scipy.optimize import newton, least_squares

from pyscf import lib
from pyscf.lib import logger
from pyscf.ao2mo import _ao2mo
from pyscf.ao2mo.incore import _conc_mos
from pyscf.pbc import df, dft, scf
from pyscf.pbc.mp.kmp2 import get_nocc, get_nmo, get_frozen_mask
from pyscf import __config__

from fcdmft.gw.mol.gw_ac import _get_scaled_legendre_roots
from fcdmft.gw.pbc.krgw_ac import get_rho_response, get_rho_response_metal, \
                get_rho_response_gpu, get_rho_response_metal_gpu, \
                get_rho_response_head, get_rho_response_head_gpu, \
                get_rho_response_wing, get_rho_response_wing_gpu, \
                get_qij, get_qij_gpu
from mpi4py import MPI
import cupy as cp

rank = MPI.COMM_WORLD.Get_rank()
size = MPI.COMM_WORLD.Get_size()
comm = MPI.COMM_WORLD

einsum = lib.einsum

def kernel(rpa, mo_energy, mo_coeff, nw=None, verbose=logger.NOTE):
    """
    RPA correlation and total energy

    Returns:
        e_tot : RPA total energy
        e_hf : EXX energy
        e_corr : RPA correlation energy
    """
    mf = rpa._scf
    assert(rpa.frozen == 0)

    nkpts = rpa.nkpts
    nocc = rpa.nocc
    nmo = rpa.nmo
    nvir = nmo-nocc
    mo_occ = rpa.mo_occ

    # Compute HF exchange energy (EXX)
    if rpa.e_hf:
        if rank == 0:
            logger.info(rpa, 'Given EXX energy = %s', rpa.e_hf)
        e_hf = rpa.e_hf
    else:
        e_hf = 0
        if rank == 0:
            dm = mf.make_rdm1()
            rhf = scf.KRHF(rpa.mol, rpa.kpts, exxdiv=mf.exxdiv)
            if hasattr(rpa._scf, 'sigma'):
                rhf = scf.addons.smearing_(rhf, sigma=rpa._scf.sigma, method="fermi")
            rhf.with_df = mf.with_df
            rhf.with_df._cderi = mf.with_df._cderi
            e_hf = rhf.energy_elec(dm)[0]
            e_hf += mf.energy_nuc()
            logger.info(rpa, 'Computed EXX energy = %s', e_hf)
        comm.Barrier()
        e_hf = comm.bcast(e_hf, root=0)

    # check metal
    mo_occ_1d = np.array(mo_occ).reshape(-1)
    is_metal = False
    if np.linalg.norm(np.abs(mo_occ_1d - 1.) - 1.) > 1e-5:
        is_metal = True
        if rpa.fc:
            if rank == 0:
                logger.warn(rpa, 'FC not available for metals - setting rpa.fc to False')
            rpa.fc = False

    # Grids for integration on imaginary axis
    freqs,wts = _get_scaled_legendre_roots(nw)

    # Compute RPA correlation energy
    e_corr = get_rpa_ecorr(rpa, freqs, wts)

    # Compute totol energy
    e_tot = e_hf + e_corr

    if rank == 0 and rpa.verbose >= logger.DEBUG:
        logger.debug(rpa, '  RPA total energy = %s', e_tot)
        logger.debug(rpa, '  EXX energy = %s, RPA corr energy = %s', e_hf, e_corr)

    return e_tot, e_hf, e_corr

def get_rpa_ecorr(rpa, freqs, wts, max_memory=8000):
    '''
    Compute RPA correlation energy
    '''
    
    # Each GPU is assiged to a MPI process
    num_gpus = cp.cuda.runtime.getDeviceCount()
    if rank == 0:
        logger.info(rpa, "Number of available GPUs: %d "%(num_gpus))
    gpu_id = rank % num_gpus
    gpu_properties = cp.cuda.runtime.getDeviceProperties(gpu_id)
    gpu_name = gpu_properties['name']
    gpu_uuid = gpu_properties['uuid']
    device = cp.cuda.Device(gpu_id)
    device.use()
    
    mo_energy = cp.asarray(rpa._scf.mo_energy)
    mo_coeff = cp.asarray(rpa._scf.mo_coeff)
    nocc = rpa.nocc
    nmo = rpa.nmo
    nkpts = rpa.nkpts
    kpts = rpa.kpts
    nw = len(freqs)
    mydf = rpa.with_df
    mo_occ = rpa.mo_occ

    # possible kpts shift center
    kscaled = rpa.mol.get_scaled_kpts(kpts)
    kscaled -= kscaled[0]

    is_metal = False
    mo_occ_1d = np.array(mo_occ).reshape(-1)
    if np.linalg.norm(np.abs(mo_occ_1d - 1.) - 1.) > 1e-5:
        is_metal = True
        if rpa.fc:
            if rank == 0:
                logger.warn(rpa, 'FC not available for metals - setting rpa.fc to False')
            rpa.fc = False

    segsize = nkpts // size
    if rank >= size-(nkpts-segsize*size):
        start = rank * segsize + rank-(size-(nkpts-segsize*size))
        stop = min(nkpts, start+segsize+1)
    else:
        start = rank * segsize
        stop = min(nkpts, start+segsize)

    if rpa.fc:
        # Set up q mesh for q->0 finite size correction
        if not rpa.fc_grid:
            q_pts = np.array([1e-3,0,0]).reshape(1,3)
        else:
            Nq = 4
            q_pts = np.zeros((Nq**3-1,3))
            for i in range(Nq):
                for j in range(Nq):
                    for k in range(Nq):
                        if i == 0 and j == 0 and k== 0:
                            continue
                        else:
                            q_pts[i*Nq**2+j*Nq+k-1,0] = k * 5e-4
                            q_pts[i*Nq**2+j*Nq+k-1,1] = j * 5e-4
                            q_pts[i*Nq**2+j*Nq+k-1,2] = i * 5e-4
        nq_pts = len(q_pts)
        q_abs = cp.asarray(rpa.mol.get_abs_kpts(q_pts))

        # Get qij = 1/sqrt(Omega) * < psi_{ik} | e^{iqr} | psi_{ak-q} > at q: (nkpts, nocc, nvir)
        qij = cp.zeros((nq_pts, nkpts, nocc, nmo - nocc),dtype=np.complex128)
        for k in range(nq_pts):
            qij[k] = get_qij_gpu(rpa, q_abs[k], mo_coeff, start, stop)
            
    comm.Barrier()
    e_corr = 0j
    for kL in range(nkpts):
        if rank == 0:
            cput0 = (time.process_time(), time.perf_counter())

        # Lij: (ki, L, i, j) for looping every kL
        Lij = {}
        # kidx: save kj that conserves with kL and ki (-ki+kj+kL=G)
        # kidx_r: save ki that conserves with kL and kj (-ki+kj+kL=G)
        kidx = np.zeros((nkpts),dtype=np.int64)
        kidx_r = np.zeros((nkpts),dtype=np.int64)
        naux = None
            
        for i in range(start,stop):
            logger.debug(rpa, "    Read and Tranform Lpq (kL: %s / %s, ki: %s) @ rank %d"%(kL+1, nkpts, i, rank))
            kpti = kpts[i]
            for j, kptj in enumerate(kpts):
                # Find (ki,kj) that satisfies momentum conservation with kL
                kconserv = -kscaled[i] + kscaled[j] + kscaled[kL]
                is_kconserv = np.linalg.norm(np.round(kconserv) - kconserv) < 1e-12
                if is_kconserv:
                    kidx[i] = j
                    kidx_r[j] = i
                    Lij_out = None
                    # Read (L|pq) and ao2mo transform to (L|ij)
                    Lpq = []
                    for LpqR, LpqI, sign in mydf.sr_loop([kpti, kptj], max_memory=0.1*rpa._scf.max_memory, compact=False):
                        Lpq.append(LpqR+LpqI*1.0j)
                    # support uneqaul naux on different k points
                    Lpq = np.vstack(Lpq).reshape(-1,nmo**2)
                    tao = []
                    ao_loc = None
                    moij, ijslice = _conc_mos(mo_coeff[i].get(), mo_coeff[j].get())[2:]
                    Lij_out = _ao2mo.r_e2(Lpq, moij, ijslice, tao, ao_loc, out=Lij_out)           
                    Lij[f"k{i}"] = Lij_out.reshape(-1,nmo,nmo)
                    naux = Lij_out.shape[0]
        comm.Barrier()
        if rank == 0:
            logger.timer(rpa, "Read and Tranform Lpq at kL: %s / %s" % (kL+1, nkpts), *cput0)
            
        if kL == 0:
            if rank == 0:
                cput0 = (time.process_time(), time.perf_counter())
                logger.info(rpa, "  Compute the e_corr contribution at kL: %s / %s"%(kL+1, nkpts))

            for w in range(nw):
                # body polarizability
                if is_metal:
                    Pi = get_rho_response_metal_gpu(rpa, freqs[w], mo_energy, mo_occ, Lij, kidx, start, stop)
                else:
                    Pi = get_rho_response_gpu(rpa, freqs[w], mo_energy, Lij, kidx, start, stop)

                if rpa.fc:
                    for iq in range(nq_pts):
                        # head Pi_00
                        Pi_00 = get_rho_response_head_gpu(rpa, freqs[w], mo_energy, qij[iq])
                        Pi_00 = 4. * cp.pi/cp.linalg.norm(q_abs[iq])**2 * Pi_00
                        # wings Pi_P0
                        Pi_P0 = get_rho_response_wing_gpu(rpa, freqs[w], mo_energy, Lij, qij[iq], start, stop)
                        Pi_P0 = cp.sqrt(4.*cp.pi) / cp.linalg.norm(q_abs[iq]) * Pi_P0
                        comm.Barrier()
                        
                        if rank == 0:
                            summed_Pi = np.zeros((naux, naux), dtype=np.complex128)
                            summed_Pi_P0 = np.zeros(naux, dtype=np.complex128)
                        else:
                            summed_Pi = None
                            summed_Pi_P0 = None
                            
                        comm.Reduce(Pi.get(), summed_Pi, op=MPI.SUM, root=0)
                        comm.Reduce(Pi_P0.get(), summed_Pi_P0, op=MPI.SUM, root=0)
                        Pi_00_gather = comm.gather(Pi_00.get())
    
                        if rank == 0:
                            Pi_00 = np.sum(Pi_00_gather)
                            Pi_P0 = cp.asarray(summed_Pi_P0)
                            Pi = cp.asarray(summed_Pi)
                            
                            # assemble Pi
                            Pi_fc = cp.zeros((naux+1,naux+1),dtype=Pi.dtype)
                            Pi_fc[0,0] = Pi_00
                            Pi_fc[0,1:] = Pi_P0.conj()
                            Pi_fc[1:,0] = Pi_P0
                            Pi_fc[1:,1:] = Pi
                        
                            ec_w = cp.log(cp.linalg.det(cp.eye(1+naux) - Pi_fc))
                            ec_w += cp.trace(Pi_fc)
                            e_corr += 1./(2.*cp.pi) * 1./nkpts * 1./nq_pts * ec_w * wts[w]
                else:
                    comm.Barrier()
                    if rank == 0:
                        summed_Pi = np.zeros((naux, naux), dtype=np.complex128)
                    else:
                        summed_Pi = None
                    comm.Reduce(Pi.get(), summed_Pi, op=MPI.SUM, root=0)

                    if rank == 0:
                        Pi = cp.asarray(summed_Pi)
                        ec_w = cp.log(cp.linalg.det(cp.eye(naux) - Pi))
                        ec_w += cp.trace(Pi)
                        e_corr += 1./(2.*cp.pi) * 1./nkpts * ec_w * wts[w]
        else:
            for w in range(nw):
                if is_metal:
                    Pi = get_rho_response_metal_gpu(rpa, freqs[w], mo_energy, mo_occ, Lij, kidx, start, stop)
                else:
                    Pi = get_rho_response_gpu(rpa, freqs[w], mo_energy, Lij, kidx, start, stop)
                    
                # comm.Barrier()
                if rank == 0:
                    summed_Pi = np.zeros((naux, naux), dtype=np.complex128)
                else:
                    summed_Pi = None
                comm.Reduce(Pi.get(), summed_Pi, op=MPI.SUM, root=0)
                
                if rank == 0:
                    Pi = cp.asarray(summed_Pi)
                    ec_w = cp.log(cp.linalg.det(cp.eye(naux) - Pi))
                    ec_w += cp.trace(Pi)
                    e_corr += 1./(2.*cp.pi) * 1./nkpts * ec_w * wts[w]
        comm.Barrier()
        e_corr = comm.bcast(e_corr,root=0)
        if rank == 0:
            logger.debug(rpa, "  kL = %s / %s, RPA corr energy = %s"%(kL+1, nkpts, e_corr.real))
            logger.timer(rpa, "the corr energy contribution at kL: %s / %s" % (kL+1, nkpts), *cput0)

    return e_corr.real

class KRPA(lib.StreamObject):

    # applying finite size corrections or not
    fc = getattr(__config__, 'krpa_KRPA_fc', False)

    def __init__(self, mf, frozen=0):
        self.mol = mf.mol
        self._scf = mf
        self.verbose = self.mol.verbose
        self.stdout = self.mol.stdout
        self.max_memory = mf.max_memory

        #TODO: implement frozen orbs
        if frozen > 0:
            raise NotImplementedError
        self.frozen = frozen

        # DF-KGW must use GDF integrals
        if getattr(mf, 'with_df', None):
            self.with_df = mf.with_df
        else:
            raise NotImplementedError
        self._keys.update(['with_df'])

##################################################
# don't modify the following attributes, they are not input options
        self._nocc = None
        self._nmo = None
        self.kpts = mf.kpts
        self.nkpts = len(self.kpts)
        self.mo_energy = mf.mo_energy
        self.mo_coeff = mf.mo_coeff
        self.mo_occ = mf.mo_occ
        self.e_corr = None
        self.e_hf = None
        self.e_tot = None
        self.fc_grid = False

        keys = set(('fc'))
        self._keys = set(self.__dict__.keys()).union(keys)

    def dump_flags(self):
        log = logger.Logger(self.stdout, self.verbose)
        log.info('')
        log.info('******** %s ********', self.__class__)
        log.info('method = %s', self.__class__.__name__)
        nocc = self.nocc
        nvir = self.nmo - nocc
        nkpts = self.nkpts
        log.info('RPA nocc = %d, nvir = %d, nkpts = %d', nocc, nvir, nkpts)
        if self.frozen > 0:
            log.info('frozen orbitals = %s', str(self.frozen))
        logger.info(self, 'RPA finite size corrections = %s', self.fc)
        return self

    @property
    def nocc(self):
        return self.mol.nelectron // 2
    @nocc.setter
    def nocc(self, n):
        self._nocc = n

    @property
    def nmo(self):
        return len(self._scf.mo_energy[0])
    @nmo.setter
    def nmo(self, n):
        self._nmo = n

    get_nocc = get_nocc
    get_nmo = get_nmo
    get_frozen_mask = get_frozen_mask

    def kernel(self, mo_energy=None, mo_coeff=None, nw=40):
        """
        Args:
            mo_energy : 2D array (nkpts, nmo), mean-field mo energy
            mo_coeff : 3D array (nkpts, nmo, nmo), mean-field mo coefficient
            nw: integer, grid number

        Returns:
            self.e_tot : RPA total eenrgy
            self.e_hf : EXX energy
            self.e_corr : RPA correlation energy
        """
        if mo_coeff is None:
            mo_coeff = np.array(self._scf.mo_coeff)
        if mo_energy is None:
            mo_energy = np.array(self._scf.mo_energy)

        nmo = self.nmo
        naux = self.with_df.get_naoaux()
        nkpts = self.nkpts
        mem_incore = (2 * nkpts * nmo**2 * naux) * 16/1e6
        mem_now = lib.current_memory()[0]
        if (mem_incore + mem_now > 0.99*self.max_memory):
            if rank == 0:
                logger.warn(self, 'Memory may not be enough!')
            raise NotImplementedError

        cput0 = (time.process_time(), time.perf_counter())
        if rank == 0:
            self.dump_flags()
        self.e_tot, self.e_hf, self.e_corr = \
                kernel(self, mo_energy, mo_coeff, nw=nw, verbose=self.verbose)

        if rank == 0:
            logger.timer(self, 'RPA', *cput0)
        return self.e_tot, self.e_hf, self.e_corr

if __name__ == '__main__':
    from pyscf.pbc import gto, dft, scf
    from pyscf.pbc.lib import chkfile
    import os
    # This test takes a few minutes
    # Test on diamond
    cell = gto.Cell()
    cell.build(unit = 'angstrom',
            a = '''
                0.000000     1.783500     1.783500
                1.783500     0.000000     1.783500
                1.783500     1.783500     0.000000
            ''',
            atom = 'C 1.337625 1.337625 1.337625; C 2.229375 2.229375 2.229375',
            dimension = 3,
            max_memory = 12000,
            verbose = 5,
            pseudo = 'gth-pbe',
            basis='gth-dzv',
            precision=1e-12)

    kpts = cell.make_kpts([3,1,1],scaled_center=[0,0,0])
    gdf = df.GDF(cell, kpts)
    gdf_fname = 'gdf_ints_311.h5'
    gdf._cderi_to_save = gdf_fname
    if not os.path.isfile(gdf_fname):
        gdf.build()

    chkfname = 'diamond_311.chk'
    if os.path.isfile(chkfname):
        kmf = scf.KRHF(cell, kpts)
        kmf.with_df = gdf
        kmf.with_df._cderi = gdf_fname
        data = chkfile.load(chkfname, 'scf')
        kmf.__dict__.update(data)
    else:
        kmf = scf.KRHF(cell, kpts)
        kmf.with_df = gdf
        kmf.with_df._cderi = gdf_fname
        kmf.conv_tol = 1e-12
        kmf.chkfile = chkfname
        kmf.kernel()

    rpa = KRPA(kmf)
    rpa.fc = False
    rpa.kernel()
    if rank == 0:
        print(rpa.e_tot, rpa.e_corr)
    assert(abs(rpa.e_corr- -0.1852772037535004) < 1e-6)
    assert(abs(rpa.e_tot- -10.694392044197565) < 1e-6)

    # with finite size corrections
    rpa.fc = True
    rpa.kernel()
    if rank == 0:
        print(rpa.e_tot, rpa.e_corr)
    assert(abs(rpa.e_corr- -0.20723563246774715) < 1e-6)
    assert(abs(rpa.e_tot- -10.716350472911811) < 1e-6)

    # Test on Na (metallic)
    cell = gto.Cell()
    cell.build(unit = 'angstrom',
           a = '''
           -2.11250000000000   2.11250000000000   2.11250000000000
           2.11250000000000  -2.11250000000000   2.11250000000000
           2.11250000000000   2.11250000000000  -2.11250000000000
           ''',
           atom = '''
           Na   0.00000   0.00000   0.00000
           ''',
           dimension = 3,
           max_memory = 126000,
           verbose = 5,
           pseudo = 'gth-pade',
           basis='gth-dzvp-molopt-sr',
           precision=1e-10)

    kpts = cell.make_kpts([2,2,1],scaled_center=[0,0,0])
    gdf = df.GDF(cell, kpts)
    gdf_fname = 'gdf_ints_221.h5'
    gdf._cderi_to_save = gdf_fname
    if not os.path.isfile(gdf_fname):
        gdf.build()

    chkfname = 'na_221.chk'
    if os.path.isfile(chkfname):
        kmf = dft.KRKS(cell, kpts)
        kmf = scf.addons.smearing_(kmf, sigma=5e-3, method="fermi")
        kmf.xc = 'lda'
        kmf.with_df = gdf
        kmf.with_df._cderi = gdf_fname
        data = chkfile.load(chkfname, 'scf')
        kmf.__dict__.update(data)
    else:
        kmf = dft.KRKS(cell, kpts)
        kmf = scf.addons.smearing_(kmf, sigma=5e-3, method="fermi")
        kmf.xc = 'lda'
        kmf.with_df = gdf
        kmf.with_df._cderi = gdf_fname
        kmf.conv_tol = 1e-12
        kmf.chkfile = chkfname
        kmf.kernel()

    rpa = KRPA(kmf)
    rpa.kernel()
    assert(abs(rpa.e_corr- -0.04031792880477294) < 1e-6)
    assert(abs(rpa.e_tot- -47.60693294927457) < 1e-6)

import numpy as np
import librosa
import plca.plca as plca

EPS = np.finfo(np.float32).eps

def compute_divergence(X, Y):
    #return np.sum(X * np.log(X / Y) - X + Y)
    div = np.sum(Y - X * np.log(Y))
    if div < 0 : print "WARNING: Negative divergance :("
    return div


def normalize(A, axis=None):
    Ashape = A.shape
    try:
        norm = A.sum(axis) + EPS
    except TypeError:
        norm = A.copy()
        for ax in reversed(sorted(axis)):
            norm = norm.sum(ax)
        norm += EPS
    if axis:
        nshape = np.array(Ashape)
        nshape[axis] = 1
        norm.shape = nshape
    return A / norm


class AdaptiveDictionary(object):
    """
    A dictionary that updates itself based on parameters of the current frame.
    """

    def __init__(self, init_bases, instr_names, instrs_to_cols, alpha=0.5):
        self.P_fZ = init_bases
        self.instr_names = instr_names
        self.num_instrs = len(instr_names)
        self._initialized = False
        self.instrs_to_cols = instrs_to_cols
        self.V_prev = None
        self.Pn_Z_prev = None
        self.Pn_Zf_prev = None
        self.alpha = alpha


    # TODO: Add computing KL divergence

    @property
    def dict(self):
        return dict(zip(self.instr_names, [self.P_fZ[:, i[0]:i[1]] for i in [self.instrs_to_cols[instr] for instr in self.instr_names ] ]))

    @property
    def matrix(self):
        return self.P_fZ


    def update_dict(self, V, Pn_Z, need_update=True, num_iters=25):
        assert num_iters >= 1, "Must have positive number of iterations!"
        if not self._initialized or not need_update:
            self._initialized = True
            # If first iteration, we can't do anything,
            # or if we didn't need to update the dictionary,
            # move history forward
            self.V_prev = V
            self.Pn_Z_prev = Pn_Z
            # May not strictly need to be computed every time
            self.Pn_Zf_prev = self.compute_Pn_Zf(self.P_fZ, Pn_Z)
            return self.P_fZ, Pn_Z

        Pn_fZ = None
        P_fZ = None

        # Compute KL divergence here at some point

        print "Starting adaptive dictionary update..."
        for i in xrange(num_iters):
            print "-- Iteration %d" % i

            Pn_Zf, Pn_Zf_prev = self.E_step(self.P_fZ, Pn_Z, self.Pn_Z_prev)

            P_fZ, Pn_Z = self.M_step(V, self.V_prev, Pn_Zf, Pn_Zf_prev, self.alpha)

        self.V_prev = V
        self.Pn_fZ_prev = Pn_fZ
        self.Pn_Z_prev = Pn_Z
        self.P_fZ = P_fZ

        return P_fZ, Pn_Z

    def compute_Pn_Zf(self, P_fZ, Pn_Z, Pn_Z_prev=None):
        if Pn_Z_prev is not None:
            T = Pn_Z.shape[1] + Pn_Z_prev.shape[1]
        else:
            T = Pn_Z.shape[1]
        denom = np.zeros((P_fZ.shape[0], T))
        for instr in self.instr_names:
            i_start, i_end = self.instrs_to_cols[instr]
            Pn_z = Pn_Z[i_start:i_end, :]
            if Pn_Z_prev is not None:
                Pn_z = np.concatenate((Pn_Z_prev[i_start:i_end, :], Pn_z), axis=1)
            denom += np.dot(P_fZ[:, i_start:i_end], Pn_z)

        Pn_Zf = np.zeros((P_fZ.shape[1], P_fZ.shape[0], T), dtype=np.float32)
        for instr in self.instr_names:
            i_start, i_end = self.instrs_to_cols[instr]
            for i in range(i_start, i_end):
                P_fz = P_fZ[:, i]
                Pn_z = Pn_Z[i, :]
                if Pn_Z_prev is not None:
                    Pn_z = np.concatenate((Pn_Z_prev[i, :], Pn_z), axis=1)

                Pn_Zf[i,:,:] = np.outer(P_fz, Pn_z) / (denom + EPS)

        if Pn_Z_prev is not None:
            return Pn_Zf[:, :, Pn_Z_prev.shape[1]:], Pn_Zf[:, :, :Pn_Z_prev.shape[1]]
        else:
            return Pn_Zf

    def E_step(self, P_fZ, Pn_Z, Pn_Z_prev):
        Pn_Zf = self.compute_Pn_Zf(P_fZ, Pn_Z, Pn_Z_prev)
        return Pn_Zf

    def M_step(self, V, V_prev, Pn_Zf, Pn_Zf_prev, alpha):
        assert Pn_Zf.shape[0] == self.P_fZ.shape[1]
        V_comb = np.concatenate((V_prev, V), axis=1)
        phi_fZ = np.zeros((Pn_Zf.shape[1], Pn_Zf.shape[0]), dtype=np.float32)
        phin_Z = np.zeros((Pn_Zf.shape[0], Pn_Zf.shape[2]), dtype=np.float32)
        beta = Pn_Zf_prev.shape[2]

        phi_fz_tot = 0
        phin_z_tot = 0

        for instr_idx in range(self.num_instrs):
            instr = self.instr_names[instr_idx]
            i_start, i_end = self.instrs_to_cols[instr]
            for i in range(i_start, i_end):
                phi_fZ[:,i] = 1.0/V.shape[1] * (V*Pn_Zf[i,:,:]).sum(axis=1) \
                                + alpha/float(beta) * (V_prev * Pn_Zf_prev[i,:,:]).sum(axis=1)
                phin_Z[i,:] = (V*Pn_Zf[i,:,:]).sum(axis=0)

            phi_fZ[:,i_start:i_end] /= phi_fZ[:,i_start:i_end].sum(axis=0) + EPS

        P_fZ = phi_fZ
        Pn_Z = phin_Z / (phin_Z.sum(axis=0) + EPS)

        return P_fZ, Pn_Z


class AdaptiveSourceSeparator(object):
    """
    A source separation algorithm that adaptively alters the template dictionary.
    """

    def __init__(self, templates, instr_names, instrs_to_cols, threshold, fs, segment_size, adapt=False):
        self.adaptive_dict = AdaptiveDictionary(templates, instr_names, instrs_to_cols)
        self.segment_size = segment_size
        self.instr_names = instr_names
        self.instrs_to_cols = instrs_to_cols
        self.threshold = threshold
        self.fs = fs
        self.adapt = adapt

    def process_segment(self, mixed_segment):
        seg_size = len(mixed_segment)
        pad_size = 0
        if seg_size != self.segment_size:
            pad_size += (seg_size + self.segment_size - 1)/self.segment_size * self.segment_size - seg_size

        print "Adding padding..."
        pad_size += 2048
        mixed_segment = np.pad(mixed_segment, pad_size/2, mode='constant')
        if seg_size % 2 == 1:
            mixed_segment = np.append(mixed_segment, np.array([0]))

        print "Computing STFT..."
        S = librosa.stft(mixed_segment, n_fft=2048, hop_length=512)

        mag = np.abs(S)
        phase = np.exp(1j * np.angle(S))

        Pn_F = mag / mag.sum()

        print "Computing preliminary PLCA..."
        _, Z, H, norm, _, _= plca.PLCA.analyze(mag, self.adaptive_dict.matrix.shape[1], initW=self.adaptive_dict.matrix, updateW=False, minpruneiter=500)


        print "Computing divergence..."
        Pn_F_est = np.dot(self.adaptive_dict.matrix, np.dot(np.diag(Z), H))

        div = compute_divergence(Pn_F, Pn_F_est)
        print "     -- KL Divergence: %f" % div

        sources = {}
        if not self.adapt or (div < self.threshold):
            print "Skip updating dictionary..."
            if self.adapt:
                self.adaptive_dict.update_dict(mag, np.dot(np.diag(Z), H) / (EPS + mag.sum(axis=0)), False)
            ZH = np.dot(np.diag(Z), H)
        else:
            print "Updating dictionary..."
            P_fZ, Pn_Z = self.adaptive_dict.update_dict(Pn_F, np.dot(np.diag(Z), H), True)
            ZH = Pn_Z


        print "Recovering sources..."
        W = self.adaptive_dict.matrix
        for instr in self.instr_names:
            i_start, i_end = self.instrs_to_cols[instr]
            W_i = W[:,i_start:i_end]
            ZH_i = ZH[i_start:i_end,:]
            mag_i =  norm * np.dot(W_i, ZH_i)

            S_i = mag_i * phase

            assert S_i.shape == S.shape
            source = librosa.istft(S_i, hop_length=512)
            source = source[pad_size/2:-pad_size/2]

            if seg_size % 2 == 1:
                source = source[:-1]

            if len(source) != seg_size:
                print "WARNING, size differs from original"

            sources[instr] = source

        return sources, div

